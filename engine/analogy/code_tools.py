"""Read-only, episode-scoped access to immutable candidate source code.

No candidate module is imported or executed. Node IDs are a frozen allow-list, never
paths, and runtime source is checked against its registered SHA256 before use.
"""
from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import json
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


@dataclass
class CodeToolOptions:
    enabled: bool = True
    max_calls: int = 10
    max_total_chars: int = 120000
    max_read_chars: int = 20000
    default_read_lines: int = 200
    max_read_lines: int = 400
    max_diff_chars: int = 20000

    def __post_init__(self):
        for f in fields(self):
            if f.name != "enabled" and int(getattr(self, f.name)) < 1:
                raise ValueError(f"code_tools.{f.name} must be positive")
        if self.default_read_lines > self.max_read_lines:
            raise ValueError("default_read_lines exceeds max_read_lines")


def options_from_config(config=None):
    if isinstance(config, CodeToolOptions):
        return config
    raw = _get(config, "code_tools", config)
    return CodeToolOptions(**{f.name: _get(raw, f.name, f.default) for f in fields(CodeToolOptions)})


def is_completed(node):
    """Generated or queued nodes must not enter the source allow-list."""
    if _get(node, "stage") == "root":
        return False
    status = _get(node, "execution_status")
    if status in {"queued", "running", "pending"}:
        return False
    return (status in {"completed", "budget_exhausted", "failed", "timeout", "protocol_error"}
            or _get(node, "exec_time") is not None or bool(_get(node, "finish_time")))


def completed_branch_nodes(agent, current, limit=10):
    branch = _get(agent, "branch_all_nodes", {}).get(_get(current, "branch_id"), []) or []
    return sorted((n for n in branch if is_completed(n)), key=lambda n: _get(n, "ctime", 0) or 0)[-limit:]


def _source_index(source):
    lines = source.splitlines()
    result = {"status": "ok", "imports": [], "symbols": [], "constants": [],
              "runtime_bindings": [], "top_level": []}
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as exc:
        return {**result, "status": "syntax_error", "error": str(exc), "line_reading_available": True}

    def entry(node, **extra):
        return {"start_line": node.lineno, "end_line": getattr(node, "end_lineno", node.lineno), **extra}

    def safe_literal(node):
        # ast.literal_eval only on small literal syntax: no calls, attribute access,
        # names, huge containers, or expression evaluation.
        if len(list(ast.walk(node))) > 80:
            return None
        try:
            value = ast.literal_eval(node)
            rendered = json.dumps(value, ensure_ascii=False, default=str, allow_nan=False)
            return json.loads(rendered) if len(rendered) <= 500 else None
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return None

    def visit(body, prefix=""):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}.{node.name}" if prefix else node.name
                # Exact first source line is evidence; end_line indexes the body but
                # does NOT claim its unreturned contents were read.
                result["symbols"].append(entry(node, symbol=name, kind=type(node).__name__,
                                               signature=lines[node.lineno - 1].strip()[:1000]))
                visit(node.body, name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                result["imports"].append(entry(node, source=ast.get_source_segment(source, node)))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = safe_literal(node.value) if node.value is not None else None
                if value is not None:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            result["constants"].append(entry(node, name=f"{prefix}.{target.id}" if prefix else target.id,
                                                            value=value, scope=prefix or "module"))
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == "bind":
                    bound = {kw.arg: ast.unparse(kw.value)[:300] for kw in child.keywords
                             if kw.arg and isinstance(kw.value, (ast.Name, ast.Attribute))}
                    item = entry(child, receiver=ast.unparse(child.func.value)[:200], callbacks=bound,
                                 note="syntactic binding; runtime identity not inferred")
                    if item not in result["runtime_bindings"]:
                        result["runtime_bindings"].append(item)
    visit(tree.body)
    result["top_level"] = [entry(n, kind=type(n).__name__) for n in tree.body
                           if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                                 ast.Import, ast.ImportFrom))]
    return result


class CodeReadingSession:
    def __init__(self, nodes, current_id, *, workspace=None, runtime_enabled=False,
                 run_id="unknown", options=None):
        self.options = options_from_config(options)
        self.current_id = str(current_id)
        self.run_id = str(run_id)
        self.workspace = Path(workspace).resolve() if workspace else None
        self.runtime_enabled = bool(runtime_enabled)
        self.sources = {}
        self.ledger = []
        self.anchors = []
        self.calls_used = 0
        self.chars_used = 0
        self._budget_stopped = False
        # Freeze both metadata and source now, not on the first later tool call.
        for node in nodes:
            node_id = str(_get(node, "id", ""))
            if node_id in self.sources:
                continue
            parent = _get(node, "parent")
            record = {"node_id": node_id, "stage": _get(node, "stage"),
                      "parent_id": _get(parent, "id") if parent is not None else _get(node, "parent_id"),
                      "run_id": self.run_id, "available": False}
            self.sources[node_id] = record
            if not re.fullmatch(r"[A-Za-z0-9_-]+", node_id):
                record["reason"] = "invalid_node_id"
                continue
            if not is_completed(node):
                record["reason"] = "candidate_not_completed"
                continue
            try:
                if self.runtime_enabled:
                    if self.workspace is None:
                        raise ValueError("runtime_workspace_unavailable")
                    directory = self.workspace / "candidate_results" / "candidates" / node_id
                    # Reject symlink escapes even if a registered name is well formed.
                    directory.resolve().relative_to(self.workspace / "candidate_results" / "candidates")
                    source_path = directory / "solution.py"
                    meta_path = directory / "candidate.json"
                    source_path.resolve().relative_to(directory.resolve())
                    meta_path.resolve().relative_to(directory.resolve())
                    if source_path.stat().st_size > 4_000_000 or meta_path.stat().st_size > 4_000_000:
                        raise ValueError("registered_source_or_metadata_too_large")
                    registered = json.loads(meta_path.read_text())
                    source_bytes = source_path.read_bytes()
                    digest = hashlib.sha256(source_bytes).hexdigest()
                    if registered.get("node_id") != node_id or registered.get("source_sha256") != digest:
                        raise ValueError("registered_source_hash_or_node_mismatch")
                    source = source_bytes.decode("utf-8")
                    record.update(source_sha256=digest, origin="registered_solution", source_reference=str(source_path))
                else:
                    source = str(_get(node, "code", "") or "")
                    if len(source.encode("utf-8")) > 4_000_000:
                        raise ValueError("source_too_large")
                    record.update(source_sha256=hashlib.sha256(source.encode()).hexdigest(), origin="search_node_snapshot")
                if not source:
                    raise ValueError("empty_source")
                record.update(available=True, source=source, lines=source.splitlines(), index=_source_index(source))
            except (OSError, ValueError, UnicodeError, TypeError) as exc:
                record["reason"] = str(exc)

    @classmethod
    def from_search(cls, agent, parent_node, options=None, allowed_nodes=None):
        cfg = _get(agent, "cfg")
        acfg = _get(cfg, "analogy")
        if allowed_nodes is None:
            context_cfg = _get(acfg, "context")
            allowed_nodes = completed_branch_nodes(agent, parent_node, int(_get(context_cfg, "trajectory_nodes", 10)))
        nodes = [parent_node, *allowed_nodes]
        direct_parent = _get(parent_node, "parent")
        if direct_parent is not None:
            nodes.append(direct_parent)
        workspace = _get(cfg, "workspace_dir")
        return cls(nodes, _get(parent_node, "id"), workspace=workspace,
                   runtime_enabled=_get(_get(cfg, "candidate_runtime"), "enabled", False),
                   run_id=Path(str(workspace)).parent.name if workspace else _get(cfg, "exp_name", "unknown"),
                   options=options if options is not None else acfg)

    def tools(self):
        nullable_id = {"type": ["string", "null"], "description": "null means the current executed candidate"}
        nullable_int = {"type": ["integer", "null"], "minimum": 1}
        definitions = [
            ("candidate_code_index", "Index allowed candidate source without executing it. Use read_candidate_code to inspect method bodies.",
             {"node_id": nullable_id}),
            ("read_candidate_code", "Read numbered candidate source. Use symbol OR start_line/end_line. Continue with returned start_line/start_column; null uses defaults.",
             {"node_id": nullable_id, "symbol": {"type": ["string", "null"]}, "start_line": nullable_int,
              "end_line": nullable_int, "start_column": {"type": ["integer", "null"], "minimum": 0},
              "max_lines": nullable_int}),
            ("diff_candidate_code", "Read actual source diff; default current candidate versus its direct parent. offset resumes diff lines; symbol optionally limits one function/class.",
             {"base_node_id": {"type": ["string", "null"]}, "target_node_id": nullable_id,
              "symbol": {"type": ["string", "null"]}, "offset": {"type": ["integer", "null"], "minimum": 0},
              "offset_column": {"type": ["integer", "null"], "minimum": 0}}),
        ]
        return [{"type": "function", "function": {"name": name, "description": description, "strict": True,
                 "parameters": {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}}}
                for name, description, props in definitions]

    def _metadata(self, record):
        return {k: record.get(k) for k in ("node_id", "parent_id", "stage", "run_id", "source_sha256", "origin")}

    def _lookup(self, node_id=None):
        node_id = self.current_id if node_id is None or node_id == "current" else str(node_id)
        if node_id == "parent" and node_id not in self.sources:
            node_id = self.sources.get(self.current_id, {}).get("parent_id")
        if node_id not in self.sources:
            raise ValueError("node_not_allowed; choose an ID from allowed_nodes")
        record = self.sources[node_id]
        if not record["available"]:
            raise ValueError(f"source_unavailable: {record.get('reason', 'unknown')}")
        return record

    @staticmethod
    def _size(value):
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))

    def _safe_index(self, record, max_chars):
        index = copy.deepcopy(record["index"])
        omitted = {}
        # Preserve leading functions and safe literal facts while making omission explicit.
        keys = ("top_level", "imports", "runtime_bindings", "constants", "symbols")
        while self._size(index) > max_chars:
            key = next((k for k in keys if index.get(k)), None)
            if key is None:
                return {"status": index["status"], "truncated": True, "reason": "index_budget"}
            index[key].pop()
            omitted[key] = omitted.get(key, 0) + 1
        index["omitted_entries"] = omitted
        return index

    def _record_index_anchors(self, record, index):
        # Facts extracted from AST are returned as evidence but don't grant access to
        # an entire unreturned body; only complete, one-line literal/signature facts.
        for kind in ("symbols", "constants", "imports", "runtime_bindings"):
            for item in index.get(kind, []):
                start, end = item["start_line"], item["end_line"]
                if kind == "symbols":
                    line = record["lines"][start - 1].strip()
                    if item.get("signature") != line or not line.endswith(":"):
                        continue
                    end = start
                elif end != start:
                    continue
                self.anchors.append({"node_id": record["node_id"], "source_sha256": record["source_sha256"],
                                     "start_line": start, "end_line": end, "evidence_kind": "index_fact"})

    def index_summary(self, node_id=None, max_chars=6000, register_anchors=True):
        try:
            record = self._lookup(node_id)
            result = {**self._metadata(record), "line_count": len(record["lines"]),
                      "index": self._safe_index(record, max(100, (max_chars - 1000) // 2)),
                      "note": "AST structure and literal facts only; plan is intent. Read bodies before diagnosing method details."}
            while len(json.dumps(result, ensure_ascii=False, indent=2)) > max_chars:
                index = result["index"]
                key = next((k for k in ("top_level", "imports", "runtime_bindings", "constants", "symbols") if index.get(k)), None)
                if key is None:
                    result = {**self._metadata(record), "index": {"status": "omitted_by_budget"}}
                    break
                index[key].pop()
                index.setdefault("omitted_entries", {})[key] = index.get("omitted_entries", {}).get(key, 0) + 1
            if register_anchors:
                self._record_index_anchors(record, result["index"])
            return result
        except ValueError as exc:
            return {"node_id": node_id or self.current_id, "available": False, "reason": str(exc)}

    def register_index_summary(self, summary):
        record = self.sources.get(summary.get("node_id"))
        if record and record.get("source_sha256") == summary.get("source_sha256"):
            self._record_index_anchors(record, summary.get("index", {}))

    def allowed_nodes(self):
        return [{**self._metadata(record), "available": record["available"],
                 **({"reason": record["reason"]} if not record["available"] else {})}
                for record in self.sources.values()]

    def _read(self, record, args, char_limit):
        symbol, start, end = args.get("symbol"), args.get("start_line"), args.get("end_line")
        if symbol is not None and (start is not None or end is not None):
            raise ValueError("symbol and line range are mutually exclusive")
        if symbol is not None:
            matches = [s for s in record["index"].get("symbols", []) if s["symbol"] == symbol]
            if not matches:
                raise ValueError("symbol_not_found; inspect candidate_code_index")
            start, end = matches[0]["start_line"], matches[0]["end_line"]
        start = 1 if start is None else start
        end = len(record["lines"]) if end is None else end
        max_lines = args.get("max_lines") or self.options.default_read_lines
        column = args.get("start_column") or 0
        if any(not isinstance(v, int) or isinstance(v, bool) for v in (start, end, max_lines, column)):
            raise ValueError("line and column arguments must be integers")
        if not 1 <= start <= end <= len(record["lines"]) or column < 0 or max_lines < 1:
            raise ValueError("invalid_line_range_or_column")
        if column > len(record["lines"][start - 1]):
            raise ValueError("start_column exceeds line length")
        last = min(end, start + min(max_lines, self.options.max_read_lines) - 1)
        content, used, continuation, complete = [], 0, None, []
        for line_no in range(start, last + 1):
            line = record["lines"][line_no - 1]
            offset = column if line_no == start else 0
            prefix = f"{line_no}: " if offset == 0 else f"{line_no} [column {offset}]: "
            room = char_limit - used - len(prefix) - 1
            if room < 1:
                continuation = {"node_id": record["node_id"], "start_line": line_no, "start_column": offset, "end_line": end, "symbol": None}
                break
            segment = line[offset:offset + room]
            content.append(prefix + segment)
            used += len(prefix) + len(segment) + 1
            if offset == 0 and len(segment) == len(line):
                complete.append(line_no)
            if offset + len(segment) < len(line):
                continuation = {"node_id": record["node_id"], "start_line": line_no, "start_column": offset + len(segment), "end_line": end, "symbol": None}
                break
        if continuation is None and last < end:
            continuation = {"node_id": record["node_id"], "start_line": last + 1, "start_column": 0, "end_line": end, "symbol": None}
        anchors = [{"node_id": record["node_id"], "source_sha256": record["source_sha256"],
                    "start_line": n, "end_line": n, "evidence_kind": "source_read"} for n in complete]
        return {**self._metadata(record), "content": "\n".join(content), "start_line": start,
                "end_line": (start + len(content) - 1), "truncated": continuation is not None,
                "continuation": continuation}, anchors

    def _diff(self, args, char_limit):
        target = self._lookup(args.get("target_node_id"))
        base = self._lookup(args.get("base_node_id") or target.get("parent_id") or "__no_parent__")
        symbol = args.get("symbol")
        def lines(record):
            if symbol is None:
                return record["lines"], 1
            matches = [s for s in record["index"].get("symbols", []) if s["symbol"] == symbol]
            if not matches:
                return [], 1
            start, end = matches[0]["start_line"], matches[0]["end_line"]
            return record["lines"][start - 1:end], start
        left, left_start = lines(base)
        right, right_start = lines(target)
        if symbol is not None and not left and not right:
            raise ValueError("symbol_not_found_in_either_source")
        diff = list(difflib.unified_diff(left, right, fromfile=base["node_id"], tofile=target["node_id"], lineterm=""))
        # Unified hunk offsets refer to the sliced symbol; explicit bases avoid
        # presenting those offsets as full-source line numbers.
        offset = 0 if args.get("offset") is None else args["offset"]
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0 or offset > len(diff):
            raise ValueError("invalid_diff_offset")
        column = args.get("offset_column") or 0
        if not isinstance(column, int) or isinstance(column, bool) or column < 0:
            raise ValueError("invalid_diff_column")
        if column and (offset == len(diff) or column > len(diff[offset])):
            raise ValueError("diff_column exceeds line length")
        output, size, next_offset, next_column = [], 0, offset, column
        partial_lines = []
        for position in range(offset, len(diff)):
            line = diff[position]
            begin = column if position == offset else 0
            room = char_limit - size - 1
            if room < 1:
                next_offset, next_column = position, begin
                break
            fragment = line[begin:begin + room]
            output.append(fragment)
            size += len(fragment) + 1
            if begin or begin + len(fragment) < len(line):
                partial_lines.append({"diff_line_offset": position, "start_column": begin,
                                      "end_column": begin + len(fragment), "line_chars": len(line)})
            if begin + len(fragment) < len(line):
                next_offset, next_column = position, begin + len(fragment)
                break
            next_offset, next_column = position + 1, 0
        changed = []
        base_symbols = {s["symbol"]: s for s in base["index"].get("symbols", [])}
        target_symbols = {s["symbol"]: s for s in target["index"].get("symbols", [])}
        for name in sorted(set(base_symbols) | set(target_symbols)):
            a, b = base_symbols.get(name), target_symbols.get(name)
            atext = base["lines"][a["start_line"] - 1:a["end_line"]] if a else None
            btext = target["lines"][b["start_line"] - 1:b["end_line"]] if b else None
            if atext != btext:
                changed.append({"symbol": name, "base_range": [a["start_line"], a["end_line"]] if a else None,
                                "target_range": [b["start_line"], b["end_line"]] if b else None})
        return {"base": self._metadata(base), "target": self._metadata(target), "content": "\n".join(output),
                "partial_diff_lines": partial_lines,
                "changed_symbols": changed[:30], "changed_symbols_omitted": max(0, len(changed) - 30),
                "hunk_line_origin": {"base": left_start, "target": right_start},
                "truncated": next_offset < len(diff),
                "continuation": {"offset": next_offset, "offset_column": next_column, "base_node_id": base["node_id"],
                                 "target_node_id": target["node_id"], "symbol": symbol} if next_offset < len(diff) else None}, []

    def dispatch(self, name, arguments, max_chars=None):
        args = copy.deepcopy(arguments) if isinstance(arguments, dict) else {}
        if self.calls_used >= self.options.max_calls or self._budget_stopped:
            result = {"status": "budget_exhausted", "remaining_calls": 0 if self.calls_used >= self.options.max_calls else self.options.max_calls - self.calls_used,
                      "remaining_chars": max(0, self.options.max_total_chars - self.chars_used)}
            self.ledger.append({"tool": name, "arguments": args, "response": copy.deepcopy(result)})
            return result
        self.calls_used += 1
        remaining = self.options.max_total_chars - self.chars_used
        # Reserve serialized metadata and JSON escaping overhead before reading.
        cap = min(remaining, self.options.max_diff_chars if name == "diff_candidate_code" else self.options.max_read_chars)
        if max_chars is not None:
            cap = min(cap, max(1, int(max_chars)))
        anchors = []
        try:
            if cap < 1500:
                self._budget_stopped = True
                raise ValueError("code_character_budget_exhausted")
            allowed_params = {x["function"]["name"]: set(x["function"]["parameters"]["properties"]) for x in self.tools()}
            if name not in allowed_params:
                raise ValueError("unknown_code_tool")
            if set(args) - allowed_params[name]:
                raise ValueError("unexpected_arguments")
            if name == "candidate_code_index":
                record = self._lookup(args.get("node_id"))
                result = {**self._metadata(record), "index": self._safe_index(record, cap - 1200),
                          "allowed_nodes": self.allowed_nodes(), "truncated": False}
            elif name == "read_candidate_code":
                record = self._lookup(args.get("node_id"))
                result, anchors = self._read(record, args, max(1, (cap - 1800) // 2))
            else:
                result, anchors = self._diff(args, max(1, (cap - 7000) // 2))
            result.setdefault("status", "ok")
        except (ValueError, TypeError, KeyError) as exc:
            result = {"status": "error", "reason": str(exc)}
        result["budget"] = {"calls_used": self.calls_used, "remaining_calls": self.options.max_calls - self.calls_used,
                            "remaining_chars": 0}
        # Normal source output is prebounded. If extensive metadata nevertheless
        # exceeds the serialized cap, report a bounded error instead of altering
        # any supposedly exact source lines or emitting invalid JSON.
        if self._size(result) > cap:
            result = {"status": "error", "reason": "response_metadata_exceeds_budget; request a narrower symbol or range",
                      "budget": result["budget"]}
            anchors = []
        for _ in range(4):
            result["budget"]["remaining_chars"] = max(0, remaining - self._size(result))
        if self._size(result) > cap:
            result = {"status": "budget_exhausted"} if cap >= 29 else {}
            anchors = []
        self.chars_used += self._size(result)
        self.anchors.extend(anchors)
        if name == "candidate_code_index" and result.get("status") == "ok":
            self._record_index_anchors(record, result["index"])
        self.ledger.append({"tool": name, "arguments": args, "response": copy.deepcopy(result)})
        return result

    def validate_anchor(self, anchor):
        if not isinstance(anchor, dict):
            return False
        try:
            start, end = anchor["start_line"], anchor["end_line"]
            if not isinstance(start, int) or not isinstance(end, int) or start < 1 or start > end:
                return False
            record = self.sources.get(anchor["node_id"])
            if not record or end > len(record.get("lines", [])):
                return False
            matches = [a for a in self.anchors if a["node_id"] == anchor["node_id"]
                       and a["source_sha256"] == anchor["source_sha256"]]
            return all(any(a["start_line"] <= line <= a["end_line"] for a in matches) for line in range(start, end + 1))
        except (KeyError, TypeError):
            return False
