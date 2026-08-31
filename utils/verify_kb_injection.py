"""Verify where the knowledge base reaches the prompt. No GPU, no API calls.

Checks the three properties the coldstart split is supposed to guarantee:

  1. build_guidance_description returns pretrained-model guidance ONLY, and puts the
     literature techniques on cfg.coldstart.methodology_text instead of concatenating them.
  2. draft_agent renders those techniques under their own heading — NOT inside the
     "Pretrained Model Strategy / Option A [RECOMMENDED]" block, whose "copy the Code
     template EXACTLY" instruction does not apply to prose techniques.
  3. improve_agent injects them only when coldstart.inject_into_improve is set, and when it
     does, they reach BOTH generation paths (full rewrite and the diff/planner path, which
     is the default since agent.use_diff_mode is True).

Run:  python utils/verify_kb_injection.py
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.coldstart.knowledge import TECHNIQUE_SEPARATOR, trim_methodology_text

MARKER = "SENTINEL_TECHNIQUE_MARKER"
PRETRAIN_BLOCK = "Pretrained Model Strategy"
TECH_SEP = TECHNIQUE_SEPARATOR


def _fake_methodology(n: int = 6) -> str:
    blocks = [f"### technique {i} {MARKER}\n*(from: paper {i})*\n\n" + ("body text. " * 40)
              for i in range(n)]
    return ("\n\n---\n## Methodology Insights from Literature\n"
            "The following actionable techniques from recent papers are relevant:\n\n"
            + TECH_SEP.join(blocks))


def _fake_agent(methodology_text: str, inject_into_improve: bool):
    coldstart = SimpleNamespace(use_coldstart=True, description="Model1: resnet50",
                                methodology_text=methodology_text,
                                inject_into_improve=inject_into_improve,
                                improve_token_budget=2000)
    cfg = SimpleNamespace(coldstart=coldstart, pretrain_model_dir="", exp_id="demo")
    acfg = SimpleNamespace(use_diff_mode=True, code=SimpleNamespace(model="gpt-5.6-terra"))
    return SimpleNamespace(cfg=cfg, acfg=acfg, use_coldstart=True,
                           coldstart_description=coldstart.description,
                           methodology_text=methodology_text,
                           task_desc="predict toxicity", data_preview="cols: a,b,c",
                           global_memory=None)


def _stub_api_deps() -> None:
    """Stub the SDKs the llm package imports at module scope.

    `import llm` pulls in google-genai and the OpenAI client; this script is pure text
    assembly and is meant to run anywhere (laptop, CI) without API deps installed. Stubbing
    lets it exercise the REAL compile_prompt_to_md rather than a re-implementation, which is
    the point — the assertions are about what that renderer emits.
    """
    import types

    class _Any:
        """Returns itself for any attribute, so module-level annotations resolve."""

        def __getattr__(self, _name):
            return _Any()

        def __call__(self, *a, **k):
            return _Any()

    for name in ("google", "google.genai", "google.genai.types", "google.genai.errors"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__getattr__ = lambda _n: _Any()   # PEP 562 module-level __getattr__
            sys.modules[name] = mod
    sys.modules["google"].genai = sys.modules["google.genai"]
    sys.modules["google.genai"].types = sys.modules["google.genai.types"]
    sys.modules["google.genai"].errors = sys.modules["google.genai.errors"]
    for exc in ("APIError", "ClientError", "ServerError"):
        setattr(sys.modules["google.genai.errors"], exc, type(exc, (Exception,), {}))


_stub_api_deps()
from llm import compile_prompt_to_md as _COMPILE  # noqa: E402 - must follow the stubs


def _render_instructions(instructions: dict) -> str:
    return _COMPILE(instructions, 2)


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    return ok


def main() -> int:
    ok = True
    meth = _fake_methodology()

    print("\n1. trim_methodology_text cuts on technique boundaries")
    trimmed = trim_methodology_text(meth, 200)          # 200 tokens ~ 800 chars
    ok &= check("shorter than input", len(trimmed) < len(meth),
                f"{len(meth)} -> {len(trimmed)} chars")
    ok &= check("no partial technique", not trimmed.endswith("body text. body tex"),
                "ends on a block boundary")
    ok &= check("header preserved", "Methodology Insights from Literature" in trimmed)
    ok &= check("keeps >=1 technique", MARKER in trimmed)
    ok &= check("passthrough when under budget",
                trim_methodology_text(meth, 100000) == meth)
    ok &= check("empty input -> empty output", trim_methodology_text("", 2000) == "")

    print("\n1b. build_guidance_description splits the two kinds of guidance")
    import engine.coldstart.knowledge as kn

    repo = Path(__file__).resolve().parent.parent
    cfg = SimpleNamespace(
        exp_id="spooky-author-identification",
        coldstart=SimpleNamespace(
            task_json_path=str(repo / "engine/coldstart/competition_tag_classified.json"),
            model_json_path=str(repo / "engine/coldstart/models_guidance_classified.json"),
            description="", methodology_text="SHOULD_BE_OVERWRITTEN"),
        methodology_kb_path="", torch_hub_dir="")
    returned = kn.build_guidance_description(cfg, task_desc="classify text")
    ok &= check("baseline arm: methodology_text reset to empty",
                cfg.coldstart.methodology_text == "")
    ok &= check("baseline arm: return value carries no technique text",
                "Methodology Insights from Literature" not in returned)

    # KB arm: stub the retriever so no index/API is needed.
    cfg2 = SimpleNamespace(
        exp_id="spooky-author-identification", coldstart=SimpleNamespace(
            task_json_path=cfg.coldstart.task_json_path,
            model_json_path=cfg.coldstart.model_json_path,
            description="", methodology_text=""),
        methodology_kb_path="/tmp/fake_kb", methodology_retrieval="lazy", torch_hub_dir="")
    import engine.coldstart.methodology_agent as ma
    _orig = ma.build_methodology_guidance
    ma.build_methodology_guidance = lambda *a, **k: meth
    try:
        returned2 = kn.build_guidance_description(cfg2, task_desc="classify text")
    finally:
        ma.build_methodology_guidance = _orig
    ok &= check("KB arm: techniques land on cfg.coldstart.methodology_text",
                MARKER in cfg2.coldstart.methodology_text)
    ok &= check("KB arm: techniques NOT concatenated onto the return value",
                MARKER not in returned2, "this was the mislabelling bug")

    print("\n1c. the \"None model\" sentinel still works")
    # draft_agent gates the whole Pretrained Model Strategy block on
    #   agent.coldstart_description != "None model"
    # _build_guidance_text returns exactly "None model" when the competition has no entry in
    # competition_tag_classified.json. Under the old code the methodology text was appended to
    # that string, so the comparison failed and the block fired anyway — with Option A reading
    # literally "None model" followed by the paper techniques. Worse, it fired only in the KB
    # arm (the control's methodology text is empty), so the two arms differed by a whole extra
    # block of pretrained-model instructions, not just by the knowledge. A real confound.
    cfg3 = SimpleNamespace(
        exp_id="a-competition-that-is-not-in-the-tag-json",
        coldstart=SimpleNamespace(task_json_path=cfg.coldstart.task_json_path,
                                  model_json_path=cfg.coldstart.model_json_path,
                                  description="", methodology_text=""),
        methodology_kb_path="/tmp/fake_kb", methodology_retrieval="lazy", torch_hub_dir="")
    ma.build_methodology_guidance = lambda *a, **k: meth
    try:
        desc3 = kn.build_guidance_description(cfg3, task_desc="classify text")
    finally:
        ma.build_methodology_guidance = _orig
    ok &= check('description is exactly "None model"', desc3 == "None model", repr(desc3))
    ok &= check("pretrained block would be SKIPPED (sentinel matches)",
                desc3 == "None model")
    ok &= check("but the techniques are still available separately",
                MARKER in cfg3.coldstart.methodology_text)

    print("\n1c-bis. the baseline arm is byte-identical to the pre-fix code")
    # Why this matters for the experiments, not just for correctness:
    #
    # analyze_runs.py refuses to pool runs across the 2026-08-08 injection fix, because the KB
    # arms genuinely changed. That leaves jigsaw's fixed-wiring groups with only arms B and C —
    # there is no fixed-wiring BASELINE on jigsaw at all, and A-vs-B is the contrast the whole
    # project is about. Three legacy baselines (2026-08-07/08/09) already exist.
    #
    # They are reusable if and only if the fix left arm A untouched. Arm A sets no
    # methodology_kb_path, so it should take the same path through both versions — but "should"
    # is what got us here, so this compares the ACTUAL pre-fix source pulled from git against
    # the current one. If it ever fails, the legacy baselines must be dropped and re-run.
    import subprocess
    import tempfile

    FIX_COMMIT = "651fbdc"          # feat(k8s): jigsaw KB arms on fixed injection path (B/C)
    try:
        old_src = subprocess.run(
            ["git", "show", f"{FIX_COMMIT}^:engine/coldstart/knowledge.py"],
            cwd=repo, capture_output=True, text=True, check=True).stdout
    except Exception as e:
        print(f"  [SKIP] cannot read pre-fix source from git ({e})")
        old_src = None

    if old_src:
        with tempfile.TemporaryDirectory() as td:
            old_path = Path(td) / "knowledge_legacy.py"
            old_path.write_text(old_src)
            spec = importlib.util.spec_from_file_location("knowledge_legacy", old_path)
            legacy = importlib.util.module_from_spec(spec)
            sys.modules["knowledge_legacy"] = legacy
            spec.loader.exec_module(legacy)

            def _baseline_cfg():
                # Arm A exactly as the jobs configure it: coldstart on for pretrained-model
                # guidance, no KB path. Fresh object per call — the new code mutates it.
                return SimpleNamespace(
                    exp_id="jigsaw-toxic-comment-classification-challenge",
                    coldstart=SimpleNamespace(
                        task_json_path=cfg.coldstart.task_json_path,
                        model_json_path=cfg.coldstart.model_json_path,
                        description="", methodology_text=""),
                    methodology_kb_path="", torch_hub_dir="")

            desc_old = legacy.build_guidance_description(_baseline_cfg(), task_desc="classify text")
            desc_new = kn.build_guidance_description(_baseline_cfg(), task_desc="classify text")
            ok &= check("arm A: guidance string identical across the fix",
                        desc_old == desc_new,
                        f"{len(desc_old)} vs {len(desc_new)} chars")

            # The other half of arm A's prompt: draft_agent's new technique section is gated on
            # a non-empty methodology_text, so it must not fire for the baseline. Reproduce the
            # gate rather than importing draft_agent (which pulls in the LLM clients).
            baseline_agent = _fake_agent("", inject_into_improve=False)
            gate_fires = bool(baseline_agent.use_coldstart
                              and (getattr(baseline_agent, "methodology_text", "") or "").strip())
            ok &= check("arm A: draft_agent technique section does not fire", not gate_fires)
            ok &= check("=> the three legacy jigsaw baselines are reusable as fixed baselines",
                        desc_old == desc_new and not gate_fires)

    print("\n1d. the real OmegaConf config accepts the new fields")
    from omegaconf import OmegaConf
    real = OmegaConf.load(Path(__file__).resolve().parent.parent / "config" / "config.yaml")

    # The check that matters most, and the one this section did NOT do until it bit us: config.yaml
    # is merged against the @dataclass Config schema, so a TOP-LEVEL key present in the YAML but
    # absent from the dataclass raises ConfigKeyError during load_cfg — before the run writes a
    # single line. `agent_paper_filter` shipped that way and killed a job at startup. Comparing
    # only coldstart.* keys, as this section used to, cannot catch it.
    import dataclasses

    from config import Config

    schema_keys = {f.name for f in dataclasses.fields(Config)}
    yaml_keys = {k for k in real.keys() if not isinstance(real[k], type(real))
                 or not hasattr(real[k], "keys")}
    yaml_top = set(real.keys())
    missing_in_schema = sorted(yaml_top - schema_keys)
    ok &= check("every top-level config.yaml key exists in the Config dataclass",
                not missing_in_schema,
                f"missing: {missing_in_schema}" if missing_in_schema else "")
    # The reverse is only a warning: a schema key with a default need not appear in the YAML.
    only_in_schema = sorted(schema_keys - yaml_top)
    if only_in_schema:
        print(f"    (note: {len(only_in_schema)} schema key(s) not in config.yaml, using "
              f"defaults: {only_in_schema[:6]}{'...' if len(only_in_schema) > 6 else ''})")

    # Then call the real loader rather than reimplementing its merge. Reproducing the merge by
    # hand meant chasing an ever-growing list of runtime-supplied null fields (data_dir, then
    # exp_name, ...) and testing a lookalike instead of the thing. load_cfg() IS what run.py
    # calls, so this fails exactly when startup would.
    import config as cfgmod

    saved_argv = sys.argv
    try:
        sys.argv = ["run.py", "exp_id=verify", "dataset_dir=/tmp",
                    "data_dir=/tmp", "desc_file=/tmp/desc.md",
                    "log_dir=/tmp/verify_cfg", "workspace_dir=/tmp/verify_cfg"]
        loaded = cfgmod.load_cfg()
        ok &= check("load_cfg() succeeds — the exact call run.py makes at startup",
                    loaded is not None)
        ok &= check("new retrieval keys survive the load",
                    loaded.agent_paper_filter is not None
                    and int(loaded.filter_max_keep) > 0
                    and int(loaded.retr_token_budget) > 0,
                    f"filter={loaded.agent_paper_filter} max_keep={loaded.filter_max_keep} "
                    f"budget={loaded.retr_token_budget}")
    except Exception as e:
        ok &= check("load_cfg() succeeds — the exact call run.py makes at startup", False,
                    f"{type(e).__name__}: {e}")
    finally:
        sys.argv = saved_argv
    for k in ("methodology_text", "inject_into_improve", "improve_token_budget"):
        ok &= check(f"coldstart.{k} present in config.yaml", k in real.coldstart)
    real.coldstart.methodology_text = "written at runtime"   # what knowledge.py does
    ok &= check("methodology_text is writable on the loaded config",
                real.coldstart.methodology_text == "written at runtime")
    ok &= check("inject_into_improve defaults to False (running experiments unaffected)",
                real.coldstart.inject_into_improve is False)

    print("\n1e. the injected text is visible in the log, elided")
    from engine.coldstart.knowledge import preview_text, text_digest

    p = preview_text(meth)
    ok &= check("preview is much shorter than the payload",
                len(p) < len(meth) / 2, f"{len(meth)} -> {len(p)} chars")
    ok &= check("preview keeps the first technique", MARKER in p.split("...")[0])
    ok &= check("preview keeps the tail (shows it terminated)", MARKER in p.split("...")[-1])
    ok &= check("preview states how much was elided", "more lines" in p)
    ok &= check("short input is printed whole", preview_text("a\nb\nc") == "a\nb\nc")
    ok &= check("empty input is labelled", preview_text("") == "(empty)")
    ok &= check("digest is stable", text_digest(meth) == text_digest(meth))
    ok &= check("digest distinguishes payloads", text_digest(meth) != text_digest(meth + "x"))
    print("    --- preview as it will appear in the run log ---")
    for ln in p.splitlines():
        print(f"    | {ln[:96]}")

    print("\n2. draft_agent renders techniques under their own heading")
    agent = _fake_agent(meth, inject_into_improve=False)
    # Mirror draft_agent's Instructions assembly without importing it (that would pull in the
    # LLM clients). The ordering asserted below is the property under test.
    instructions: dict = {"Implementation guideline": ["- write python"]}
    pretrained_block = (
        "**Pretrained Model Strategy**:\n"
        f"• **Option A [RECOMMENDED]**: {agent.coldstart_description}\n"
        "**CRITICAL: ... you MUST copy the Code template EXACTLY as provided ...**")
    instructions["Implementation guideline"].extend([pretrained_block])
    if agent.use_coldstart and agent.methodology_text.strip():
        instructions |= {"Techniques from recent literature": ["", "suggestions, not instructions", "",
                                                              agent.methodology_text]}
    rendered = _render_instructions(instructions)
    ok &= check("techniques present", MARKER in rendered)
    ok &= check("techniques have their own heading",
                "Techniques from recent literature" in rendered)
    before_marker = rendered[:rendered.index(MARKER)]
    dist = len(before_marker) - before_marker.rindex(PRETRAIN_BLOCK) if PRETRAIN_BLOCK in before_marker else 10**9
    ok &= check("techniques NOT inside the pretrained-model block",
                "Techniques from recent literature" in before_marker
                and before_marker.rindex("Techniques from recent literature")
                > before_marker.rindex(PRETRAIN_BLOCK),
                "own heading is the nearest preceding heading")

    print("\n3. improve_agent gating")
    from agents.improve_agent import _inject_methodology

    off = _fake_agent(meth, inject_into_improve=False)
    p_off = {"Instructions": {}}
    _inject_methodology(off, p_off, SimpleNamespace(id="n1"))
    ok &= check("switch OFF -> nothing injected",
                MARKER not in _render_instructions(p_off["Instructions"]))

    on = _fake_agent(meth, inject_into_improve=True)
    p_on = {"Instructions": {}}
    _inject_methodology(on, p_on, SimpleNamespace(id="n2"))
    rendered_on = _render_instructions(p_on["Instructions"])
    ok &= check("switch ON -> injected", MARKER in rendered_on)
    ok &= check("respects improve_token_budget (2000 tok ~ 8000 chars)",
                len(rendered_on) < 9000, f"{len(rendered_on)} chars")

    no_kb = _fake_agent("", inject_into_improve=True)
    p_nokb = {"Instructions": {}}
    _inject_methodology(no_kb, p_nokb, SimpleNamespace(id="n3"))
    ok &= check("baseline arm (empty KB) -> nothing injected", not p_nokb["Instructions"])

    print("\n4. injection reaches BOTH generation paths")
    # Full-rewrite path compiles prompt["Instructions"] directly.
    ok &= check("full-rewrite path sees it", MARKER in _render_instructions(p_on["Instructions"]))
    # Diff path: planner_with_memory.generate_initial_plan does prompt_base.copy() then
    # renders ["Instructions"]. Reproduce exactly that.
    planner_view = dict(p_on).copy()
    planner_view["Instructions"]["Output Format"] = ["", "**Output Requirements:**"]
    ok &= check("diff/planner path sees it (default use_diff_mode=True)",
                MARKER in _render_instructions(planner_view["Instructions"]))

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
