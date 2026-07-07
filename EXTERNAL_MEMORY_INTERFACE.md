# MLEvolve 外置 Memory 接口设计方案

> 目标:在不改变现有检索/写入语义的前提下,把 `GlobalMemoryLayer` 从"被各 agent 直接耦合的具体类"重构为"一个可替换后端的接口实现",从而支持挂载外置 memory 系统(自建向量库 / Mem0 / Zep / MCP memory server 等)。
>
> 所有改动点均标注所引用的源文件与行号(基于当前仓库状态)。

---

## 0. 设计原则

1. **契约最小化**:接口只暴露 agent 真正调用的方法,不把内部实现(FAISS、BM25、sentence-transformers)泄漏到契约中。
2. **行为零回归**:默认后端 `builtin` 必须与现状逐字节等价(仍写 `records.json`、仍用 RRF 检索)。
3. **注入集中化**:后端的构造只在一处发生(`engine/agent_search.py`),通过工厂按配置选择。
4. **顺带修两个既有缺陷**:写入无锁的并发竞态、以及 `node_metadata_map` 被外部直接读取的泄漏耦合。

---

## 1. 现状:被依赖的"隐式契约"

`GlobalMemoryLayer` 定义于 `agents/memory/global_memory.py:21`,在 `engine/agent_search.py:88-97` 被直接构造并赋给 `self.global_memory`。各 agent 通过 `agent.global_memory.*` 使用它。把所有调用点汇总成下表——这就是外置后端**必须**满足的契约面:

| 被调用的成员 | 调用方(文件:行) | 用途 |
|---|---|---|
| `save_node(node, parent_node) -> bool` | `agents/result_parse_agent.py:372`(经 `_save_to_global_memory`,定义于 `:368`,在 `run()` 的 `:442` 触发) | 写入一个节点经验 |
| `retrieve_similar_records(query_text, top_k, alpha, dissimilar, label_filter, stage_filter, min_score)` | `agents/planner/planner_with_memory.py:119` 与 `:127`(label=1/-1);`agents/debug_agent.py:124` | 相似/相异检索 |
| `generate_guidance_prompt(...)` | 当前无直接外部调用,但属公共 API(`global_memory.py:165`),保留以兼容 | 把检索结果拼成提示词 |
| `.records`(用 `len(...) > 0` 判空) | `agents/improve_agent.py:302`;`agents/debug_agent.py:118` | 判断 memory 是否就绪/非空 |
| `.node_metadata_map[record_id]`(**直接字典访问**) | `agents/debug_agent.py:40-41`(在 `_format_debug_memory_guidance`,定义于 `:21`) | 取 `parent_metric/current_metric` 等旁路元数据 |
| `use_global_memory` 开关 | `agents/result_parse_agent.py:151,186,397`(切换 review 函数签名,见 `get_review_func_spec`,`:113`);`agents/improve_agent.py:300` | 决定是否走带记忆的两阶段规划 |

需要特别指出的两处:

- **`.records` 与 `.node_metadata_map` 是公共属性级耦合**(`global_memory.py:45-46`)。外置后端若把数据存在远端,无法廉价地物化整张 `records` 列表来支持 `len()`,也不应暴露内部字典。这两处必须改为方法调用。
- `draft_agent`、`evolution_agent`、`fusion_agent` 用的是**另一套树内 memory**——`SearchNode.fetch_child_memory()`(`engine/search_node.py:196`),例如 `agents/draft_agent.py:66`、`agents/evolution_agent.py:60`、`agents/fusion_agent.py:184`。它与 `GlobalMemoryLayer` 无关,**不在本方案范围内**,无需改动。

数据传输对象(DTO)已经现成且干净:`MemRecord`(`agents/memory/record.py:12`)是 `@dataclass`,自带 `to_dict()`/`from_dict()`(`record.py:34,22`),天然适合作为跨后端的序列化载体。

---

## 2. 目标架构

```
            agent 调用点(result_parse / planner_with_memory / debug / improve)
                              │  只依赖 BaseMemory
                              ▼
            ┌──────────────────────────────────────────┐
            │  BaseMemory (ABC)  agents/memory/base.py   │   ← 新增:稳定契约
            └──────────────────────────────────────────┘
                  ▲                              ▲
   ┌──────────────┘                              └───────────────┐
   │ builtin                                              external│
┌───────────────────────────┐                  ┌────────────────────────────┐
│ GlobalMemoryLayer          │                  │ ExternalMemoryAdapter      │  ← 新增:外置后端
│ (现有类,实现接口+加锁)     │                  │ (HTTP/MCP/Mem0/Zep/向量库) │
│ FAISS+BM25+bge,写 json     │                  │ 远端读写 + 本地过滤兜底     │
└───────────────────────────┘                  └────────────────────────────┘
                  ▲                              ▲
                  └──────────────┬───────────────┘
                                 │
              build_memory(cfg)  agents/memory/factory.py   ← 新增:按配置选后端
                                 │
              engine/agent_search.py:88-97 改为调用工厂
```

`node → (MemRecord, metadata, label)` 的提取逻辑当前内嵌在 `GlobalMemoryLayer.save_node`(`global_memory.py:57-89`),依赖三个私有方法 `_should_save_node`(`:247`)、`_extract_code_summary`(`:254`)、`_determine_label`(`:271`)。为避免每个后端重复实现,把它们抽成无状态纯函数模块 `agents/memory/encoding.py`,builtin 与 external 共用。

---

## 3. 逐文件改动清单

### 3.1 新增 `agents/memory/base.py` —— 稳定契约

```python
"""Stable memory contract. Implemented by GlobalMemoryLayer (builtin) and
ExternalMemoryAdapter (external). Consumed only via agent.global_memory.* —
see §1 for the exhaustive call-site table."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from .record import MemRecord


class BaseMemory(ABC):
    # ---- 写入 (result_parse_agent.py:372) ----
    @abstractmethod
    def save_node(self, node: Any, parent_node: Optional[Any] = None) -> bool:
        """保存一个搜索节点;返回是否真正写入。实现必须保证线程安全。"""

    # ---- 检索 (planner_with_memory.py:119/127, debug_agent.py:124) ----
    @abstractmethod
    def retrieve_similar_records(
        self,
        query_text: str,
        top_k: int = 2,
        alpha: float = 0.5,
        dissimilar: bool = False,
        label_filter: Optional[int] = None,
        stage_filter: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[Tuple[MemRecord, float]]:
        """返回按相关度降序的 (record, score)。score 的标度由后端定义(见 §5)。"""

    # ---- 提示词生成 (公共 API, global_memory.py:165) ----
    @abstractmethod
    def generate_guidance_prompt(
        self,
        query_text: str,
        top_k: int = 2,
        alpha: float = 0.5,
        dissimilar: bool = False,
        stage_filter: Optional[str] = None,
    ) -> str: ...

    # ---- 取代直接属性访问 ----
    @abstractmethod
    def count(self) -> int:
        """记录条数;后端可返回缓存/远端计数。取代 len(self.records)。"""

    @abstractmethod
    def get_metadata(self, record_id: str) -> Optional[Dict[str, Any]]:
        """旁路元数据 (exec_time/parent_metric/current_metric/parent_error)。
        取代 debug_agent.py:40-41 对 node_metadata_map 的直接读取。"""

    # ---- 非抽象便捷方法 ----
    def is_empty(self) -> bool:
        return self.count() == 0

    def flush(self) -> None:
        """可选:外置后端用于落盘/排空异步写队列(见 §4)。默认 no-op。"""
```

### 3.2 新增 `agents/memory/encoding.py` —— 共享的 node→record 提取

把 `global_memory.py:247-290` 的三个私有方法平移为纯函数,使两个后端共用同一套"什么节点该存 / 标签怎么定 / 摘要怎么取"的逻辑:

```python
"""Pure node→record encoding, shared by all backends.
Lifted verbatim from GlobalMemoryLayer._should_save_node / _extract_code_summary /
_determine_label (global_memory.py:247-290) so semantics stay identical."""
from datetime import datetime
from typing import Any, Optional, Tuple, Dict
from .record import MemRecord

def should_save_node(node) -> bool: ...          # ← global_memory.py:247
def extract_code_summary(node) -> str: ...       # ← global_memory.py:254
def determine_label(node, parent_node) -> int: ...  # ← global_memory.py:271

def build_record(node, parent_node) -> Tuple[MemRecord, Dict[str, Any]]:
    """组装 (MemRecord, metadata)。逻辑对应 global_memory.py:57-89。"""
    ...
```

> 重构方式:`GlobalMemoryLayer` 的对应私有方法改为转调 `encoding.*`,保证现有行为不变;`ExternalMemoryAdapter` 直接复用。

### 3.3 改造 `agents/memory/global_memory.py` —— 实现接口 + 加锁

1. 类声明改为 `class GlobalMemoryLayer(BaseMemory):`(`global_memory.py:21`)。
2. 新增两个方法以满足契约:
   ```python
   def count(self) -> int:
       return len(self.records)              # 取代外部 len(self.records)

   def get_metadata(self, record_id):
       return self.node_metadata_map.get(record_id)
   ```
3. **修并发缺陷**:`save_node`(`:51`)当前在 `run.py` Phase 2 的 `ThreadPoolExecutor` 流水线中被并发调用,而它会改 `self.records`、重建 BM25、追加 FAISS(`retriever.py:120`)并整体重写 `records.json`(`global_memory.py:345`),**类内无任何锁**(`save_node_lock` 定义于 `agent_search.py:55`,但仅 `solution_manager.py:65,145` 在用,未覆盖 memory 写入)。改法:
   ```python
   def __init__(self, ...):
       ...
       self._lock = threading.Lock()
   def save_node(self, node, parent_node=None) -> bool:
       with self._lock:
           ...   # 现有 :56-103 全部移入临界区
   ```

### 3.4 新增 `agents/memory/factory.py` —— 配置驱动的后端选择

```python
import logging
from typing import Optional
from .base import BaseMemory

logger = logging.getLogger("MLEvolve")

def build_memory(cfg) -> Optional[BaseMemory]:
    """集中构造 memory 后端。取代 agent_search.py:88-97 的硬编码 new。"""
    acfg = cfg.agent
    if not acfg.use_global_memory:
        return None
    backend = getattr(acfg, "memory_backend", "builtin")
    memory_dir = str(cfg.workspace_dir / "global_memory")   # 同 agent_search.py:91

    if backend == "builtin":
        from .global_memory import GlobalMemoryLayer
        return GlobalMemoryLayer(
            memory_dir=memory_dir,
            embedding_model_path=acfg.memory_embedding_model_path,
            embedding_device=acfg.memory_embedding_device,
            similarity_threshold=acfg.memory_similarity_threshold,
        )
    if backend == "external":
        from .external_adapter import ExternalMemoryAdapter
        return ExternalMemoryAdapter(acfg.memory_external, namespace=cfg.exp_id)
    raise ValueError(f"Unknown agent.memory_backend: {backend!r}")
```

> 附带收益:`EmbeddingModel` 的构造(`global_memory.py:38-42`,当前**写死** `model_type="local"`、`device` 默认 `cuda`)现在被关进 `builtin` 分支。外置后端不再被迫加载本地 bge 模型,也就绕开了"无 GPU 时初始化即崩"的问题。

### 3.5 改造 `engine/agent_search.py` —— 用工厂注入

把 `agent_search.py:87-105` 的整段替换为:

```python
from agents.memory.factory import build_memory
...
# Global memory
self.global_memory = None
if self.acfg.use_global_memory:
    try:
        self.global_memory = build_memory(self.cfg)
        if self.global_memory is not None:
            logger.info(f"[AgentSearch] Memory backend ready: "
                        f"{getattr(self.acfg, 'memory_backend', 'builtin')}")
    except Exception as e:
        import traceback
        logger.warning(f"[AgentSearch] Failed to initialize memory: {e}")
        logger.debug(traceback.format_exc())
        self.global_memory = None
else:
    logger.info("[AgentSearch] Global memory is disabled by config")
```

> 已确认 `GlobalMemoryLayer` 在全仓库**仅**于此处被构造(`agent_search.py:90`),因此改这一处即可覆盖 `run.py` 与 `__init__.py` 的 `Experiment` 包装两条入口。

### 3.6 改造 `agents/debug_agent.py` —— 去掉对内部字典的直接访问

`debug_agent.py:40-41` 当前为:
```python
if record.record_id in agent.global_memory.node_metadata_map:
    metadata = agent.global_memory.node_metadata_map[record.record_id]
```
改为走契约方法:
```python
metadata = agent.global_memory.get_metadata(record.record_id)
if metadata:
    ...
```

### 3.7 改造判空调用点 —— `len(records)` → `count()`

- `agents/improve_agent.py:302`:`len(agent.global_memory.records) > 0` → `not agent.global_memory.is_empty()`(同时保留 `:300-301` 的 `use_global_memory` 与 `is not None` 判断)。
- `agents/debug_agent.py:118`:`if agent.global_memory and len(agent.global_memory.records) > 0:` → `if agent.global_memory and not agent.global_memory.is_empty():`

### 3.8 新增 `agents/memory/external_adapter.py` —— 外置后端骨架

```python
"""External memory backend. Maps MemRecord <-> remote store and performs
retrieval over the network. Must never block or crash the search loop:
all remote failures degrade to save=False / retrieve=[] (mirrors the existing
try/except posture at global_memory.py:105 and :120)."""
from __future__ import annotations
import logging, queue, threading
from typing import Any, Dict, List, Optional, Tuple
from .base import BaseMemory
from .record import MemRecord
from . import encoding

logger = logging.getLogger("MLEvolve")


class ExternalMemoryAdapter(BaseMemory):
    def __init__(self, ext_cfg, namespace: str):
        self.cfg = ext_cfg
        self.namespace = namespace          # 任务隔离键,替代 builtin 的 memory_dir 作用域
        self._client = self._make_client(ext_cfg)   # http / mcp / mem0 / zep / qdrant
        self._lock = threading.Lock()
        # 异步写:避免每个节点一次远端往返拖慢 ThreadPoolExecutor 流水线
        self._wq: "queue.Queue" = queue.Queue()
        self._worker = threading.Thread(target=self._drain, daemon=True)
        self._worker.start()
        self._count_cache = self._client.count(self.namespace)

    # ---- 写入:入队即返回(at-least-once) ----
    def save_node(self, node, parent_node=None) -> bool:
        if not encoding.should_save_node(node):     # 复用 §3.2,语义同 global_memory.py:247
            return False
        record, metadata = encoding.build_record(node, parent_node)
        self._wq.put((record, metadata))
        with self._lock:
            self._count_cache += 1
        return True

    def _drain(self):
        while True:
            record, metadata = self._wq.get()
            try:
                self._client.upsert(self.namespace, record.to_dict(), metadata)
            except Exception as e:
                logger.warning(f"[ExtMemory] upsert failed (dropped): {e}")
            finally:
                self._wq.task_done()

    # ---- 检索:远端 topN + 客户端兜底过滤 ----
    def retrieve_similar_records(self, query_text, top_k=2, alpha=0.5,
                                 dissimilar=False, label_filter=None,
                                 stage_filter=None, min_score=0.0):
        try:
            # 若后端不支持 label/stage 下推,则多取一些再本地过滤
            # (本地过滤逻辑对应 global_memory.py:139-149)
            raw = self._client.search(
                self.namespace, query_text,
                top_k=max(top_k * 5, 20), alpha=alpha, dissimilar=dissimilar,
                filters=self._pushdown(label_filter, stage_filter),
            )
        except Exception as e:
            logger.warning(f"[ExtMemory] search failed: {e}")
            return []
        hits = [(MemRecord.from_dict(d), s) for d, s in raw]
        hits = self._client_side_filter(hits, label_filter, stage_filter, min_score, dissimilar)
        return hits[:top_k]

    def generate_guidance_prompt(self, query_text, top_k=2, alpha=0.5,
                                 dissimilar=False, stage_filter=None) -> str:
        recs = self.retrieve_similar_records(query_text, top_k, alpha,
                                             dissimilar, stage_filter=stage_filter)
        # 复用 builtin 的排版,或抽 global_memory.py:165-240 的纯排版到 encoding.py
        ...

    def count(self) -> int:
        with self._lock:
            return self._count_cache

    def get_metadata(self, record_id) -> Optional[Dict[str, Any]]:
        try:
            return self._client.get_metadata(self.namespace, record_id)
        except Exception:
            return None

    def flush(self) -> None:
        self._wq.join()
```

> 各类后端的接入点都在 `_make_client`:
> - **HTTP 自建服务**:`upsert`/`search`/`count` 映射到你自己的 REST 端点。
> - **MCP memory server**:`_client` 包一层 MCP 调用。
> - **Mem0 / Zep**:`upsert` → `client.add(...)`,`search` → `client.search(...)`,用 `namespace`(=`exp_id`)作 user/session id。
> - **Qdrant/Milvus 等向量库**:adapter 内部自带 embedder(与 builtin 解耦),`search` 走向量检索 + payload 过滤。

### 3.9 配置改动

`config/config.yaml:74-78`(`agent` 段)新增:

```yaml
  # --- Global memory ---
  use_global_memory: True
  memory_backend: builtin          # builtin | external   ← 新增
  memory_similarity_threshold: 0.7
  memory_embedding_device: cuda
  memory_embedding_model_path: "BAAI/bge-base-en-v1.5"
  memory_external:                 # 仅当 memory_backend=external 时使用   ← 新增
    type: http                     # http | mcp | mem0 | zep | qdrant
    base_url: ""
    api_key: ""
    timeout_s: 10
    async_writes: True
```

`config/__init__.py` 的 `AgentConfig`(`:82`,现有字段 `use_global_memory:94 / memory_similarity_threshold:95 / memory_embedding_device:96 / memory_embedding_model_path:97`)新增:

```python
    memory_backend: str                 # "builtin" | "external"
    memory_external: dict               # 或定义独立的 @dataclass MemoryExternalConfig
```

> 注:按 `CLAUDE.md` 说明,这些 dataclass 仅作类型提示,真实取值在 YAML;两处都要加。

---

## 4. 并发与写入策略

- **builtin**:用 §3.3 的类内 `_lock` 把整个 `save_node` 设为临界区,修掉现有竞态(写 `records`/FAISS/`records.json` 全程串行化)。
- **external**:`save_node` 仅入队即返回(§3.8),后台 daemon 线程串行 flush,保证不在 `result_parse_agent.run()`(`:442`)的关键路径上引入远端 RTT。语义为 at-least-once;`flush()` 供 `run.py` 在收尾(`save_run` 附近)排空队列。
- 两个后端都遵循既有"失败即降级"的姿态:写失败返回 `False`、检索失败返回 `[]`,绝不抛到搜索主循环(对齐 `global_memory.py:105,120`)。

---

## 5. 语义边界与风险(务必在文档/代码注释中写明)

1. **`score` 不可移植**:builtin 的 score 来自 RRF / 负 L2(`retriever.py:208,229`),外置后端可能返回 cosine 或自有标度。因此 `min_score` 阈值是**后端相关**的,跨后端不可直接套用。
2. **`dissimilar=True` 的语义**:builtin 取全量后反向排序(`global_memory.py:151-153`)。外置相似度 API 通常只返回"最相似",adapter 需要后端支持"最不相似"或自行取大 topN 反转,否则该参数语义会退化。`planner_with_memory.py` 当前两处检索都是 `dissimilar=False`(`:123,131`),`debug_agent` 也是(`:128`),所以短期风险低,但接口需保留正确语义。
3. **`label`/`stage` 过滤下推**:`label_filter`/`stage_filter`(`global_memory.py:124-149`)若后端不支持,必须在 adapter 端兜底过滤(取更大的远端 topN 再裁剪),否则会漏召回。
4. **记录偏薄**:`MemRecord.method` 只存代码摘要(`encoding.extract_code_summary`,源 `global_memory.py:254-269`),不是完整代码。若外置系统要支持"复用整段解法",需扩展 DTO(加 `code` 字段),这是接口之外的独立增强,不影响本方案。

---

## 6. 兼容性与迁移

- 默认 `memory_backend: builtin` ⇒ 行为与现状**完全一致**(同样的 `GlobalMemoryLayer`、同样的 `records.json`、同样的检索路径)。
- agent 侧改动仅 3 处且语义等价:`debug_agent.py:40-41`、`improve_agent.py:302`、`debug_agent.py:118`。其余调用方(`planner_with_memory.py`、`result_parse_agent.py`)本就只调用契约内方法,**无需改动**。
- `EmbeddingModel`(`embedding_models.py:20`)保持不变;它已支持 `local/openai/azure/custom`,仅是此前未被 builtin 暴露——外置后端可按需复用它,也可完全不用。

---

## 7. 验证计划

1. **接口一致性测试**:实现一个内存版 `DictMemory(BaseMemory)`(纯 dict,无网络),注入后跑 `improve`/`debug` 路径,断言这些 agent 只触达 `BaseMemory` 声明的方法(可用 mock 断言无 `node_metadata_map`/`records` 直接访问)。
2. **回归对拍**:同一任务分别用 `builtin`(改造前 vs 改造后)各跑若干步,对比 `records.json` 与检索召回完全一致,确认零回归。
3. **并发压测**:多线程并发调用 `save_node`,断言 builtin 不再出现 `records.json` 损坏 / FAISS 与 BM25 计数不一致。
4. **外置冒烟**:起一个本地 mock HTTP server(实现 `upsert/search/count/get_metadata`),设 `memory_backend=external` 跑通写入→检索→提示词注入全链路,并验证远端故障时搜索不中断、`flush()` 能排空队列。

---

## 附:改动文件总览

| 操作 | 文件 | 关键位置 |
|---|---|---|
| 新增 | `agents/memory/base.py` | `BaseMemory` 契约 |
| 新增 | `agents/memory/encoding.py` | 提取自 `global_memory.py:247-290` 的纯函数 |
| 新增 | `agents/memory/factory.py` | `build_memory(cfg)` |
| 新增 | `agents/memory/external_adapter.py` | `ExternalMemoryAdapter` 骨架 |
| 改造 | `agents/memory/global_memory.py:21,51` | 实现 `BaseMemory`、`save_node` 加锁、加 `count/get_metadata` |
| 改造 | `engine/agent_search.py:87-105` | 改调 `build_memory` |
| 改造 | `agents/debug_agent.py:40-41,118` | 用 `get_metadata` / `is_empty` |
| 改造 | `agents/improve_agent.py:302` | 用 `is_empty` |
| 改造 | `config/config.yaml:74-78` | 加 `memory_backend` / `memory_external` |
| 改造 | `config/__init__.py:82-97` | `AgentConfig` 加对应字段 |
