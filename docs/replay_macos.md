# Mac MPS 单步 replay

本文说明如何在 Apple Silicon Mac 上，从已保存的父节点运行单步 replay。
实验统一使用 `replay_improve.py`，通过参数选择原 run、父节点、seed 和 arm。
只有显式传入 `--profile mac-mps` 才启用 Mac 路径；集群继续使用原入口和依赖。
四组材料和共同规则见 [replay 说明](replay_improve.md)。

当前 Mac 实现仅支持启用 candidate runtime 的 Jigsaw Unintended Bias 任务，以及
[adapter 注册表](../utils/replay_macos/adapters/__init__.py)中已有的父源码。
每个父节点需要独立校准；换成未登记的源码需要先实现适配器，不能仅更改命令参数。

## 代码入口与本机资源

Mac 代码集中在一个包中：

```text
utils/
  prepare_macos_parent.py       # 准备／校准 CLI
  replay_macos/
    __init__.py
    runtime.py                 # profile、环境、来源与 baseline 校验
    preparation.py             # 准备源码及显式 --execute 校准
    adapters/
      __init__.py              # 按原始源码 SHA256 选择已登记的 adapter
      common.py                # 共享源码校验与严格文本替换
      parent_<id>.py           # 各父模型的具体实现
```

各 adapter 使用相同接口，按原始源码哈希选择：

```python
from utils.replay_macos.adapters import get_adapter

adapter = get_adapter(original_source_sha256)
version = adapter.ADAPTER_VERSION
adapted_code = adapter.adapt(original_code)
```

适配器负责平台移植，保留原模型、损失、采样和优化器语义。未知源码直接拒绝。
已校准 baseline 可在来源、适配版本、预算和环境一致时复用。
数据、corpus、原 run、baseline 和输出的含义见 [资源说明](replay_resources.md)；
具体本机路径保存在 [资源清单 JSON](replay_resources.json)，清单不会被入口自动读取。

## 本机 Python 3.14 环境

从仓库根目录操作，直接使用现有的原生 Apple Silicon Python 3.14 `.venv`。
Mac 的依赖版本单独记录；集群继续使用原来的依赖文件和环境，不要把 Mac 依赖安装到集群：

```bash
.venv/bin/python -c 'import sys, platform; print(sys.executable, sys.version, platform.machine())'
.venv/bin/python -m pip install -r requirements_macos_replay.txt
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > .venv/requirements.macos-replay.lock.txt
```

不要用裸 `pip`，避免命令落到其他解释器。Mac 依赖文件为 Python 3.14 固定直接依赖，
其中 NumPy、pandas、scikit-learn、PyTorch 等使用有兼容 wheel 的版本；安装时让 pip 解析传递依赖。不要使用集群的
`--no-deps` 安装步骤，也不要再叠加 `requirements_ml.txt` 或 `requirements_domain.txt`。
冻结安装结果后，整组 A/B/C/D 使用同一个环境。这个文件只覆盖当前父模型及 replay
运行所需的包；LLM 不能假定集群的大量领域包在 Mac 上也已安装。

Mac 依赖文件固定 PyTorch 2.14.1，入口要求至少 2.14.1。升级环境后需重新校准，
使用新的 baseline 目录；失败或未完成的校准不能作为四组共同基线。

Memory 使用 CPU，并保留原 memory 内容；训练、验证和测试推理使用 MPS float32。
CPU 完成文本和 SciPy 稀疏预处理，MPS 接收非零元素的行号、列号、数值等普通张量，
不构造 SparseMPS 张量，也不展开完整 dense 特征矩阵。Mac 使用
`DataLoader(num_workers=0)`。校准或 replay 时 MPS 不可用会直接拒绝；
`PYTORCH_ENABLE_MPS_FALLBACK` 必须未设置或为 `0`，不能静默转到 CPU 后仍标为 MPS 实验。

## 准备输入

先把原 run 的恢复材料复制到本地，保持原 run 为只读来源：

- `logs/`，包含原 journal 和 config；
- 原配置启用 global memory 时所需的 `workspace/global_memory/records.json`；
- `workspace/candidate_results/contract/`；
- 截至 parent 的候选源码、诊断和选中快照所需材料，见 [恢复边界](replay_improve.md#恢复边界与运行条件)；
- 原 parent 的 `workspace/candidate_results/candidates/<parent_id>/review.json`，状态须为 `passed`；
- 与原 contract SHA256 完全一致的公开 `train.csv`、`test.csv`，以及任务描述；
- 与实验计划相同版本的论文 corpus，四组使用同一路径和哈希。

只有精简日志下载包不足以恢复实验；不能用抽样 CSV 替代原公开数据。
历史 checkpoint 权重不用于续训。Mac 校准和每个 improve child 都从其代码重新训练；
历史预测和 manifest 只用作来源和诊断证据。

## 设置本次实验参数

以下命令从仓库根目录执行，准备、校准和 replay 共用同一组参数。
先替换所有占位符；`REPLAY_SOURCE_RUN` 是原 run 的目录名，`REPLAY_PARENT_ID` 是完整节点 ID，
`REPLAY_SEED` 填整数。每个 seed 使用独立的 experiment ID，同一 seed 的四个 arm 共用该 ID。

```bash
REPLAY_RUNS_ROOT='/absolute/path/to/source-runs'
REPLAY_SOURCE_RUN='<source-run-name>'
REPLAY_PARENT_ID='<full-parent-node-id>'
REPLAY_DATA_DIR="$REPLAY_RUNS_ROOT/$REPLAY_SOURCE_RUN/workspace/input"
REPLAY_CORPUS='/absolute/path/to/paper_corpus'
REPLAY_BASELINE="$PWD/mac_baselines/<baseline-name>"
REPLAY_OUTPUT_ROOT="$PWD/runs_mac_mps"
REPLAY_EXPERIMENT_ID='<experiment-id>'
REPLAY_SEED='<integer-seed>'
```

在当前 shell 设置 `LLM_API_KEY` 和可达的 `LLM_BASE_URL`；`LLM_MODEL`、
`LLM_REASONING_EFFORT` 可覆盖原配置。四个 arm 使用相同环境和模型设置。
密钥只放环境变量。`.env.mac` 不会被入口自动加载，也不会自动读取某个代理的配置文件。

## 先生成共同父源码，再手动执行校准

先运行准备命令。默认只准备移植产物，不启动训练：

```bash
.venv/bin/python utils/prepare_macos_parent.py \
  --runs-root "$REPLAY_RUNS_ROOT" \
  --source-run "$REPLAY_SOURCE_RUN" \
  --parent-node-id "$REPLAY_PARENT_ID" \
  --data-dir "$REPLAY_DATA_DIR" \
  --output-dir "$REPLAY_BASELINE"
```

输出目录先包含 `original_solution.py`、`solution.py` 和 `baseline.json`。
检查生成的源码和适配元数据后，由你在同一命令中加入 `--execute`，才真正执行一次
共同父模型的校准训练。校准不生成新的改进方案；失败或未执行的准备产物不能用于
四组 replay。校准会使用原公开数据和验证 contract，产生新的节点 ID、实际 Mac
指标、诊断和正常源码/快照哈希链。它可能需要较长时间，请把它作为一次真实训练安排。
准备后的源码不可手动修改；需要修改适配器时，更新版本并使用新的准备目录。
已成功执行或失败的目录不能再次校准，新一次校准必须使用新的 `--output-dir`。

已有完成且匹配当前参数及环境的 baseline 时，跳过准备和校准，直接运行 replay。
Replay 本身不会自动启动父模型校准。

原历史日志、时间戳和源环境指标不会被改写。四组恢复同一个原历史前缀，再追加同一个
已成功校准的 Mac 父节点。原 parent 的剩余预算在准备时冻结，校准耗时不扣除四组预算；
不会用旧 root 时间与现在的 Mac 节点时间重新计算剩余预算。
子模型执行上限仍由原配置的 `exec.timeout`、candidate budget 和冻结的剩余预算共同限定。

## 手动启动每个 arm

下面示例启用 [材料协议 v2](replay_improve.md#可选材料协议-v2mac-diff-replay)，要求原配置
`agent.use_diff_mode=True`。四个 arm 必须使用相同协议；省略 `--material-protocol v2`
则使用默认材料流程。此开关不改变父模型适配，匹配的共同 baseline 可继续复用。

```bash
.venv/bin/python replay_improve.py \
  --profile mac-mps --material-protocol v2 \
  --runs-root "$REPLAY_RUNS_ROOT" \
  --source-run "$REPLAY_SOURCE_RUN" \
  --parent-node-id "$REPLAY_PARENT_ID" \
  --prepared-baseline "$REPLAY_BASELINE" \
  --output-root "$REPLAY_OUTPUT_ROOT" \
  --experiment-id "$REPLAY_EXPERIMENT_ID" \
  --arm A \
  --seed "$REPLAY_SEED" \
  --data-dir "$REPLAY_DATA_DIR" \
  --corpus "$REPLAY_CORPUS"
```

A 完成后，使用相同参数依次运行 B、C、D，仅改 `--arm`。也可直接在终端串行运行整组：

```bash
for REPLAY_ARM in A B C D; do
  .venv/bin/python replay_improve.py \
    --profile mac-mps --material-protocol v2 \
    --runs-root "$REPLAY_RUNS_ROOT" \
    --source-run "$REPLAY_SOURCE_RUN" \
    --parent-node-id "$REPLAY_PARENT_ID" \
    --prepared-baseline "$REPLAY_BASELINE" \
    --output-root "$REPLAY_OUTPUT_ROOT" \
    --experiment-id "$REPLAY_EXPERIMENT_ID" \
    --arm "$REPLAY_ARM" \
    --seed "$REPLAY_SEED" \
    --data-dir "$REPLAY_DATA_DIR" \
    --corpus "$REPLAY_CORPUS" || break
done
```

单 arm 命令与整组循环按需选择，不要依次执行而重复运行 A。每个 arm 都会调用 LLM 并训练
新 child；循环遇到命令失败后停止。换 seed 时修改 `REPLAY_SEED` 和 `REPLAY_EXPERIMENT_ID`，
其余共同输入保持一致。不要同时运行相同 experiment ID 的相同 arm。

输出位于 `$REPLAY_OUTPUT_ROOT/replay_<experiment-id>/{A,B,C,D}/`。配对信息包含
Mac profile、环境、原 parent、有效 Mac parent、适配及基线哈希、输入和预算。
不一致时，在清空已有 arm 结果之前拒绝执行。相同参数重跑同一个 arm 仍遵循原 replay
的覆盖语义：清空该 arm，再运行；其他 arm 和原 run 保持不变。

## 评分与解释

公共验证指标由原 candidate runtime 和固定 contract 计算。若还要私有评分，需要
另外安装 MLE-bench、应用 [Jigsaw 评分补丁](../patches/mlebench/README.md)，并在准备
命令中通过 `--dataset-dir /path/to/mlebench-data` 记录本机数据根目录，供后续 replay 使用。
也可以在 replay 命令中传入同名参数覆盖它，四组必须一致。校准本身只计算公共验证指标。
没有私有答案或评分依赖时，私有评分单独记录为不可用或失败，不能把缺失分数当作零。

Mac child 的提升相对共同 Mac baseline 解释。原 GPU 指标和耗时保留为历史证据，
不能直接当作 Mac 的性能测量，也不能把 GPU→MPS 移植差异解释成材料带来的改进。
Mac 单步的停滞计数从零开始，避免仅因设备迁移后的分数变化触发强制突破提示。
调整时间预算、环境或适配版本后，使用新的实验 ID，并重新准备相应共同基线。

离线检查验证入口隔离、恢复和提示行为；小型合成数据已验证 MPS 前向、反向、优化器更新
与 checkpoint 恢复。完整校准的耗时、峰值内存及 A/B/C/D 成功率仍需手动实验验证。
开发期间不会自动启动上述校准或实验。
