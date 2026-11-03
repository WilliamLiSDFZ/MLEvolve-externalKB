# Mac MPS 单步 replay

此模式适配两个已锁定的 Jigsaw parent：

- 原 run `20260914_071735_jubias-base-gpt56sol-s60`，parent
  `36845c7c3f1c4beca8eebc7a37852e7c`，适配版本 `jigsaw-parent5-mps-v1`。
- 原 run `20260912_081234_jubias-base-gpt56sol-s57`，parent
  `f327226c666446d58718f3801048032c`，适配版本 `jigsaw-f327226c-mps-v1`。

适配器检查源码哈希，每个 parent 必须独立校准；不会尝试迁移任意候选。
原集群命令、依赖文件、K8s 入口和 CUDA 可见设备规则保持原来的用法；只有显式传入
`--profile mac-mps` 才启用 Mac replay。四组的材料定义见 [replay 说明](replay_improve.md)。

## 代码入口与本机资源

Mac 代码集中在一个包中，原准备命令保持兼容：

```text
utils/
  prepare_macos_parent.py       # 原 CLI，只调用 preparation.main
  replay_macos/
    __init__.py
    runtime.py                 # profile、环境、来源与 baseline 校验
    preparation.py             # 准备源码及显式 --execute 校准
    adapters/
      __init__.py              # 按原始源码 SHA256 选择已登记的 adapter
      common.py                # 共享源码校验与严格文本替换
      parent_36845c7c.py
      parent_f327226c.py
```

两个 adapter 使用相同接口，不按 seed 选择：

```python
from utils.replay_macos.adapters import get_adapter

adapter = get_adapter(original_source_sha256)
version = adapter.ADAPTER_VERSION
adapted_code = adapter.adapt(original_code)
```

未知源码直接拒绝；每个实现也独立校验源码。两种模型的数值移植分别保留，
原适配版本和生成源码哈希不变，已完成的 baseline 可继续校验和使用。
旧内部模块 `utils.macos_parent_adapter` / `utils.macos_parent_f327226c` 已移入此包；
调用端统一使用上述接口。现有手动启动脚本使用的
`from utils.replay_macos import require_mps` 入口也保持兼容。

本机已下载的数据、corpus、原 run、baseline 和实验目录见
[资源说明](replay_resources.md) 与 [资源清单 JSON](replay_resources.json)。
清单只用于查找与核对，不是启动配置。`s68` 是历史下载目录名，其中的数据和 corpus
也用于 S67/S69；这次未移动任何实验资源，也未改写历史 manifest 或启动脚本。

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

Mac 固定使用 PyTorch 2.14.1。2.9.1 在此父模型不断变化的稀疏批次上出现 MPS
驱动内存持续增长，本机完整校准曾因此 OOM；小批次预测/梯度检查不足以发现它。
2.14 包含原生 Metal `index_select` / `index_add` 和分配器修复，
见 [官方发布说明](https://pytorch.org/blog/pytorch-2-14-release-blog/)。
Mac 入口会拒绝低于 2.14.1 的版本。不要通过关闭 MPS 内存上限绕过问题。
升级后使用新的校准目录，保留失败目录的日志；已失败的快照不能充当完成的共同基线。

Memory 使用 CPU，并保留原 memory 内容；训练、验证和测试推理使用 MPS float32。
CPU 完成文本和 SciPy 稀疏预处理，MPS 接收非零元素的行号、列号、数值等普通张量，
不构造 SparseMPS 张量，也不展开完整 dense 特征矩阵。Mac 首版使用
`DataLoader(num_workers=0)`。校准或 replay 时 MPS 不可用会直接拒绝；
`PYTORCH_ENABLE_MPS_FALLBACK` 必须未设置或为 `0`，不能静默转到 CPU 后仍标为 MPS 实验。

S57 的 `f327226c` 父模型使用稀疏 EmbeddingBag 和 SparseAdam。其独立适配器以普通
MPS 张量保存每次出现的特征梯度，按原顺序先裁剪、再合并重复特征，只更新本批涉及的
参数行及优化器状态；没有换成 AdamW。模型、损失、批次采样、训练轮数与 checkpoint
内容保持原定义。每个父节点的适配版本独立校验，原 S60 移植源码保持逐字一致。

## 准备输入

先把原 run 的恢复材料复制到本地，保持原 run 为只读来源：

- `logs/`，包含原 journal 和 config；
- `workspace/global_memory/records.json`；
- `workspace/candidate_results/contract/`；
- 截至 parent 的候选源码、诊断和选中快照所需材料，见 [恢复边界](replay_improve.md#恢复边界与运行条件)；
- 原 parent 的 `workspace/candidate_results/candidates/<parent_id>/review.json`，状态须为 `passed`；
- 与原 contract SHA256 完全一致的公开 `train.csv`、`test.csv`，以及任务描述；
- 与实验计划相同版本的论文 corpus，四组使用同一路径和哈希。

只有精简日志下载包不足以恢复实验；不能用抽样 CSV 替代原公开数据。
历史 checkpoint 权重不用于续训。Mac 校准和每个 improve child 都从其代码重新训练；
历史预测和 manifest 只用作来源和诊断证据。

以下示例中的 `/path/to/...` 均需替换成本机实际路径。

## 先生成共同父源码，再手动执行校准

先运行准备命令。默认只准备移植产物，不启动训练：

```bash
.venv/bin/python utils/prepare_macos_parent.py \
  --runs-root "$HOME/nautilus/results" \
  --source-run 20260914_071735_jubias-base-gpt56sol-s60 \
  --parent-node-id 36845c7c3f1c4beca8eebc7a37852e7c \
  --data-dir /path/to/jigsaw/prepared/public \
  --output-dir mac_baselines/jubias-parent5
```

输出目录先包含 `original_solution.py`、`solution.py` 和 `baseline.json`。
检查生成的源码和适配元数据后，由你在同一命令中加入 `--execute`，才真正执行一次
共同父模型的校准训练。校准不生成新的改进方案；失败或未执行的准备产物不能用于
四组 replay。校准会使用原公开数据和验证 contract，产生新的节点 ID、实际 Mac
指标、诊断和正常源码/快照哈希链。它可能需要较长时间，请把它作为一次真实训练安排。
准备后的源码不可手动修改；需要修改适配器时，更新版本并使用新的准备目录。
已成功执行或失败的目录不能再次校准，新一次校准必须使用新的 `--output-dir`。

原历史日志、时间戳和 A10 指标不会被改写。四组恢复同一个原历史前缀，再追加同一个
已成功校准的 Mac 父节点。原 parent 的剩余预算在准备时冻结，校准耗时不扣除四组预算；
不会用旧 root 时间与现在的 Mac 节点时间重新计算剩余预算。原 improve 候选的
7,200 秒上限继续适用。

## 手动启动每个 arm

先在当前 shell 配置 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` 和
`LLM_REASONING_EFFORT`，四组保持一致。密钥只放环境变量，不写在命令参数里。
集群内的 `http://cliproxy:8317/v1` 必须替换为本机可达的服务地址。
`.env.mac` 只是可选的本地存放位置，入口不会自动加载它。

```bash
.venv/bin/python replay_improve.py \
  --profile mac-mps \
  --runs-root "$HOME/nautilus/results" \
  --source-run 20260914_071735_jubias-base-gpt56sol-s60 \
  --parent-node-id 36845c7c3f1c4beca8eebc7a37852e7c \
  --prepared-baseline mac_baselines/jubias-parent5 \
  --output-root runs_mac_mps \
  --experiment-id jubias-parent5-mac-s67 \
  --arm A \
  --seed 67 \
  --data-dir /path/to/jigsaw/prepared/public \
  --corpus /path/to/cluster-paper-corpus
```

A 完成后，使用相同参数依次运行 B、C、D，仅改 `--arm`。每个 arm 都会真实调用
improve/review LLM，并训练新 child；此命令不是只生成代码。建议先串行完成一组，
测清内存和耗时后再安排其他 seed。不要同时运行相同 experiment ID 的相同 arm。

输出位于 `runs_mac_mps/replay_jubias-parent5-mac-s67/{A,B,C,D}/`。配对信息包含
Mac profile、环境、原 parent、有效 Mac parent、适配及基线哈希、输入和预算。
不一致时，在清空已有 arm 结果之前拒绝执行。相同参数重跑同一个 arm 仍遵循原 replay
的覆盖语义：清空该 arm，再运行；其他 arm 和原 run 保持不变。

## 评分与解释

公共验证指标由原 candidate runtime 和固定 contract 计算。若还要私有评分，需要
另外安装 MLE-bench、应用 [Jigsaw 评分补丁](../patches/mlebench/README.md)，并在准备
命令中通过 `--dataset-dir /path/to/mlebench-data` 记录本机数据根目录，供后续 replay 使用。
也可以在 replay 命令中传入同名参数覆盖它，四组必须一致。校准本身只计算公共验证指标。
没有私有答案或评分依赖时，私有评分单独记录为不可用或失败，不能把缺失分数当作零。

Mac child 的提升相对共同 Mac baseline 解释。旧 A10 指标和耗时保留为历史证据，
不能直接当作 Mac 的性能测量，也不能把 GPU→MPS 移植差异解释成材料带来的改进。
Mac 单步的停滞计数从零开始，避免仅因设备迁移后的分数变化触发强制突破提示。
调整时间预算、环境或适配版本后，使用新的实验 ID，并重新准备相应共同基线。

离线检查验证入口隔离、恢复和提示行为；小型合成数据已验证 MPS 前向、反向、优化器更新
与 checkpoint 恢复。完整校准的耗时、峰值内存及 A/B/C/D 成功率仍需手动实验验证。
开发期间不会自动启动上述校准或实验。
