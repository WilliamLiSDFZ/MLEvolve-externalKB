# 单步 improve 四组实验

入口为仓库根目录的 `replay_improve.py`。每次调用只执行一个 arm 的
**恢复父节点 → 准备注入 → improve → code review → 执行训练代码 → 解析和验证结果**。
不会启动新的 MLEvolve 搜索，也不会自动选择下一节点或再做 debug。

从仓库根目录执行，先把占位符替换为本次实验的路径和标识：

```bash
python replay_improve.py \
  --runs-root /absolute/path/to/source-runs \
  --source-run '<source-run-name>' \
  --parent-node-id '<full-parent-node-id>' \
  --experiment-id '<experiment-id>' \
  --arm A \
  --corpus /absolute/path/to/paper_corpus
```

另三个 arm 使用相同参数，只把 `--arm` 改成 `B`、`C`、`D`。所有四组都要传相同
`--corpus`，即使 A/B 不读论文。默认 `--runs-root` 是当前仓库的 `runs/`。
集群部署可参考 [Kubernetes 配置示例](../k8s/replay-abcd.md)。Mac 的准备、校准和
终端批量命令见 [Mac MPS replay](replay_macos.md)，无需为每个实验创建启动脚本。

父节点必须是 `logs/journal.json` 中已有、成功且有效的非 root 节点，传完整 `id`，
不接受 step 或 ID 前缀。它可以是 draft、debug 或 improve 等阶段产生的成功节点；
本次操作固定为 improve。不是读取 `best_solution.py` 或最终 solution。

配置继承原 run 的 `logs/config.yaml`。当前环境的 `LLM_API_KEY` 始终替换历史凭据；
如设置 `LLM_MODEL`、`LLM_BASE_URL`、`LLM_REASONING_EFFORT`，它们也会覆盖原配置，
四个 arm 必须相同。`--seed` 可选，默认原 run 的 seed，用于随机材料和进程内随机数；
它不保证远端 LLM 确定性。验证划分仍使用原 run 的 seed 和 contract。

原配置中的 `agent.use_diff_mode` 决定代码生成方式。开启时，planner 先制定计划，
coder 再输出 SEARCH/REPLACE 补丁修改父源码；关闭时生成完整的新源码。
diff 是常规 MLEvolve 已有的生成方式，默认配置开启。材料协议 v2 目前只支持 diff 路径。

## 四组材料

| Arm | 注入正文 | 长度 |
| --- | --- | --- |
| A | 空，不调用 analogy | 0 |
| B | NLTK Gutenberg 小说原文，随机书籍和起点，不清理章节标题 | 恰好 12,000 字符 |
| C | 随机论文，经 LLM 判定与当前任务无关后，从全文提取建议 | 恰好 12,000 字符 |
| D | 现有 analogy agent 的节点诊断、检索和建议报告 | 不超过 12,000 字符 |

材料预算为 `report_char_budget=12000`，**单位是 Python 字符数，
不是 token**；统计注入正文，不包括共用的提示语。

B 从 Austen / Chesterton 小说白名单中选取。NLTK 数据缺失时，下载 Gutenberg
到当前 arm 的目录。原文 SHA256、书名和字符起点均保存。

C 从论文 corpus 随机不放回抽样，最多尝试 20 篇；先取得 PDF 全文，再用任务描述和
全文做一次相关性判断。相关、无法完整读取或超过输入上下文预算的论文会被跳过，
保留原因。选中无关论文后，只用其全文抽取机制、实施建议、假设、限制和验证标准，
不提供父节点代码或日志，不分析当前瓶颈。建议不足时最多生成/续写 3 次；仍不足则失败。
超过预算的 Markdown 截取前 12,000 字符，因此最后一句可能不完整；完整输出另存。
不会用小说或重复建议补齐 C。

D 调用原 `retrieve_for_node`，继承原配置的检索设置并启用全文读取。空报告或超预算
会将本次标记为失败，不静默变成 A。D 的结构化报告和检索轨迹照常存档。

默认模式下，B/C/D 通过相同的 `Supplementary material` 区块送入原 improve 的 diff/full-rewrite
流程。为保持四组生成流程可比，这个实验入口不启用 D 专属的结构化选择/handoff
附加提示；D 复用的是原 analogy 的报告生成过程。普通 `run.py` 的 analogy 行为不变。
Mac 路径可显式启用下述材料协议 v2，四个 arm 必须使用相同协议。

## 可选材料协议 v2（Mac diff replay）

在四个 arm 的命令中都加入 `--profile mac-mps --material-protocol v2`。
该协议要求原配置启用 diff mode；未指定时保持默认流程，集群入口不启用 v2。
该协议只改变建议传递和记录，材料生成、原父节点、memory、模型配置、训练预算和评分规则不变。
已有环境和适配源码不变时，可直接复用已完成的 Mac baseline。

- A 的正文为空，B/C 仍恰好 12,000 字符，D 仍为不超过 12,000 字符的原报告。
  缺失 B/C/D 正文会失败，不作为 A 继续；C 的截断前输出、D 的隐藏结构化字段及
  已丢弃机制都不额外提供给 planner/coder。
- 初始 planner、refine/direct planner、coder（包括重试）和 review 都收到同一份完整正文，
  使用四组相同的区块和输出约束。初始计划保留为自然语言，refine 可根据证据修正方向。
- 最终计划必须包含 `material_decision`：`decision` 为 `adopt` / `reject` / `no_material`，
  同时填写 `source_quotes`、`reason`、`adaptation`、`target_locations`。
  采用需要可核对的原文引用、具体改法和目标位置；拒绝必须解释理由；仅空正文允许 `no_material`。
  决定被冻结后交给 coder，不要求 D 必须采用，也不把声明采用视为已实现。
- 任务、评估指标、验证划分、训练 seed 和执行预算固定。模型、采样、训练损失和优化方法
  可以有理据地改变。coder 无法落实计划时输出 `MATERIAL_CONFLICT: 原因`，该 arm 明确失败。
- review 在原有调用中返回 `material_review`，记录 `consistent` / `inconsistent` / `uncertain`、
  理由及源码原文证据，绑定 review 输入源码哈希。若 reviewer 修补了代码，最终源码的一致性
  记为 `uncertain`，保留输入代码的判断；不为确认采用额外调用 LLM，也不为 D 强制修补。
- 结构化计划、diff、review 各自沿用最多三次的上限；初始自然语言规划一次。
  不合格材料决定可在现有规划上限内重试。LLM 传输层的终止错误（包括其拒绝的不合法 JSON/schema）
  立即结束，不叠加外层重试；已取得的非法输出也保存。各组遵循相同规则，实际调用次数可因失败不同。
  diff 不完整、没有代码变化或 review 无法完成时均失败，不回退成完整重写或直接训练父代码。

每个 arm 的 `logs/material_handoff/` 保存 `audit.json`、完整 `material.txt`、逐次调用的
`calls/*.json`（传给 LLM 包装器的完整 prompt/schema、返回值或错误）、`parent.py`、
`generated.py`、`reviewed.py` 及相对 parent 的 diff。初始自然语言计划保存在对应调用的 response。
`result.json` 摘录协议、决定、最终代码一致性判断和 audit 路径；每组材料哈希只进入本组记录，
不会放入四组共用的 `experiment.json`。共同 identity 会记录 `material_protocol=v2`。
`generated_node.json.prompt_input` 是 improve 构建的汇总提示；审查实际投喂内容时以 `calls/` 为准。

切换协议时，为整组分配新的 experiment ID，四个 arm 全部使用同一协议运行。
协议标识可写入 ID 便于区分；不同协议的结果不组成一组配对实验。
清理实验输出时，下载来源、corpus、公开数据和已完成的共同 baseline 仍是重新运行的输入。

离线验证：

```bash
.venv/bin/python -m unittest \
  tests.verify_replay_material \
  tests.verify_replay_material_planners \
  tests.verify_replay_material_coder_review \
  tests.verify_replay_material_transport
```

## 恢复边界与运行条件

恢复的是 **journal 中截至 parent 的已完成节点前缀**：按原 `step`/追加顺序保留历史，
重建父子关系、分支记录和 local best，并清除搜索结束时的计数器。原 run 当时仍在执行
的并发候选不在这个快照中。原任务描述、数据预览和已有 memory 会复用，不再调用
LLM 重写任务或重新判断指标方向。
若所有历史提示都缺少任务描述/数据预览，才退回原 description 和当前数据预览；
这时恢复的是共同输入，不能声称与原 run 的提示逐字一致。

需要原 `logs/`、父节点创建/完成时间和公开输入数据。优先复制原 `workspace/input/`；
缺失时读取原 `data_dir`，也可用 `--data-dir /path/to/prepared/public` 明确指定。
复制会解引用软链接，四组不会共用可写的输入文件。只拿下载到本机的简化结果包可能
不够：原配置启用 global memory 时必须有 `workspace/global_memory/records.json`；
启用 candidate runtime 时必须有原 `workspace/candidate_results/contract/` 和节点诊断
材料。缺少这些内容会明确报错，不偷偷更换验证集或关闭 memory。

原配置启用 candidate runtime 时，保留完全相同的验证 contract。历史节点的源码、
执行事件、选中快照的验证预测和 manifest 供 D 读取，无需复制模型 checkpoint 权重。
准备注入后，历史候选移到 `source/candidates/` 归档，新结果导出只考虑新 child。
已有预训练模型目录仍按原配置读取；HF、Torch、临时文件等可写缓存放在当前 arm 内。

提示里的剩余时间根据原 parent 的完成时间和原 run 时限重建并冻结；四组使用相同
的单次执行上限，C/D 准备材料的耗时不扣减训练预算。实际训练上限仍同时受原
`exec.timeout`、runtime candidate budget 和恢复的剩余预算约束。Job 的外部时限应另外
预留准备数据、LLM、训练和评分所需时间。

默认集群路径中，每个 arm 只有一个执行槽，继承 Job 的 CUDA 可见设备；GPU/CPU 配额和挂载权限由 Job
负责，脚本不会调度 GPU，也不是执行任意生成代码的安全沙箱。四组应申请相同资源。

原配置需要格式校验服务时，入口会启动自己的本地服务并在结束时关闭。训练后复用
原本的 validation metric；若配置了 MLE-bench `dataset_dir`，还会尝试对新 child 的
submission 做离线私有评分。私有答案不送入 LLM，评分不可用/失败会单独记录。

## 输出与覆盖

```text
runs/replay_<experiment_id>/
  experiment.json              # 原 run、parent、seed、配置、预算等配对约束
  A/                           # B/C/D 结构相同
    result.json                # 本次 child 的指标、有效性、submission、私有评分状态
    source/logs/               # 原日志副本
    source/candidates/         # 截止 parent 的历史源码/诊断材料，准备完成后归档
    injection/                 # 实际注入正文、来源、C全文/判断/建议，或D报告
    logs/
      config.yaml
      restored_journal.json
      generated_node.json      # review 前的代码、计划与汇总提示
      child_solution.py        # review 后本次代码
      journal.json             # 恢复前缀 + 新 child
      executions/              # 原始训练输出
      analogy/                 # D 的原有检索轨迹
    workspace/                 # 独立input、working、submission、runtime结果
    cache/                     # 独立可写缓存
```

实验目录存在时复用；自己的 arm 目录存在时，**整个清空再创建**。不会清空其他 arm
或原 run。相同 experiment ID 的原 run、parent、配置、seed 等不一致时，在删除 arm
之前拒绝执行。不同 arm 可同时创建，公共 manifest 原子发布；不处理同一 arm 同时运行，
由调用者保证不会发生。

配对分析读取每组 `result.json` 的 `child_node_id` 和指标，不使用历史最佳 solution。
`status=completed` 表示这一次候选已评估，是否可用还要看 `is_valid`、`is_buggy` 和
`artifact_status`；`status=failed` 保存准备/执行流程的异常。无有效分数时保持 null，
不会用 0 代替失败。

离线检查：`python tests/verify_replay_improve.py`、`verify_replay_state.py`、
`verify_replay_inputs.py`、`verify_replay_injection.py`（后三个也在 `tests/` 下）。
这些检查 mock LLM；入口测试会实际执行一个只写小 CSV 的 Python 子进程。

## 可选 Mac MPS 模式

Apple Silicon 本机运行需显式传入 `--profile mac-mps`，使用本机 Python 3.14 `.venv`、独立输出目录和
一次性校准的共同 Mac 父节点。省略该参数继续使用本文原来的集群流程。
准备输入、手动校准和 A/B/C/D 命令见 [Mac MPS replay](replay_macos.md)。
