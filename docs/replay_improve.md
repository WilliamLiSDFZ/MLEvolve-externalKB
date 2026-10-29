# 单步 improve 四组实验

入口为仓库根目录的 `replay_improve.py`。每个 Job 调用一次，只执行一个 arm 的
**恢复父节点 → 准备注入 → improve → code review → 执行训练代码 → 解析和验证结果**。
不会启动新的 MLEvolve 搜索，也不会自动选择下一节点或再做 debug。

```bash
python replay_improve.py \
  --runs-root /workspace/MLEvolve/runs \
  --source-run ORIGINAL_RUN_NAME \
  --parent-node-id FULL_PARENT_NODE_ID \
  --experiment-id noise_pair_001 \
  --arm A \
  --corpus /workspace/Agentic_Knowledge_Base/output/paper_corpus
```

另三个 Job 使用相同参数，只把 `--arm` 改成 `B`、`C`、`D`。所有四组都要传相同
`--corpus`，即使 A/B 不读论文。默认 `--runs-root` 是当前仓库的 `runs/`。
已提供 [三个 seed、每个四个 arm 的 Job 配置](../k8s/replay-abcd.md)。

父节点必须是 `logs/journal.json` 中已有、成功且有效的非 root 节点，传完整 `id`，
不接受 step 或 ID 前缀。它可以是 draft、debug 或 improve 等阶段产生的成功节点；
本次操作固定为 improve。不是读取 `best_solution.py` 或最终 solution。

配置继承原 run 的 `logs/config.yaml`。当前 Job 的 `LLM_API_KEY` 始终替换历史凭据；
如设置 `LLM_MODEL`、`LLM_BASE_URL`、`LLM_REASONING_EFFORT`，它们也会覆盖原配置，
四个 Job 必须相同。`--seed` 可选，默认原 run 的 seed，用于随机材料和进程内随机数；
它不保证远端 LLM 确定性。验证划分仍使用原 run 的 seed 和 contract。

## 四组材料

| Arm | 注入正文 | 长度 |
| --- | --- | --- |
| A | 空，不调用 analogy | 0 |
| B | NLTK Gutenberg 小说原文，随机书籍和起点，不清理章节标题 | 恰好 12,000 字符 |
| C | 随机论文，经 LLM 判定与当前任务无关后，从全文提取建议 | 恰好 12,000 字符 |
| D | 现有 analogy agent 的节点诊断、检索和建议报告 | 不超过 12,000 字符 |

这里沿用旧 analogy 实验的 `report_char_budget=12000`，**单位是 Python 字符数，
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

B/C/D 通过相同的 `Supplementary material` 区块送入原 improve 的 diff/full-rewrite
流程。为保持四组生成流程可比，这个实验入口不启用 D 专属的结构化选择/handoff
附加提示；D 复用的是原 analogy 的报告生成过程。普通 `run.py` 的 analogy 行为不变。

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

每个 arm 只有一个执行槽，继承 Job 的 CUDA 可见设备；GPU/CPU 配额和挂载权限由 Job
负责，脚本不会调度 GPU，也不是执行任意生成代码的安全沙箱。四组应申请相同资源。

旧流程需要格式校验服务时，入口会启动自己的本地服务并在结束时关闭。训练后复用
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
      generated_node.json      # review 前的代码、计划与生成提示
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
