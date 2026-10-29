# 三个 seed 的单步 replay Jobs

三个配置文件各包含 A/B/C/D 四个独立 Job，共 12 个 Job：

| Replay seed | 配置文件 | 配对实验 ID |
| --- | --- | --- |
| 67 | [job-jigsaw-unintended-replay-abcd-s67.yaml](job-jigsaw-unintended-replay-abcd-s67.yaml) | `jubias-noise-parent5-s67` |
| 68 | [job-jigsaw-unintended-replay-abcd-s68.yaml](job-jigsaw-unintended-replay-abcd-s68.yaml) | `jubias-noise-parent5-s68` |
| 69 | [job-jigsaw-unintended-replay-abcd-s69.yaml](job-jigsaw-unintended-replay-abcd-s69.yaml) | `jubias-noise-parent5-s69` |

所有 Job 从同一个历史状态开始：

- 原 run：`20260914_071735_jubias-base-gpt56sol-s60`
- Parent ID：`36845c7c3f1c4beca8eebc7a37852e7c`
- 原 `step=5`，`stage=improve`，branch 2，执行成功、有效。
- Parent validation metric：`0.8924585643326409`，越高越好。
- 原验证 contract 的 seed 保持 **60**。67/68/69 只作为 replay 材料抽样和入口 RNG 的 seed，
  不改写父代码中的训练 seed，也不保证远端 LLM 确定性。

选择的是已有 baseline 的中间 improve 节点，可以复用其代码和测量记录，并保留后续改进空间。
2026-10-28 已通过 `mlevolve-agentic-knowledge-base-dev-cpu` 只读检查 PVC：原日志、公开输入、
memory、验证 contract，以及截至 parent 的 6 个 journal 节点所需源码和诊断文件均存在。
memory 中对应前缀的记录为 3 条；论文 corpus 为 38,273 篇，`records_sha1=5cd1d5e7afd0`。
此次检查没有修改 PVC 或 pod。

## 提交前

配置使用原目录 **`/workspace/MLEvolve`**，venv 为 `/workspace/MLEvolve/.venv`。
检查时该 checkout 仍在 `feature/jev_rerank`，尚无 `replay_improve.py`。
由你在提交前把本地 `feature/noise_effect` 的本次变更同步到原目录，包括：

- `replay_improve.py` 和它使用的 agents / engine / utils 代码；
- `k8s/replay-entrypoint.sh`；
- 迁移后的 `tests/` 和三个 Job 配置。

确认没有正在使用该 checkout 的实验后再同步，运行期间保持代码不变。
这些 Job 不执行 `git pull`、切换分支或更新共享 venv。新的入口只在新 Job 容器内部准备
编译器，然后运行 replay；不会调用启动完整搜索的 `run_single_task.sh`。

每个 Job 配置相同的 **8 CPU / 48 GiB RAM / 1 张 NVIDIA A10 GPU**，与原配置的 CPU
数量一致；GPU affinity 仅允许 `NVIDIA-A10`，镜像、PVC、共享内存挂载沿用 S66 Sol 模板。
LLM 固定为 `gpt-5.6-sol / high`，使用 `mlevolve-llm-proxy` Secret 和
`http://cliproxy:8317/v1`。不需要为这些 Job 配置 `SERVER_ID`。

Replay Python 进程时限为 **6 小时（21,600 秒）**，包括数据复制、材料准备、LLM、训练和
评分；超时先发送 TERM，120 秒后仍未退出则强制结束。启动检查和编译器准备在此计时之前。
整个 Kubernetes Job 时限为 **9 小时（32,400 秒）**，为排队、启动和收尾预留 3 小时余量；
排队过久仍可能压缩实际运行时间。
本次从 parent 恢复的提示中共同剩余预算仍约 6 小时 41 分钟，实际 replay 受上述 6 小时
外部时限约束；单次训练由原 candidate runtime 限制在 7,200 秒内。每个 arm 独立复制
输入并新建可写缓存，需预留相应 PVC 空间。

## 自行提交

```bash
kubectl --context nautilus -n ecepxie apply -f k8s/job-jigsaw-unintended-replay-abcd-s67.yaml
kubectl --context nautilus -n ecepxie apply -f k8s/job-jigsaw-unintended-replay-abcd-s68.yaml
kubectl --context nautilus -n ecepxie apply -f k8s/job-jigsaw-unintended-replay-abcd-s69.yaml
```

结果在：

```text
/workspace/MLEvolve/runs/replay_jubias-noise-parent5-s67/{A,B,C,D}/result.json
/workspace/MLEvolve/runs/replay_jubias-noise-parent5-s68/{A,B,C,D}/result.json
/workspace/MLEvolve/runs/replay_jubias-noise-parent5-s69/{A,B,C,D}/result.json
```

重跑同一个 experiment ID / arm 会清空该 arm 的旧结果。不要同时运行同一 arm 两次。
四组材料、恢复边界和输出字段见 [replay 说明](../docs/replay_improve.md)。

新增 12 个 Job 已通过形状、配对参数、资源一致性检查，以及入口参数转发的本地 mock
检查。全量 `k8s/validate.py` 仍报告旧 GPT-6/Sol 模板与 S56 的四项端口 95/96 重复，
这些是已有配置问题；本次 replay 不使用 `SERVER_ID`，未修改历史模板。
