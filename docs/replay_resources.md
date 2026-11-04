# Replay 资源清单

[replay_resources.json](replay_resources.json) 记录本机资源的位置及关联关系，
具体实验名称、父节点 ID 和历史结果保留在这份清单中。它是静态盘点记录，入口不会自动读取；
使用方法见 [单步 replay](replay_improve.md) 和 [Mac MPS replay](replay_macos.md)。

## 清单结构

| 字段 | 内容 |
|---|---|
| `datasets` | 公开训练/测试数据与可选私有评分数据的位置、文件指纹 |
| `corpora` | 论文库位置、版本指纹和记录数 |
| `sources` | 原 run 的恢复材料、验证 contract、下载来源 |
| `bundles` | 一个父节点对应的 source、dataset、corpus、adapter 和 Mac baseline |
| `baselines` | Mac 校准目录、状态、有效父节点及测量结果 |
| `experiments` | 已记录实验的输出目录、seed 和所属 bundle |
| `as_of` / `verification` | 盘点日期及当时的检查范围 |

`bundles` 是清单中的关联条目，不要求磁盘上存在同名文件夹。
当前 `path_base` 为 `user_home`：路径需相对于本机用户主目录展开。
下载 manifest 的文件映射按各自的 `files_base_path` 解释，不能假定都相对于 source run。

## 从清单填写命令参数

先选择一个 bundle，再沿其中的引用查找资源：

| 参数 | 如何确定 |
|---|---|
| `--runs-root` | `sources` 中对应 run 目录的上一级目录 |
| `--source-run` | bundle 的 `source_run`，也是原 run 的目录名 |
| `--parent-node-id` | bundle 的完整 `parent_node_id` |
| `--data-dir` | 与该 source 的 contract 一致的公开输入目录 |
| `--corpus` | bundle 引用的 corpus 目录 |
| `--prepared-baseline` | bundle 引用且状态为 `completed` 的 Mac baseline 目录 |
| `--output-root` | 本次输出目录，与来源和 baseline 目录分开 |

`--experiment-id` 和 `--seed` 按本次实验选择。同一配对的四个 arm 使用相同参数，
只改变 `--arm`。无需为每次实验新建启动脚本。

## 输入与输出边界

- source run、公开数据、验证 contract、corpus 和已完成的 baseline 作为固定输入。
  即使公开 CSV 相同，不同来源的验证 contract 也不能合并；不同版本的 corpus 不能互换。
- 下载包可能只包含恢复所需的部分材料。根据 `contents` 和 `checkpoint_weights_included`
  判断内容；论文库本身不足以恢复 replay，也不能把缺少权重的下载包用于原 checkpoint 续训。
- Mac 目前仅支持已登记源码的 adapter，并要求 candidate runtime；清单有一个父节点条目
  不代表任意模型都能在 MPS 上运行。每个父节点须有对应的完成校准，失败项只供诊断。
- 历史 manifest 中包含路径和内容身份。移动输入或改写 baseline 需要同步处理其引用，
  清理实验输出时不要把仍需复用的输入一起删除。
- 新增、迁移或删除资源后更新清单；`as_of` 之后的状态以实际文件为准，历史实验 manifest
  仍保留各次运行的原始记录。
