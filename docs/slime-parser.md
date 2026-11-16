# slime RL 日志解析

选择 RL 后自动区分 verl 与 slime，使用相同的核对、步数选择、整行复制和历史记录流程。只解析文本，不执行日志中的 Python、Namespace 或启动命令

## 配置输入

支持 Megatron 的 `参数名 ........ 值` 参数表、单行 `Namespace(...)`，以及顶层 slime 参数字典 / JSON。每个字段独立读取；类型无效或重复配置冲突仅使相关字段留空，并显示字段名和原日志行号。字段顺序不影响识别，备注只保留用户输入

支持标准日期时间、Ray pid、rank 和 ANSI 颜色前缀。暂不通用解析任意 YAML、Shell 启动脚本、嵌套包装或自定义框架输出；无法确认时不套用框架默认值

| 表格字段 | 来源与口径 |
| --- | --- |
| 模型、系列、参数 | `hf_checkpoint` 模型目录；参数量仅采用名称中明确的 B/M 标识 |
| 层数、专家数 | `num_layers`、`num_experts`；重计算分组层数不作为模型层数 |
| 同步/异步 | 明确的 fully_async rollout 标记识别为异步；标准生成函数本身不能证明同步，不根据 async_train 函数名或设备共置推断 |
| 序列 | `rollout_max_prompt_len → rollout_max_response_len`，两者均为配置上限 |
| 集群规格 | 共置时 actor 设备不重复计算；分离部署时须明确 actor、rollout 及 critic 分配后相加，未知拓扑留空 |
| MBS / GBS / GAS | `micro_batch_size` / `global_batch_size` / 显式梯度累积值；动态 batch 不反推固定 GAS |
| train_batch_size × rollout_n | `rollout_batch_size × n_samples_per_prompt`，表示生成样本数，不直接视为训练 GBS |
| 训练并行策略、后端 | 已打印的 Megatron 并行参数与 `train_backend` |
| 推理并行策略、后端 | SGLang 当前或旧名称的 TP/PP/DP/EP 参数，冲突时相关维度留空；每引擎设备数不直接视为 TP |
| 重计算 | `recompute_granularity`、`recompute_method`、`recompute_num_layers`；uniform 全层分组与 block 部分层重计算区分 |

## 性能输入

采用 `perf <rollout_id>: {...}` 的明确指标。训练和 rollout 分开打印的同一编号合并；从 0 开始的编号保留。默认包含全部已记录编号，可选择闭区间。相同记录去重，冲突值提示拆分运行。各指标分别平均所选范围内实际记录到的步；训练和 rollout 编号覆盖范围不同，不会清空另一阶段已有的均值。每项指标显示有效步数、未记录步数；缺失不补 0，完全未记录则留空。无效数值仍使该项均值留空并提示，不作为缺失值跳过，也不影响其他指标

| 表格字段 | 来源与口径 |
| --- | --- |
| 吞吐 TPS | `perf/actor_train_tok_per_s`，整个 rollout batch 的 token 数 / actor_train_time；是训练阶段全局 tokens/s，不是 verl 端到端单设备吞吐 |
| 训练耗时 actor_update | `perf/actor_train_time` |
| 推理耗时 | `perf/rollout_time`，包含框架该阶段计时范围，不声称纯模型解码时间 |
| resharding 耗时 | `perf/update_weights_time`；Timer 在 perf 输出时清零，可能包含前一轮权重更新，不自动移动编号 |
| 单步耗时 | 优先 `perf/step_time`；没有该字段时，仅在同条记录中两项均有效时取 `perf/train_wait_time + perf/train_time`，不将可能重叠的阶段相加 |
| 训练 MFU | 当前适配源码没有直接打印 MFU，留空；`perf/actor_train_tflops` 不作为 MFU |

生成阶段的 `perf/tokens_per_gpu_per_sec`、effective/longest sample 速率不混入主表 TPS；没有编号的 Timer、TFLOPS 不用于反推性能。只找到配置而没有上述指标时仍可核对、复制和保存，性能留空，统计范围隐藏

解析说明记录来源、计时范围及缺失原因，不自动填写备注。不同框架的 TPS 必须先核对口径再比较，不能直接计算不同口径的 A5/A3 比值

## 核对依据与验证边界

实现依据本地 slime 源码快照 `8088a4b4450ac374e1439cf2310ba3add370997d`，不是所有 slime 分支或发布版本的兼容保证

- `slime/observability/logging_utils.py:21`：日志前缀
- `slime/observability/train_metric_utils.py:361`：训练 perf 字段、训练 TPS 与 step_time
- `slime/utils/data.py:329`：Timer 保存整个 rollout batch 的序列长度
- `slime/observability/rollout_metrics.py:61`、`:260`：rollout 时间、生成吞吐及日志输出
- `slime/observability/timer.py:37`：Timer 累计与重置
- `slime/backends/megatron_utils/actor.py:586`、`:617`：perf 输出与 update_weights 计时
- `slime/utils/arguments.py`、`slime/backends/sglang_utils/arguments.py`：配置定义及 SGLang 参数别名
- [Megatron 参数表输出](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py)：`_print_args` 打印参数名、点号和参数值

当前未找到本地 slime 原始训练日志，使用按以上源码构造的合成日志测试配置、乱序、损坏字段、重复步、缺失指标、范围重算、历史数据序列化与框架隔离；尚未完成实际 slime 训练日志回归。公开仓库不包含用户原始日志或私人路径
