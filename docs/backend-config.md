# 并行与重计算规则

配置解析与性能统计独立。缺失配置不表示关闭功能，不填默认值；不借用 ref、critic 或 rollout 的配置填写 actor。所有说明写入解析说明，备注仅保留用户输入

VeOmni 原生 SFT 兼容 `model.accelerator` 与 `train.accelerator` 两种布局；重计算兼容 `model.accelerator.gradient_checkpointing`、`train.gradient_checkpointing` 和 `train.accelerator.gradient_checkpointing`。逐字段合并：相同值去重，不同值或损坏的同义字段仅阻止该字段填写；不会因为一个字段缺失就切换整套结构。只打印 enable=true 时不补其他版本的默认层范围

| 框架 / 后端 | 训练并行 | 重计算 |
| --- | --- | --- |
| verl / FSDP、FSDP2 | 由设备总数和 `actor.fsdp_config.fsdp_size` 推导复制、分片规模；读取嵌套 SP，兼容 actor 顶层 SP>1 的覆盖 | 读取 model 的 `enable_gradient_checkpointing`；开启不自动等于全部层，已核验的 Omni Thinker 分类保留 |
| verl / VeOmni | 由设备总数 / Ulysses SP 得到数据并行规模，再按 `actor.veomni.fsdp_size` 拆分；EP 独立读取 | 读取 model 的 `enable_gradient_checkpointing`；只确认开启时显示范围待核实 |
| verl / FSDP-Turbo | 读取 `actor.fsdp_config.turbo_config.distributed` 中的显式值 | 后端 `memory.recompute` 及模块计划优先；不能因通用开关 false 判定关闭；只有两处开关均明确关闭才标为未开 |
| verl / Megatron Core、显式 MindSpeed 策略 | 读取 actor 的 Megatron 进程组配置，与模型覆盖中的并行维度冲突时该维度留空；Sequence Parallel 采用模型覆盖后的开关 | 使用 Megatron 自己的重计算配置；不套用 HF 梯度检查点开关 |
| slime / Megatron Core | 独立读取 TP、PP、CP、EP、ETP、VPP、DP、Sequence Parallel；SGLang 并行单独读取 | 与 verl 共用 Megatron 规则 |
| VeOmni 原生 SFT | 保留 DP、DP-Replica、DP-Shard、TP、PP、CP、Ulysses SP、EP；字段独立，不将 DP-Shard 直接解释为所有参数的实际 FSDP 通信组规模 | enable 是总开关，recompute_layers 确定范围，selective_recompute_layers 确定范围内使用 SAC 的前若干层 |
| 标准化 SFT JSONL | 该格式目前只承载逐步性能，配置需手动填写 | 同左 |
| 其他未适配后端 | 显示未适配原因，不套用已支持后端的规则 | 同左 |

## Megatron 重计算

- `recompute_granularity=None`：未开重计算
- `selective`：选择性重计算
- `full + uniform + 正整数 recompute_num_layers`：全重计算，按配置层数分组
- `full + block + 正整数 recompute_num_layers`：按层重计算，记录每流水线阶段配置的层数；没有核对阶段实际层数时不判定全部覆盖
- 只有 `full`、方法或层数缺失：已开重计算，范围待核实
- 整个字段缺失、类型错误或未知值：留空并说明

`full` 表示检查点作用于完整 Transformer 层，不单独证明所有层都被重算。新结果和带有 block 细节的旧显示名称不会再被直接归为全重计算；旧存档本身不改写。只剩“全重计算”字样、没有原配置的历史行无法追溯判断，需要重新解析原日志

## 规模与依据

verl 原生 FSDP/FSDP2 的 FSDP mesh 按 world_size 建立；VeOmni 适配器先按 world_size / Ulysses SP 拆分 DP。两者不能套用同一个分片公式。页面显示的是配置及按源码规则得到的推导，不能代替运行时进程组记录

核对的代码入口：

- verl `verl/workers/engine/fsdp/utils.py::create_device_mesh`
- verl `verl/workers/engine/fsdp/transformer_impl.py` 的 mesh 初始化
- verl `verl/workers/config/actor.py::FSDPActorConfig.__post_init__` 的顶层 SP 兼容规则
- verl `verl/workers/engine/veomni/transformer_impl.py` 的 DP / FSDP 初始化
- verl `verl/workers/engine/megatron/transformer_impl.py` 的进程组初始化与 provider overrides
- verl `verl/trainer/config/engine/megatron.yaml` 的 uniform / block 定义
- VeOmni `veomni/arguments/arguments_types.py::GradientCheckpointingConfig`
- VeOmni `veomni/utils/recompute_utils.py` 的重计算层范围和 SAC 规则

这些规则以已核对的源码结构为适配依据；定制后端、不同结构或未记录字段需补充日志适配。组件名出现在警告或导入失败信息中，不作为实际启用后端的证据

## 验证范围

`test/backend-config.test.mjs` 覆盖跨后端重计算开关、block / uniform、嵌套并行配置、覆盖优先级、字段局部错误、混合 SAC、未知后端及旧显示名称。真实 Omni RL 和 VeOmni SFT 失败日志做配置及性能回归；FSDP-Turbo 用实际配置片段回归。没有宣称这些配置都完成了真实硬件训练验证
