# VeOmni SFT 原始日志解析

## 输入与字段

读取 VeOmni 终端原始文本，不修改、不执行日志内容。识别 `[INFO|路径:行号] 时间 >>`、ANSI 颜色和 rank 前缀；配置、HF 模型描述及性能字典分别处理，不把同名字段跨层级混用

| 表格字段 | 日志来源 |
| --- | --- |
| 模型、系列、参数 | `model.model_path` 及模型名称 |
| 序列 | `data.max_seq_len` |
| 集群规格 | `Process rank: ..., world size: ...` |
| 训练后端 | VeOmni 日志标识与 `model.accelerator.fsdp_config.fsdp_mode` |
| 并行策略 | `model.accelerator` 下的 DP 分片/副本、TP、PP、CP、Ulysses、EP |
| MBS / GBS | `train.micro_batch_size` / `train.global_batch_size` |
| GAS | `Set gradient accumulation to N.`；不从动态 batch 推算 |
| 重计算 | `model.accelerator.gradient_checkpointing` 的 enable、recompute_layers、selective_recompute_layers |
| 层数、专家数 | HF `Model config` 的 `text_config`，文本模型则使用根字段；忽略视觉层数、每 token 激活专家数 |

未记录的场景分类、调优策略和基线比不猜测。环境、日志存档和 profiling 仍由用户填写链接。重复/无效配置值只使相关字段留空，其他字段继续识别；手工备注不写入解析提示

## 指标口径与已核对源码

按本地 VeOmni 源码和原始失败日志核对，相关源文件为：

- `veomni/trainer/base.py`：`_setup` 通过 `json.dumps(asdict(args), indent=2)` 打印启动配置，并打印 Process rank / world size
- `veomni/arguments/arguments_types.py`：DP 是副本数乘分片数；GAS 根据 GBS、MBS、DP 计算并单独打印，派生属性不一定包含在 dataclass 的 JSON 中
- `veomni/utils/helper.py`：`EnvironMeter.step` 将各 DP rank 的有效 token 汇总，除以该步 delta_time，输出 `tokens_per_second(M)`；`mfu` 为实际与理论 FLOPs 的比例
- `veomni/trainer/callbacks/trace_callback.py`：效率指标通过 `wandb.log(..., step=state.global_step)` 写入 W&B；终端 tqdm 的 postfix 取训练 loss 等字段，不保证输出吞吐、MFU 或精确步耗时
- `veomni/arguments/arguments_types.py`：重计算关闭或 recompute_layers=0 为未开；enable=true、recompute_layers=-1 且 selective_recompute_layers=0 为全重计算；仅知道开启时不猜范围

如果运行另有显式指标打印，支持包含步号和原生指标键的 Python 字典或 JSON 对象，例如以下**合成示例**：

```text
[INFO|/example/VeOmni/train.py:100] 11/16/2026 10:00:00 >> {"global_step": 1, "tokens_per_second(M)": 0.02, "mfu": 0.3, "step_time_s": 10}
```

`step_time_s` 是可选的显式耗时字段，不宣称默认 VeOmni 会输出它。不能用 tqdm 的滑动速率替代该字段，也不从累计 token 数反推吞吐。子模块的 `trace/<module>/tokens_per_second(M)` 不视为全局吞吐

没有这些性能数据时允许仅保存配置，性能留空，不造出 step 0。带步号但缺少某指标的记录会标明缺失，不只平均剩余有效步。完全相同的重复步去重，冲突步号提示拆分重启或混合日志

## 验证边界

已用本地真实 SFT 失败日志验证配置字段：模型、16k 序列、8P、FSDP2、DP2 / Ulysses SP4、1/16/8、全重计算、40 层及 256 专家。该日志进度条到过 1/306 后报错，没有明确打印逐步吞吐、MFU 和单步耗时；进度条速率不纳入性能均值

逐步指标、字段损坏、重复字段、乱序、缺失值、单位转换及范围选择用小型合成日志测试。完整成功运行的原始日志、其他 VeOmni 版本的自定义输出和其他训练框架尚需分别验证。公开仓库不包含本地真实日志或私人路径
