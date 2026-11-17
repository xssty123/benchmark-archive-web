# VeOmni SFT 原始日志解析

## 输入与字段

读取 VeOmni 终端原始文本，不修改、不执行日志内容。识别 `[INFO|路径:行号] 时间 >>`、ANSI 颜色和 rank 前缀；配置、HF 模型描述及性能字典分别处理，不把同名字段跨层级混用

| 表格字段 | 日志来源 |
| --- | --- |
| 模型、系列、参数 | `model.model_path` 及模型名称；路径以 transformer 等组件目录结尾时，取可确认的模型目录名 |
| 序列 | 文本/VLM 取 `data.max_seq_len`；DiT 不把该文本配置当成实际 latent 序列长度，未明确记录时留空 |
| 集群规格 | `Process rank: ..., world size: ...` |
| 训练后端 | VeOmni 日志标识与 `model.accelerator.fsdp_config.fsdp_mode` 或 `train.accelerator.fsdp_config.fsdp_mode` |
| 并行策略 | `model.accelerator` 或 `train.accelerator` 下的 DP 分片/副本、TP、PP、CP、Ulysses、EP |
| MBS / GBS | `train.micro_batch_size` / `train.global_batch_size` |
| GAS | `Set gradient accumulation to N.`；不从动态 batch 推算 |
| 重计算 | `model.accelerator.gradient_checkpointing`、`train.gradient_checkpointing` 或 `train.accelerator.gradient_checkpointing` 的 enable、recompute_layers、selective_recompute_layers |
| 层数、专家数 | 文本/VLM 取 HF `Model config` 的 `text_config` 或根字段；DiT 依据 Detected DiT model type 匹配主干配置，层数取 `num_layers`；不取条件模型、token refiner 或视觉塔的层数 |

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

`step_time_s` 是可选的显式耗时字段，不宣称默认 VeOmni 会输出它。整份日志没有该字段时，使用训练进度条 `Epoch e/E: ... n/N [..., Xs/it, ...]` 的有效 `s/it` 算术平均估算单步耗时；有显式耗时字段时不混入估算值，也不从累计 token 数反推吞吐。子模块的 `trace/<module>/tokens_per_second(M)` 不视为全局吞吐

既没有显式指标也没有有效训练进度条 s/it 时允许仅保存配置，性能留空，不造出 step 0。带步号但缺少某指标时标明缺失，各指标分别平均实际记录到的步，缺失不补 0，也不清空已有均值；完全未记录或包含无效数值的指标仍留空并提示。完全相同的重复步去重，冲突步号提示拆分重启或混合日志

估算仅处理训练 Epoch 进度条的 `s/it`，不处理 `it/s`、加载或验证进度条。忽略步号 0、非有限数、非正数及未知速率；相同步数的重复刷新只保留最后一个有效值。多 epoch 按 `(epoch−1)×每 epoch 总步数+当前步` 展开；总步数不一致时停止估算并提示，配置和显式指标继续处理

估算耗时只平均所选范围内已打印的有效样本；其他显式指标同样按各自实际记录的步数求均值，并显示缺失/无效步数。估算标记随历史分析结果保存、范围重算保留，解析说明解释该值是平滑速率的均值，不是逐步实测耗时，备注仅保留用户输入

## 验证边界

已用本地真实 SFT 失败日志验证配置字段：模型、16k 序列、8P、FSDP2、DP2 / Ulysses SP4、1/16/8、全重计算、40 层及 256 专家。该日志进度条到过 1/306 后报错，没有明确打印逐步吞吐、MFU 和单步耗时；当前采用进度条有效 s/it 的均值作为估算耗时，TPS 和 MFU 留空

逐步指标、字段损坏、重复字段、乱序、缺失值、单位转换及范围选择用小型合成日志测试。完整成功运行的原始日志、其他 VeOmni 版本的自定义输出和其他训练框架尚需分别验证。公开仓库不包含本地真实日志或私人路径

已补充 MiniMax-H3 DiT 启动日志的实际字段回归：模型路径的 transformer 子目录、MiniMax 系列、主干 num_layers 与独立条件模型的区分。发布测试使用合成路径和配置，不包含原始日志；未明确记录的参数量、专家数和逐步性能留空
