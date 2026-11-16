# Benchmark Archive

**[打开在线工具 →](https://xssty123.github.io/benchmark-archive-web/)** · 无需安装，打开即用

RL / SFT 训练性能归档工具的网页原型，从运行日志提取性能记录，复制到在线表格；环境信息与 profiling 由团队云空间管理

**当前版本是纯前端工具**：无需 AI agent、API key 或第三方依赖，浏览器本地解析日志并复制结果，尚未接入在线表格、团队账户或共享存储

## 在线使用

访问 https://xssty123.github.io/benchmark-archive-web/ ，无需安装 Node.js

本仓库是唯一维护仓库，功能、测试、文档和部署配置都在这里更新。无需向其他仓库同步

向 `main` 推送后，GitHub Actions 自动运行 JavaScript 和 Python 测试，全部通过后部署 `public/` 到 GitHub Pages。Pull Request 只运行测试，不部署；测试失败时线上版本保持不变

日志在各自浏览器中解析，历史记录也只保存在当前浏览器。在线站点与 localhost 的历史记录互不共享；仓库和站点不保存真实训练日志、环境快照或 profiling

开发步骤、文件结构、测试要求及发布排查见 [开发说明](CONTRIBUTING.md)，环境采集用法见 [环境信息采集说明](docs/environment-commands.md)

## 本地启动

需要 Node.js 20 或更新版本

```bash
git clone https://github.com/xssty123/benchmark-archive-web.git
cd benchmark-archive-web
node server.mjs
```

打开 http://127.0.0.1:4173 ，选择 RL 或 SFT，点击“试用演示数据”即可体验流程

默认仅监听本机地址，可通过 `PORT=8080 node server.mjs` 更换端口。历史记录按站点区分，换浏览器、域名或端口不会看到原来的记录

## 使用流程

1. 选择 RL / SFT，仅选择一份单次运行日志，无需上传环境 TXT 或 profiling 文件
2. 默认解析全部已记录步骤，在核对页选择统计开始步数和结束步数（含边界）；核对自动提取的配置，未知项留空
3. 环境信息、日志和 profiling 自行归档到团队云空间；日志选择页在“添加运行日志”下方可填写三个 HTTPS 链接，也可留空后直接在在线表格补充
4. 单行结果表旁保留一个“复制整行”按钮，用于粘贴到在线表格
5. 底部点击“保存并解析下一份日志”，先保存到历史记录，成功后返回日志选择页；保存失败时保留当前内容。保存过程中若有新修改，也保留当前页，避免丢失修改

页面只有“选择运行日志 → 核对并复制”两步，不再跳转到单独的归档完成页。“解析新日志”始终清空当前任务并返回日志选择页，复制后仍停在核对页，便于继续检查。未保存的结果仅保留在当前页面；历史页可点击“返回当前核对”继续处理当前任务

RL 使用 30 列，SFT 使用 23 列；最后三列固定为 **环境信息、日志存档、profiling采集**，备注位于它们之前。模型名是复制结果的必填项，其余缺失字段可留空。无可比 A3 基线时 A5/A3 留空，当前不自动匹配基线

“历史记录”入口始终可见，新旧记录均可导出 CSV / ZIP。历史记录每行提供“编辑”，可恢复配置、指标、统计范围及材料链接；编辑后点击“保存修改并返回历史记录”更新原记录，不新增重复行。每行也提供“删除”，直接从浏览器移除该条记录及其本地附件，不经过回收站，不影响已下载文件或线上材料。旧版已软删除记录保持隐藏，不在升级时批量清理既有浏览器数据

核对页在“备注”之后显示单行结果表，按 RL 30 列 / SFT 23 列顺序排列，字段编辑、batch 编辑和统计范围应用后同步更新。点击“复制整行”复制当前数据（不含表头），可直接粘贴到 Excel 或在线表格，保留空列和最后三列材料链接，备注默认留空，仅保留用户输入；无需本地保存。更改统计范围后须先点击“应用范围”并核对重算结果，复制或保存不会自动重算。浏览器不允许自动复制时显示已选中的文本供手动复制

新解析结果默认“满血”，可改为减层。已知模型按官方配置填入总层数和总路由专家数，并显示来源：Qwen3-Omni-30B-A3B-Instruct 的 Thinker 文本主干为 48 / 128，Qwen3.5-35B-A3B 的文本主干为 40 / 256。默认值不是实测证明；日志中的显式模型覆盖配置优先，未知模型不猜总数。减层时填写实际数/总数

核对页将 MBS、GBS、GAS 分开编辑，保留“未配置”“未记录”“动态（未推算）”的说明。单行结果表、复制整行、性能记录和 CSV 统一采用 `Nan/256/dyn` 格式：未配置或未记录用 `Nan`，动态值用 `dyn`，已知数值原样保留（例如 `4/32/1`），不将缺失值补成 0。历史记录显示和导出时也采用此格式，不改写原存储记录

历史记录仍可导出本地备份包，按已有材料包含：

```text
<Run ID>/
├── run.log
├── material-links.json  # 填写了材料链接时才生成
├── environment.txt      # 旧记录包含时保留
└── profiling/           # 旧记录包含时保留
```

原始日志字节保持不变，SFT JSONL 也统一保存为 `run.log`。点击保存后保留原始日志；环境与 profiling 只填写云端链接，不生成占位 environment.txt，不抓取链接内容。历史记录中的环境与 profiling 附件仍可原样导出，不清理既有数据

历史 CSV 包含对应表头和一行记录，使用 UTF-8 BOM，并转义可能被 Excel 解释为公式的文本。复制、保存、历史展示和 CSV 导出均不向备注追加内容。统计范围、指标来源、解析说明和手动修改标记作为独立元数据随历史记录保存；解析说明在核对页单独展示

## 指标与支持范围

### RL：verl 逐步文本日志

```text
step:1 - perf/throughput:100 - perf/time_per_step:180 - timing_s/update_actor:60 - timing_s/gen:100 - timing_s/update_weights:4 - perf/mfu/actor:0.12
step:2 - perf/throughput:120 - perf/time_per_step:160 - timing_s/update_actor:55 - timing_s/gen:90 - timing_s/update_weights:3 - perf/mfu/actor:0.14
```

| 字段 | 日志键 |
| --- | --- |
| 训练MFU | `perf/mfu/actor` |
| 吞吐TPS | `perf/throughput` |
| 训练耗时 actor_update | `timing_s/update_actor` |
| 推理耗时 | `timing_s/gen` |
| resharding耗时 | `timing_s/update_weights` |
| 单步耗时s | `perf/time_per_step` |

- 默认对所有已记录训练步逐步取算术平均，包括首步、慢步、checkpoint/profiling 步；可指定包含两端的步号范围，所有指标和覆盖检查统一使用该范围
- 范围按实际 step 编号选择，不是文件行号；无记录、倒序或越界范围会报错，缺步会提示，可一键恢复全部
- 应用范围会重算性能指标并保留手工编辑的配置；复制时会应用尚未提交的有效范围，并把统计范围作为独立元数据保存。原始附件始终保留完整日志
- TPS 沿用日志的每设备吞吐，不改算总 token / 总耗时，不从硬件名称猜设备数
- resharding 字段当前采用权重更新/同步阶段总耗时，不等同于纯重分片开销
- MFU 在日志和存储中保留 0–1 比例；表单使用 0–100 百分比数值，顶部卡片、表格、复制和 CSV 显示百分号（例如 `0.13` 显示为 `13%`），最多两位小数；旧历史记录自动按百分比展示
- 某指标缺步、无效或 MFU 明确不受支持时，该项留空，不用部分覆盖均值代替
- 相同步号相同值去重、补充片段合并，冲突值报错；混合运行须先拆分。完全相同的重启记录无法仅靠此格式区分，请提供单次运行日志
- 步号区间缺失会提示；全部已记录步不等于完整训练全部步
- 支持 verl / verl-omni 的完整 Python pprint 或 JSON 运行配置块，自动提取模型、参数标识、系列、同步/异步、序列、设备规模、batch、后端、并行策略、重计算；每个字段显示配置来源
- 同步/异步取训练调度配置，不使用 rollout 引擎的 async 模式；GBS 按已适配 verl PPO 的 `ppo_mini_batch_size × rollout.n` 推算并注明，不推算动态 batch 的固定 GAS
- 未支持的配置格式、混合配置会提示；默认满血，已知模型的官方总数作为可编辑默认值，日志覆盖配置优先，未知层数和专家数提示待填写。不能把重计算分组层数当作模型层数
- 重计算统一显示为「全重计算 / 选择性重计算 / 未开重计算」，仅能确认开关的其他模型显示「已开重计算（范围待核实）」，未记录则留空；分组细节放入独立的解析说明
- Qwen3-Omni-30B-A3B-Instruct 的 Thinker + FSDP/FSDP2 + `enable_gradient_checkpointing=true` 按标准实现归为全重计算，范围是 Thinker 文本主干全部 Transformer 层（含 Attention、MoE），不表示整个多模态系统所有算子都重算。依据：[DecoderLayer 及逐层调用](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L1351-L1561)、[完整层 checkpoint 包装](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_layers.py#L33-L85)、[递归开启 checkpoint](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_utils.py#L3413-L3424)。此版本用于核对标准实现，不代表已确认历史运行使用该版本；精确版本或自定义补丁需环境快照核验，分类依据保留在解析说明
- 历史记录中的旧重计算名称在表格和 CSV 导出时统一显示，原始存档不改写；只有记录包含 Thinker 阶段及匹配的模型、后端信息时，才会将旧「开启梯度检查点」归为全重计算

### SFT：标准化逐步 JSONL

当前未使用真实 SFT 日志做回归，不宣称兼容任意 VeOmni、Megatron 或其他框架原始日志。演示只验证工作流程

经核对原日志与框架指标定义后，每个 optimizer/global step 一行，使用以下格式：

```jsonl
{"task":"SFT","step":1,"tps":21000,"tps_unit":"tokens/s","step_time_s":9.0,"mfu":0.32}
{"task":"SFT","step":2,"tps":22000,"tps_unit":"tokens/s","step_time_s":8.5,"mfu":0.34}
```

`tps_unit` 必填，允许 `tokens/s` 或 `tokens/s/device`，同一文件必须一致。JSONL 中的 MFU 为 0–1 小数比例，页面展示为百分比；指标缺失用 `null` 或省略，不能用 0 代替未知值。不把进度条速率、micro-step、滑动均值直接当作单步性能。支持真实 SFT 日志是下一步适配工作

## 数据与边界

- 不上传日志或环境文件；静态服务器仅提供页面，不接收文件
- 不自动收集服务器环境，不启动训练或 profiling，不调用 AI
- 单份日志最多 50 MB；环境与 profiling 在云空间管理，不再通过本工具上传
- 未保存结果仅在当前页面内保留，可复制到在线表格或保存历史；清除站点数据会移除本地历史记录
- ZIP 包含原始日志、已填写的材料链接，以及历史记录已存的附件；同时导出 CSV 才能备份表格记录。尚未支持备份导入、跨设备同步、协作权限和重新编辑已保存记录
- 上传到团队存储后再填写真实 HTTPS 链接，当前不会生成假链接或验证外部链接的内容与团队权限
- 所有内置演示数据均为合成数据，不能用于硬件性能结论
- 仓库仅含工具代码和合成测试，不提交真实训练日志、环境快照或 profiling

## 开发与验证

```bash
node --test test/*.test.mjs
python3 -B -m unittest discover -s test -p test_collect_environment.py
```

```text
public/parser.js    逐步解析、范围选择与覆盖检查
public/config.js    安全读取配置字面量并映射字段
public/columns.json RL/SFT 表头
public/app.js       上传、核对、保存、记录页
public/storage.js   浏览器本地持久化
public/archive.js   CSV 与无压缩 ZIP 导出
server.mjs         本地静态服务器
```

切换任务、更换日志、试用演示或点击“解析新日志”时，如当前内容尚未保存，会先确认是否放弃。刷新或关闭页面时也会由浏览器提示；已保存且未修改的历史记录不重复提示
