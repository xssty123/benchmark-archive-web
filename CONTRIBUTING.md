# 开发与维护

本项目直接在 `xssty123/benchmark-archive-web` 维护，线上地址为 https://xssty123.github.io/benchmark-archive-web/

## 本地启动

需要 Node.js 20+；完整测试另需 Python 3.8+ 和 Git。均不需要安装项目依赖；环境采集脚本只使用 Python 标准库

```bash
git clone https://github.com/xssty123/benchmark-archive-web.git
cd benchmark-archive-web
node server.mjs
```

打开 http://127.0.0.1:4173/ ，保持终端运行。修改文件后刷新浏览器即可，不需要构建。仅监听本机，不接收上传文件

## 目录

```text
public/                       实际发布的网页文件
  index.html / style.css      页面与样式
  app.js                      选择日志、核对、复制与历史记录
  parser.js                   RL/SFT 逐步日志解析与统计范围
  config.js                   运行配置提取
  columns.json                RL 30 列、SFT 23 列的列顺序
  record-fields.js            模型默认配置、batch 和历史字段展示
  recompute.js                重计算名称与范围归一
  copy-row.js / archive.js    整行复制、CSV 和 ZIP
  storage.js                  IndexedDB 本地历史存储
  scripts/collect_environment.py  下载用环境采集脚本
server.mjs                    本地预览服务器，不部署到 Pages
test/                         合成数据和临时仓库测试
docs/environment-commands.md   环境采集说明
.github/workflows/pages.yml   测试及自动部署
```

只发布 `public/`，测试、开发说明和服务器代码保留在仓库中，不进入网站部署包

## 开发步骤

1. 从最新 `main` 创建功能分支，修改相关文件
2. 运行下述两组测试，并根据改动完成浏览器核对
3. 提交并推送，Pull Request 会自动运行测试
4. 合并或直接推送到 `main` 后，测试通过即自动部署，同事刷新原网址即可

不再使用私有仓库同步或 `publish-pages.py` 发布脚本

## 测试

```bash
node --test test/*.test.mjs
python3 -B -m unittest discover -s test -p test_collect_environment.py
git diff --check
```

也可以使用 `npm test` 和 `npm run test:python`，无需 `npm install`

- `core.test.mjs`：日志解析、配置提取、统计范围、字段展示、复制列顺序、CSV 和 ZIP
- `config.test.mjs`：配置字段顺序、日志前缀、等价配置去重、同一行粘连及交错日志、逐字段独立提取（损坏位置前后的字段均保留）、重复字段局部失效、跨 worker 隔离、失败行号和原因；只使用合成样例
- `flow.test.mjs`：换日志清理、异步读取失效、返回核对保留修改、顶部指标卡片同步；使用最小 DOM 适配器，不代替真实浏览器测试
- `test_collect_environment.py`：使用临时 Git 仓库和模拟命令验证环境采集，不扫描真实训练环境

修改页面流程时，使用内置演示数据检查：选择日志 → 全部步数解析 → 调整统计范围 → 修改字段及卡片同步 → 返回继续核对 → 复制整行 → 保存并解析下一份 → 历史查看与复制 → 编辑原记录并更新 → 刷新确认记录数量不变。检查未保存时取消新建、切换任务或换日志能保留草稿；范围未应用时复制/保存须阻止且不覆盖手工指标。RL 应为 30 列，SFT 应为 23 列；再检查刷新后历史保留和采集脚本下载

真实 SFT 框架日志尚未通用适配；当前 SFT 只支持 README 中约定的 JSONL。不能用演示数据测试通过代替真实日志适配验证

## 性能记录约定

- 默认使用全部已记录训练步；吞吐按逐步日志值算术平均，不能改为总 token 除以总时间
- 统计范围包含两端步号；须明确点击应用范围后才更新性能指标，覆盖手工指标值，保留手工配置和备注
- 缺失或不能确认的字段留空，不补成 0；MBS/GBS/GAS 按 `Nan/256/dyn` 等紧凑格式展示
- 备注只保留用户输入；解析解释和指标来源使用独立字段
- A5/A3 使用可比运行的吞吐比，没有基线留空
- 最后三列固定为环境信息、日志存档、profiling采集
- 不能从日志执行代码，也不自动读取或上传用户提供的材料链接

新增解析器或修复问题时，优先补小型合成测试样例。不要提交真实训练日志、environment.txt、profiling、凭据或包含私人路径的截图

## 自动部署和排查

工作流为 `Test and Deploy Pages`：PR 仅执行 `test`，`main` 推送或手动运行先执行 `test`，通过后执行 `deploy`。部署使用本仓库自带的 `GITHUB_TOKEN`，无需跨仓库密钥

在 GitHub 的 Actions 查看具体失败步骤。测试失败先修复测试；部署失败可在修复后重新运行。`Settings → Pages` 的发布源应为 `GitHub Actions`

部署成功后访问原网址并刷新。若仍看到旧页面，先确认 Actions 对应的是预期提交，再尝试强制刷新。Pages 部署不会迁移、共享或清空浏览器历史记录；网址和存储名称保持稳定
