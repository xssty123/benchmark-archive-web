# 查看环境信息的常用脚本

在实际训练机器的容器内、实际训练 Python 环境中执行。Docker 中使用容器内的 Python；Conda 和 uv 先激活对应环境。采集的是执行当时的状态

报告顺序：CANN → PTA → 源码 commit → 包版本列表，CANN 与 PTA 相邻放置

## 1. Python 包与 PTA

所有包只记录 `pip list` 的包名和版本，不批量执行 `pip show`。PTA 的包名是 `torch_npu`，脚本会单独列出版本

Conda / Docker：

```bash
python -m pip list
```

uv 环境（先激活训练用的虚拟环境）：

```bash
source .venv/bin/activate
uv pip list --python "$(command -v python)"
```

每个包一行，不输出 Summary、Author、License、Location、Requires 等详情

## 2. CANN 版本

一键脚本只采集当前训练环境指定的一套 CANN，读取 `ASCEND_HOME_PATH`、`ASCEND_TOOLKIT_HOME` 和 `ASCEND_OPP_PATH`，不再扫描系统默认目录、用户目录或额外的 driver / NNAL 安装

- 同一安装的软链接别名、`aarch64-linux` / `x86_64-linux` 架构目录和 `opp` 路径会归一化后去重
- 多个变量指向不同安装时，报告列出来源和路径并提示冲突，不查询任何一套 CANN
- 未设置这些变量时报告“未检测到”；路径不存在时报告无效，不回退到其他安装
- `--cann-root` 显式指定一套安装，覆盖环境变量选择；重复指定的路径必须归属于同一套安装
- 只执行所选安装内的 `query_pkg_version.sh`，并保留安装信息中的 `version`、`innerversion` 等原始字段
- `--cann-info` 仅补充所选安装内的版本文件，外部文件或跨安装的符号链接会跳过并说明原因

这里记录的是采集时环境指定的安装，不代表已检查历史训练进程实际加载的动态库。CANN 未检测到或冲突时仍继续采集 PTA、源码和包列表，问题会写入报告

## 3. 源码 commit

单个仓库：

```bash
git log -1
```

一键脚本只检查指定根目录本身及其直属子目录：存在 `.git` 目录或文件的才作为仓库采集，兼容直属 worktree / 子模块，不继续进入内部查找。目录名称不写死，普通目录自动跳过；不向上查找父仓库。需要限定范围时使用 `--repo`，此时只采集明确指定的仓库，不再自动发现其他仓库

源码部分只保留各仓库 `git log -1` 的命令、退出码、commit ID、Author 和 Date，不保存提交说明。没有 Git 元数据的目录跳过；读取失败时保留错误信息。commit 时间不是构建或安装时间

脚本显式指定 Git 输出格式、日期格式并关闭颜色、装饰和签名展示，避免用户的 `format.pretty` 等设置改变报告结构。`--repo` 同样要求指定目录本身存在 `.git` 目录或文件；普通子目录和不存在的路径会跳过并记录原因，不读取父仓库，也不回退到自动扫描

## 一键生成 environment.txt

下载 [collect_environment.py](../public/scripts/collect_environment.py)，放到训练机器上。仅依赖 Python 标准库，Python 3.8+；不会安装包、修改依赖或拉取仓库

Conda / Docker，进入训练环境后，在源码项目总目录执行：

```bash
python /path/to/collect_environment.py --mode pip --root "$PWD" --output environment.txt
```

uv，进入训练项目目录并激活训练环境后执行：

```bash
source .venv/bin/activate
python /path/to/collect_environment.py --mode uv --root "$PWD" --output environment.txt
```

假如各源码仓库在 `/workspace` 下，而当前目录只是其中一个仓库，可把 `--root "$PWD"` 改为 `--root /workspace`。报告不会覆盖已有文件，再次采集请改输出文件名

可选参数：

- `--python /path/to/python`：明确指定训练解释器，pip、uv 和 PTA 版本查询均使用该解释器
- `--repo /other/source`：只采集指定仓库，可重复；相对路径以 `--root` 为基准，提供此参数后关闭自动发现
- `--cann-root /custom/Ascend/cann`：显式选择 CANN 安装，覆盖环境变量；不追加扫描其他安装
- `--cann-info /selected/cann/custom/version.info`：补充所选 CANN 安装内的版本文件，可重复；不能用来另选一套安装

只检查根目录及直属子目录，不递归、不跟随目录符号链接。普通目录没有 `.git` 标记就跳过；任何名称的直属 Git 仓库都可被识别。更深层、外部或符号链接指向的仓库可用 `--repo` 明确选择。扫描错误、命令超时、缺少工具等都会写入报告，不当成已采集成功

官方命令依据：[uv 环境检查](https://docs.astral.sh/uv/pip/inspection/)、[CANN 9.2 版本查询](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/softwareinst/instg/instg_0064.html)、[旧版 CANN 版本查询](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0064.html)
