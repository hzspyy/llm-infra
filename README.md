# llm-infra

面向 AI infra 的中文教程：GPU 与操作系统底座、算子工程、attention、模型架构与数值、
单机推理引擎、分布式、训练、生产服务、Agent 负载与扩散模型。

理论推导、源码分析与可复现实验并重。每章给出固定版本的真实 `file:line`、
可复现的 mini 实现、能跑的 lab，以及未经改写的原始输出。

**站点：** <https://hzspyy.github.io/llm-infra/>

## 目录

| 目录 | 内容 |
|---|---|
| `src/<层>/` | 章节正文（Markdown） |
| `labs/<层>/` | 实验代码 |
| `results/<机器>/` | 原始输出，只增不改 |
| `site/` | 构建产物；完整源码页在 `site/code/` |
| `outline.json` | 章节目录与内容定义 |
| `build.py`、`tools/` | 构建与检查 |

## 本地构建

```bash
python build.py
python tools/linkcheck.py
python tools/coverage_check.py
```

依赖：`markdown`、`latex2mathml`、`pygments`。生成的站点完全离线，不发起任何网络请求。

## 关于数据

`results/` 下是实验的原始工件，包含运行它们的机器名、目录布局与硬件信息。
性能数字只在其记录的硬件、版本与配置下成立，未实测的路线在正文中标为 `UNVERIFIED`。
