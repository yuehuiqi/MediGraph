# 第三方依赖与来源声明

本项目源代码采用 MIT License；第三方软件、模型和数据不因此被重新授权。

## Python 依赖

直接依赖及固定范围见 `requirements.txt`，包括 OpenAI Python SDK、python-dotenv、
Pydantic、NetworkX、NumPy、PyVis、tqdm、Tenacity、FastMCP、Requests、FastAPI、
Uvicorn、Neo4j Driver、pypdf、pytest，以及服务层与评测新增的 sqlglot（SQL AST
解析）、prometheus-client（`/metrics`）、python-json-logger（结构化日志）、
sse-starlette（SSE 响应）、psycopg（PostgreSQL 驱动）、pgvector（向量类型适配）、
redis（结果缓存客户端）、langchain-openai 与 ragas（Ragas 评测的 LLM 客户端与
框架本体）。发布前应对最终锁定环境运行：

```powershell
python -m pip freeze > outputs/environment_freeze.txt
python -m pip licenses --format=markdown
```

后一个命令需要可选工具 `pip-licenses`。以实际安装版本的包元数据和许可证正文为准。

## 随仓分发的前端资源

- **Apache ECharts**（`medigraph/analysis/assets/echarts.min.js`，Apache License 2.0，
  上游 <https://github.com/apache/echarts>）：为使生成的 BI 报告在离线环境下仍可正常
  渲染，本项目将其压缩产物原样内置，未做修改；许可证正文见该文件头部注释。
- **vis-network**：由 PyVis 在生成图谱 HTML 时内联，随 PyVis 分发，按其上游许可证使用。
- **Bootstrap**：仅由 PyVis 生成页面通过 CDN 引用，用于排版，不影响图谱渲染。

## 平台与模型

- ModelEngine / Nexent / DataMate：按各上游仓库许可证使用；
- Qwen、DeepSeek 或其他模型/API：按模型与服务提供方条款使用；
- Neo4j：仅为可选后端，按 Neo4j 官方授权条款使用。

## 数据

- CMeIE-V2：用于完整 schema 抽取训练/评测；
- DIAKG：用于糖尿病领域实体/关系标注；
- CM3KG：用于规范实体索引与结构化图谱基线。

原始数据不属于本项目原创代码。`data/models/fast_extractor.json` 中包含源文件
SHA-256 与来源字段；评测报告也记录数据和模型产物校验码。
