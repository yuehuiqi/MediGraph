# 公开资料与启发点登记（INSPIRATION_REGISTER）

> 本文件登记本项目在设计与工程实现中借鉴的公开研究方法、公开标准与通用工程模式，
> 明确标注"仅借鉴的内容"与"本项目独立完成的内容"的边界。与 `ORIGINALITY.md`
> （原创性声明）配套，共同构成本仓库的清白来源（clean-room）记录。
>
> 登记原则：只登记**已经在本仓库代码中实际使用**的方法/模式，不登记未采纳的构想；
> 每条给出代码落地位置，可交叉核对。

---

## 一、模型与算法方法

| 公开方法 | 来源 | 仅借鉴的内容 | 本项目独立完成的内容 | 代码位置 |
| --- | --- | --- | --- | --- |
| GlobalPointer / GPLinker | 苏剑林（追一科技），公开博客与开源实现思路，联合实体关系抽取的通用架构范式 | 单次前向同时输出实体片段与关系头尾指针的架构范式（避免 NER→RE 误差传播） | 基于 CMeIE-V2 全 53 关系 Schema 的训练数据构建、双编码器集成（macbert-large + roberta-wwm-ext）、温度标定融合策略、词典/LLM 三级置信度路由级联、严格 SPO 评测协议实现 | `medigraph/extraction/neural_gplinker.py`、`scripts/train_neural_extractor.py` |
| 温度标定（Temperature Scaling） | Guo et al., *On Calibration of Modern Neural Networks*（ICML 2017）；Platt Scaling 的推广，公开、通用的置信度校准方法 | 用单一温度参数对模型置信度做后处理缩放的思想 | 无框架依赖的纯 Python 实现、ECE/可靠性分箱诊断、双编码器融合场景下的标定拟合与应用 | `medigraph/extraction/calibration.py` |
| DAIL-SQL 动态少样本选择 | Gao et al., *Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation*（DAIL-SQL 论文思想） | 按问题与候选示例的相似度动态挑选少样本示例注入提示词的思路 | 面向本项目医疗分析 Schema 的示例池、字符级词汇重叠打分实现、与图谱词表 Schema-Linking 的结合 | `medigraph/analysis/nl2sql.py`（`_select_shots`） |
| Execution-Guided 自纠错 | 广泛见于 Text-to-SQL 研究（如 IRNet、SQLNet 系工作）的通用做法 | "执行失败→把报错回灌提示词→重新生成"这一通用自纠错循环范式 | 面向 SQLite 的具体错误反馈格式、AST 只读护栏与自纠错循环的组合、双库执行等价评测口径 | `medigraph/analysis/nl2sql.py`（`query()` 的 `max_correction` 循环） |

---

## 二、数据集与知识来源

以下数据集/知识库来源已在 `THIRD_PARTY_NOTICES.md`、`data/external_manifest.json`、
`docs/MODEL_AND_DATA_CARDS.md` 中登记许可证与来源边界，此处仅索引，不重复展开：

- **CMeIE-V1 / CMeIE-V2**：中文医学信息抽取数据集，用于完整 Schema 抽取训练与严格评测；
- **DIAKG**：糖尿病领域实体/关系标注数据集；
- **CM3KG**：中文医学知识图谱，用于规范实体索引与结构化图谱基线（`outputs/graph.json`）。

原始数据不属于本项目原创产出；模型训练与评测仅使用其公开发布版本。

---

## 三、通用工程模式

以下是软件工程中广泛使用、无特定单一来源的通用模式，登记是为了明确"这不是本项目
发明的新概念"，避免夸大创新性表述：

| 模式 | 说明 | 代码位置 |
| --- | --- | --- |
| Server-Sent Events + Last-Event-ID 续传 | W3C EventSource 规范定义的标准重连语义 | `service/stream.py` |
| 只读事务 + 语句超时 + 授权回调的数据库执行护栏 | SQLite/PostgreSQL 官方文档描述的标准安全模式 | `medigraph/analysis/nl2sql.py`、`medigraph/analysis/pg_relational.py` |
| 连接池（bounded pool，min/max size） | 数据库客户端库的标准工程实践 | `medigraph/analysis/pg_relational.py`（基于 `psycopg_pool`） |
| HNSW 近似最近邻索引 | Malkov & Yashunin 公开论文描述的图索引算法，此处直接使用 `pgvector` 官方实现，不涉及自行实现算法 | `medigraph/graph/pg_vector_store.py` |
| 拓扑排序分层并行调度（Kahn 算法变体） | 图论标准算法（Kahn's algorithm），DAG 调度的通用做法 | `medigraph/agents/dag_executor.py`（`topological_layers`） |

---

## 四、关于"10 个参赛项目横向评审"文档的说明

在项目复盘过程中，团队获得了一份第三方对同一赛题下 10 个参赛项目（含本项目）的
**横向技术评审报告**，以及一份基于该评审提出的"全维度升级蓝图"构想文档。

**明确声明**：

1. 上述两份文档**仅作为问题清单使用**——用于识别本项目在证据可信度、指标口径、
   评测方法论上的薄弱环节（例如本项目在自查中独立发现的"NL2SQL 非模板集样本量
   过小""GraphRAG 评测自证""疾病词表稀疏导致空结果巧合匹配"等问题）；
2. 团队**未查看、未复制、未参考**报告中提及的其他 9 个参赛项目的任何源代码、
   Prompt、测试数据、界面设计或文档文字；
3. 本项目采纳的所有改进（P0–P3 各阶段的具体修复与新增功能）均由团队基于**本项目
   自身代码库的实际检查结果**独立设计与实现，具体证据见对应模块的代码与测试；
4. 若后续需要进一步验证，可提供该横向评审报告与升级蓝图文档供审阅（不随本仓库
   一并提交，因其内容涉及对其他参赛队伍作品的评价）。

---

## 五、维护说明

- 新增对外部论文、开源项目或标准的借鉴时，必须在本文件补充一条记录，并在对应
  代码文件的 docstring 中给出简短出处说明；
- 本文件不登记内部团队讨论产生的原创设计——那些记录在 `ORIGINALITY.md`。
