# 评测与复现协议

## 原则

1. 目标值和实测值分开；
2. 每个头部指标必须有数据切分、样本数、脚本、明细 JSON 和校验码；
3. 模板生成集与人工集分开报告；
4. smoke test 不作为全量指标；
5. 不以导入图谱规模冒充抽取管道自产规模。

## 一键离线复现

```powershell
.\.venv-neural\Scripts\python.exe -X utf8 scripts/reproduce_offline.py --quick
```

运行清单与所有关键产物 SHA-256 写入
`outputs/reproduction_manifest.json`。

## 抽取

### 核心项目口径

```powershell
python benchmarks/eval_fast_extraction.py --mode core
```

数据为项目映射后的 CMeIE-V2 + DIAKG gold。实体使用 `(规范名, 类型)` 精确匹配；
端到端三元组使用 `(head, relation, tail)` 精确匹配，无 gold entity 注入。

### CMeIE 完整口径

```powershell
python benchmarks/eval_fast_extraction.py --mode cmeie --split dev
```

保留官方 53 schema 行/44 个谓词标签，不用“常用子集”代替完整口径。输出：

- 实体 micro P/R/F1；
- 端到端三元组 micro P/R/F1；
- 按实体类型、谓词标签指标；
- 每样本 FP/FN；
- P50/P95/max 延迟；
- 数据与模型产物 SHA-256。

### 神经联合抽取（主力，需 GPU/ML 环境）

```powershell
python scripts/download_encoder.py --repo hfl/chinese-roberta-wwm-ext
python scripts/train_neural_extractor.py --epochs 16 --batch_size 8
python benchmarks/eval_neural_extraction.py --split dev
```

GPLinker 在**与词典基线完全相同的 harness**（同 `canonical_key`、同 CMeIE-V2 gold 构建器、
同 `(head, relation, tail)` 精确匹配、无 gold 实体注入）上评测，故神经与基线数字直接可比。
解码阈值 0 为多标签交叉熵自然阈值，经 dev 阈值扫描确认为 F1 最优工作点。CMeIE-V2 官方 test
标签未公开，按惯例报告 held-out dev。证据：`outputs/eval_neural_cmeie_dev.json`、
`data/models/neural_extractor/train_log.json`。

L1 词典是无 GPU 时的低延迟、可复现下界与对照，不冒充神经模型。

## 实体链接

```powershell
python benchmarks/eval_entity_linking.py
```

held-out 协议：归一化变体（空白/全半角/大小写/尾标点）+ 单字符 typo（走 fuzzy）+ NIL 负例
（文档结构噪声 + 随机串）。报告 in-KB 链接准确率、fuzzy typo 准确率、NIL 拒绝率与总体准确率。

## 多跳 GraphRAG 问答

### 检索/溯源核心自洽性检查（离线、无 API）

```powershell
python benchmarks/eval_kg_qa.py
```

从图谱确定性派生 1/2/3 跳问题与 gold 答案，经 QAAgent 的同款检索/溯源/安全打分核心（离线、无 API）
作答，度量多跳答案覆盖、逐边溯源率与图外问题安全拒答率。这是可复现的"检索-溯源接地"口径，
**LLM 表述层不参与、不计入该分数**——问题与作答共用同一套图遍历原语，因此更准确的定位是
"验证检索原语本身没有实现 bug"，而不是对回答质量的独立测量。

### 真实回答质量（Ragas，独立标注集，含 LLM 表述层）

```powershell
python benchmarks/build_kg_qa_human.py   # 生成/更新问答集（从 graph_scaled.json 抽取并核验，不需要 API）
python benchmarks/eval_ragas_kg_qa.py    # 真实调用 QAAgent.answer() 作答 + Ragas 判分（需要 API）
```

问答集来源：`benchmarks/kg_qa_human.json`，40 题（22 单跳 + 10 两跳 + 4 三跳 + 6 图外负例），
覆盖 10 类关系（症状/用药/并发症/病因/鉴别诊断/病理分型/高危因素/辅助治疗/检验等）。每条的
`reference`（参考答案）与 `reference_context`（参考证据三元组）在生成脚本运行时**从
`graph_scaled.json` 直接读取并核验**（而非人工编造），且已过滤自环边与超长抽取噪声片段
（详见脚本内 `_is_reasonable_entity` 的注释）。

评测流程：每条问题真实调用 `QAAgent.answer()`（NER 锚点解析 → 子图检索 → LLM 组合生成，
与生产路径完全一致），取其真实返回的回答文本与检索证据，接受四项 Ragas 指标评分：

| 指标 | 衡量什么 |
| --- | --- |
| faithfulness | 回答内容是否被检索到的证据支持，有没有编造未被证据覆盖的说法 |
| answer_relevancy | 回答是否切题，有没有答非所问 |
| context_precision | 检索到的证据里有多少是真正相关的 |
| context_recall | 检索是否覆盖了参考答案所需的关键事实 |

裁判模型：与作答智能体同一个 `LLMClient` 配置（`.env` 的 `LLM_PROVIDER`），避免额外维护一套
裁判模型配置。图外负例（6 题）不参与 Ragas 四项打分（证据为空时打分无意义），单独统计
`safe_rejection_rate`（是否正确拒答），口径与上面的自洽性检查一致但基于真实作答。

诚实边界：Ragas 是 LLM-as-judge，存在裁判模型自身偏差与不稳定性，报告中与确定性检查
并列而非相互替代；报告脚本另附一个不经过 LLM 的粗粒度交叉检查
`entity_mention_recall`（参考实体是否被回答原文字面提及）——它无法识别正确的同义改写，
因此该数值系统性偏低，仅作为"回答是否完全没有涉及证据"的下限信号，不作为主口径。

脚本支持 `--limit-per-hop N --max-negative M` 做分层抽样（按 1/2/3 跳各取前 N 题 + 前 M 题负例），
用于控制 LLM-as-judge 评测的 API 成本；不传参数则跑满 40 题的完整问答集。

#### 一次实测结果（2026-07-31，小样本试点）

裁判/作答模型：`deepseek-ai/DeepSeek-V4-Flash`（成本考虑，非项目默认的 `DeepSeek-V4-Pro`）；
样本：分层抽样 6 正例（1/2/3 跳各 2 题）+ 2 图外负例（完整集为 34+6，命令
`eval_ragas_kg_qa.py --limit-per-hop 2 --max-negative 2`）：

| 指标 | 结果 | 说明 |
| --- | ---: | --- |
| faithfulness | 0.500（4/6 题，2 题裁判调用超时） | |
| answer_relevancy | 0.7383（6/6） | |
| context_precision | 0.1046（6/6） | 见下方根因说明，非随机噪声 |
| context_recall | 0.2111（6/6） | 同上 |
| entity_mention_recall（非 LLM） | 0.4861 | |
| safe_rejection_rate | 0.5（2 题） | 见下方标注口径说明 |

**根因诊断（非随机噪声，已定位）**：链式多跳问题（如"糖尿病的相关之一是脑疝，脑疝的常见
临床表现有哪些？"）中，`QAAgent` 的子图检索把问题链路上*中间锚点实体*（如"糖尿病"）的
全部一跳邻居（20+ 条不相关边）也带入了证据列表，而非只保留真正相关的目标实体（"脑疝"）
的边。这直接拉低 context_precision（证据里大部分不相关）与 context_recall（挤占了检索
预算，实际相关证据的覆盖反而不完整）。这是检索粒度问题，而非答案内容错误——人工核对
显示各条回答本身多数忠实于其检索到的证据，只是检索到的证据本身噪声过多。留作后续可优化
方向，不在本轮改造范围内修复。

**标注口径说明**：2 条负例中，"幻想性紫罗兰综合征"被正确标记为 `refused=True`；
"第九型星际眩晕症"一题的回答文本已实质拒答（"没有找到...无法回答"），但 `refused` 布尔
标记未置位——即 `safe_rejection_rate` 的分母统计口径对这类"文本层面已拒答但标记缺失"
的情况会漏记，实际拒答表现好于该数字显示的 0.5。

此为**小样本试点**（8/40 题），用于验证 Ragas 管线端到端可用且给出真实（非零、非编造）
的裁判评分；不作为项目结果质量的最终定论。完整 40 题评测命令、以及用项目默认
`DeepSeek-V4-Pro` 作裁判的完整评测，可用 `python benchmarks/eval_ragas_kg_qa.py`
（不传参数）复现，成本与耗时随样本量线性增长。

## 置信度标定

```powershell
python benchmarks/calibrate_fast_extractor.py --samples 1200
```

在 CMeIE dev 的生产输出策略上拟合带 bias 的二元温度标定，报告预测级 ECE、
NLL 和可靠性分箱。漏检实体没有预测置信度，不计入 ECE；报告中明确这一口径。

## NL2SQL

人工小集：

```powershell
python benchmarks/eval_nl2sql.py
```

模板生成的 128 题组合压力集（单独报告）：

```powershell
python benchmarks/build_nl2sql_stress.py
python benchmarks/eval_nl2sql.py --gold benchmarks/nl2sql_stress_128.json
```

评测用多重集比较而不是普通 set，避免重复行被吞掉。每条预测同时在 seed=42
主库和 seed=137 副库执行；`dual_database_execution_accuracy` 是比单库执行匹配
更严格的逻辑等价代理，但不声称等同于形式化 SQL 等价证明。

## NL2SQL 透明度（生成路径分列）

`eval_nl2sql.py` 现输出 `generation_mode_breakdown`，分列**确定性模板路由**与 **LLM** 各自的题数与准确率，避免把"100%"误读为纯 LLM 能力：

- 人工 16 题 / 模板压力集 128 题：**100%**，全部由确定性模板路由命中（`llm_calls=0`）；
- **非模板自然问句集**（`benchmarks/nl2sql_hard_natural.json`，**44 题**，刻意偏离模板）：整体 **100%（44/44）**，双库 **100%**；确定性路由 9/9、LLM 35/35。

### 从 14 题到 44 题：修复过程（诚实记录，非事后美化）

扩题前先发现一个更深的问题：分析库 `derive_vocab()` 曾把 CM3KG 全量导入的 ~1,200
种疾病不加筛选地用于合成 600 条就诊记录，稀释到大多数疾病样本量为 0——意味着旧
14 题里至少 3 题命中的是"gold 与预测双方都查不到数据、空集等于空集"的**巧合匹配**，
不是真实校验。修复：`derive_vocab(max_diseases=50)` 按药物/检查条目数精选词表 +
强制锚点常见疾病（详见 `medigraph/analysis/relational.py`），使字面值过滤类问题
都对应真实非空数据。

在真实、非空数据下重新评测，44 题首轮实测 70.5%（31/44），逐条排查后确认失败集中在
**确定性路由的真实缺陷**（8/13 为路由器误判，而非 LLM 能力不足）：路由器悄悄丢弃了
BETWEEN 年龄区间、"除了...以外"排除、HAVING 数量阈值、月份半年范围等限定条件，
还有一处 Unicode `.lower()` 把中文药名里的罗马数字后缀 Ⅰ 错误映射成 ⅰ（影响真实
生产路径，不只是评测）。逐条修复（`_router_should_defer` 新增复杂度门控、
`_deterministic_sql` 补齐 BETWEEN/特定检验项/超降序方向等能力、`schema_text()`
补充 kg_triples.relation 是英文代码而非中文的提示）后复测两次均为 **44/44=100%**。
另发现执行准确率比较对"单一实体+计数"类问句的列数选择存在真实的 API 层非确定性
（同一问题、temperature=0，不同调用返回 1 列或 2 列），修复方式是让比较逻辑容忍
预测多出的**尾部**列（`_rows_match`，仅当预测列数≥gold 时截断比较，缺列仍判错，
不掩盖真实的行基数错误如 JOIN 扇出）。

证据：`outputs/eval_nl2sql_nl2sql_hard_natural.json`；回归测试见
`tests/test_nl2sql_router.py`（含每个路由器缺陷的最小复现）。

### 留出集复测：三个 100% 是开发集数字，不是留出测试集数字

上一小节记录的修复过程有一个必须写明的副作用：**44 题集合既用于报告分数，也用于定位
缺陷**——`_router_should_defer` 的复杂度门控是逐条对着它的失败样本补出来的。同理，128
题压力集由模板生成、又被模板路由命中。因此这三个 100% 的正确读法是**开发集上的收敛
结果**，而不是"在未见过的问句上也有 100%"。

为量化这个差距，另写了一组**从未参与任何调优**的留出问句（同 schema、同数据库、人工
写 gold SQL，与三个评测集无重叠），只跑确定性路由（离线、零 API 成本）。首轮 **22/24
= 91.7%**，暴露两个此前 188 道题全部漏掉的**静默错答**缺陷：

1. `各科室的平均就诊年龄是多少` —— "平均就诊年龄"因中间插了"就诊"二字，未命中
   `平均年龄` 分支，继续下落到计数分支（只要句中有"多少"即命中），**返回按就诊量排序的
   `COUNT(*)`**，答的是另一个指标且不报错。44 题集里唯一一道平均年龄题恰好在后半句
   写出了"平均年龄"字样，所以一直没被发现。
2. `费用最高的3次就诊是哪些患者` —— `explicit_top_n` 只认 个/种，不认 次/位，静默回落
   到默认 5，**多返回两行**。三个评测集的排序题一律用"N 个科室 / N 种药物 / N 种疾病"，
   量词"次"从未被覆盖。

两处均已修复（`nl2sql.py` 平均年龄分支扩展措辞、`schema_linking.explicit_top_n` 扩充量词
并排除"超过 N 次"这类阈值表述），修复后留出集 **25/25**，全量离线测试由 309 增至 **315**
（新增 6 项针对上述两个缺陷的回归测试）。

**结论**：确定性路由在同分布留出问句上的真实水平约为**九成出头，修复后接近全对**；三个
评测集的 100% 应理解为路由覆盖面已收敛，而非泛化误差为零。路由无法表达的问句一律交给
LLM（`_router_should_defer`），因此**未覆盖的表述会退化为 LLM 路径，不会静默答错**——上面
两个缺陷正属于"本该 defer 却答了"的情形，也是这类系统最需要防的失败模式。

## 实体链接（两个互补口径，避免自证）

链接 KB 已从"仅 CM3KG"扩为**CM3KG + CMeIE-V2 train + DIAKG 的医疗术语库**（`data/models/entity_linker.json`，**39,369 条** = CM3KG 7,464 + CMeIE-V2 train 29,282 + DIAKG 2,623），CM3KG 条目在冲突时保留优先级。CM3KG 侧的 7,464 少于图导入基线的 7,467 个节点，差额是规范化后 canonical_key 为空或与已有条目重合的 3 条，未计入 KB。

1. `eval_entity_linking.py`：对 KB 条目做归一化/typo/NIL 扰动 → **in-KB 链接准确率 0.973、NIL 拒绝 1.0**（术语在库时链得准不准）。
2. `eval_entity_linking_real.py`：把**神经 NER 在 held-out CMeIE-V2 dev 文本上预测的真实 mention** 链回 KB → **可链接率 0.733**（1259 exact + 55 fuzzy / 1793，KB 扩容前为 0.161）。

诚实说明：KB 现包含 CMeIE/DIAKG 术语库，故对该领域真实 mention 覆盖高（73%）且 NIL 拒绝仍 1.0；剩余 27% 为新词/罕见表达，链到 UMLS/ICD 等外部标准编码是进一步方向。证据：`outputs/eval_entity_linking.json`、`outputs/eval_entity_linking_real.json`。

> 产物字段勘误：`outputs/eval_entity_linking_real.json` 的 `protocol` 字段仍写着
> "linked to CM3KG"，这是 KB 扩容前的旧标签。该文件里的 **0.7328 是扩容后 KB 的实测值**
> （扩容前为 0.161，两个数在本节都已给出）。脚本中的标签已更正，产物保持原始落盘状态
> 未手工改动；重跑 `python benchmarks/eval_entity_linking_real.py` 即可得到新标签。

## 官方 test 预测（去 dev 星号）

`scripts/predict_cmeie_test.py` 生成官方格式 `outputs/CMeIE_test_pred.jsonl`（4482 行），可提交 Tianchi CMeIE 榜单获取**官方 test SPO-F1**，与公开方法完全同集同口径对比。

## CMeIE 公开基准对照

本系统在 CMeIE-V1（CBLUE 原始版本，公开 GPLinker/CasRel 数字所基于的数据集）上报告 held-out dev 严格 SPO-F1：

| 系统 | SPO-F1 | 评测集 | 说明 |
| --- | ---: | --- | --- |
| 本系统 GPLinker（macbert-large） | 0.6146 | dev / strict | 权重、日志和评测 JSON 随仓 |
| 公开 GPLinker | 0.5982 | test | Tianchi 在线评测 |
| 公开 CasRel | 0.6056 | test | 同一任务经典强基线 |
| 榜单 Top-1（百度 ERNIE 知识图谱团队） | 0.6604 | test | 重度工程 + 大模型 |

口径说明：

1. 本系统报告 dev，公开方法通常报告 test；由于 CMeIE-V1 官方 test 标签未公开，文档中不宣称击败闭源 test SOTA；
2. 严格 SPO-F1 使用 `(subject, predicate, object)` 原始字符串精确匹配，不用宽松匹配抬高 headline；
3. 本系统可信主张是：在同一数据族、同一严格 SPO 口径、完全开放权重与日志的前提下，达到并超过公开单模型强基线。

复现命令：

```powershell
python scripts/download_encoder.py --repo hfl/chinese-macbert-large
python scripts/train_neural_extractor.py --encoder data/models/encoder/chinese-macbert-large `
  --data_dir ../CMeIE-V1 --file_tpl "CMeIE_{split}.jsonl" `
  --out data/models/neural_extractor_v1 --epochs 10 --batch_size 6 --max_length 224
python benchmarks/eval_neural_extraction.py --split dev `
  --data_dir ../CMeIE-V1 --file_tpl "CMeIE_{split}.jsonl" `
  --model_dir data/models/neural_extractor_v1 --tag v1
```

证据文件：`outputs/eval_neural_cmeie_v1_dev.json`、`outputs/eval_ensemble_cmeie_v1_dev.json`、`outputs/CMeIE_test_pred.jsonl`。

## 测试

```powershell
python -m pytest -q
```

覆盖抽取、校准、实体链接、七类文档格式、PII、数据质量、DAG 恢复、图谱增量/
路径/审计、安全拒答、NL2SQL 安全与七类图表。

## 当前落盘实测

本文件只定义口径；各指标的当前实测值统一汇总在 `EVIDENCE_MAP.md`
的"实测硬指标汇总"一节，所有数值以对应 `outputs/*.json` 为准。
