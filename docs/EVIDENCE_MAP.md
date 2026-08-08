# 评审证据导航（EVIDENCE_MAP）

> 本文件位于 `docs/`，作为评审入口集中管理，不参与运行时导入。
>
> **给评审的一页纸**：本表把赛题《ModelEngine 开源项目贡献赛》**每一个评分项**，逐条映射到本仓的**证据文件**、**已落盘的实测数字**和**一条可复现命令**。所有数字均取自仓内 `outputs/*.json`、`finetune/outputs/*.json`、`NPU/NPU_results/*`，可当场打开核对。
>
> - 运行目录：仓库根目录（下称 `.`）。
> - 环境：`python -m pip install -r requirements.txt`；离线 L1/测试/确定性 NL2SQL/图谱**不需要 API Key**。
> - **一键总复现**：`python scripts/reproduce_offline.py --quick` → 生成 `outputs/reproduction_manifest.json`（命令、耗时、环境、产物 SHA-256）。
> - 口径与协议：`docs/EVALUATION_PROTOCOL.md`；数据/模型边界：`docs/MODEL_AND_DATA_CARDS.md`。

---

## 任务一：数据处理智能体（30 分）

| 评分项(分)         | 证据文件                                                                                                                                                  | 关键实测 / 交付                                                                                                                            | 一条复现命令                                                                                                     |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| 架构设计合理性 (5) | `medigraph/agents/dataproc_agent.py`、`medigraph/agents/dag_executor.py`、`medigraph/operators/`（9 个统一算子）                                    | Agent 与算子解耦；DAG 执行器含参数化重试 / fallback / 失败依赖跳过 / 产物自反思                                                            | `python demos/demo_task1_dataproc.py --input data/corpus --max-docs 1 --goal "读取清洗质检脱敏抽取规范化校验"` |
| 智能体能力 (10)    | `finetune/`、`finetune/outputs/eval_orchestrator.json`、`finetune/outputs/qwen3p5-0p8b-orchestrator/adapter_model.safetensors`                      | **选做已落地**：0.8B LoRA 编排器 DAG 准确率 **0.973** / 可执行率 **1.000**（vs base 0.161、大 API 0.741；eval_size=112） | `python finetune/eval_orchestrator.py`                                                                         |
| 功能完整性 (10)    | `outputs/task1_processed.jsonl`、`outputs/task1_report.json`、`integration/datamate/datamate_client.py`、`outputs/datamate_pipeline_summary.json` | 多格式接入→清洗→质检→PII→抽取→链接→校验全流程可跑通并落盘；真实 DataMate REST 客户端                                                 | `python demos/demo_task1_dataproc.py --input data/corpus --max-docs 3`                                         |
| 工程质量 (5)       | `tests/`（14 文件）、`scripts/reproduce_offline.py`、`outputs/reproduction_manifest.json`                                                  | 315 项离线测试通过；一键复现 + 逐产物 SHA-256                                                                                               | `python -m pytest -q`                                                                                          |

---

## 任务二：知识图谱问答智能体（40 分 + NPU 加分 10 分）

| 评分项(分)                   | 证据文件                                                                                                                                                                             | 关键实测 / 交付                                                                                                                                                                                                                                                 | 一条复现命令                                                                              |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| 知识图谱设计 (10)            | `outputs/kg_scale_report.json`、`outputs/graph_scaled.json`、`medigraph/schema/`（cmeie_schema / ontology / relation_mapping）                                                 | 自产图谱**43,566 实体 / 70,790 三元组**（`self_produced:true`，25,492 篇文本抽取，逐边溯源）；导入基线 7,467/26,784 严格分列                                                                                                                            | `python scripts/build_scaled_kg.py`                                                     |
| 算子设计与实现 (10)          | `medigraph/extraction/neural_gplinker.py`、`medigraph/extraction/cascade.py`、`outputs/eval_neural_cmeie_dev.json`、`outputs/eval_fast_cmeie_dev.json`                       | GPLinker 联合抽取（单前向出实体+三元组，无 NER→RE 误差传播）；延迟**P50 22.95 ms**（神经）/ **0.12 ms**（词典对照，CPU）——两者非同设备实测，口径见 `docs/MODEL_AND_DATA_CARDS.md` §2.3                                                                                                                                | `python benchmarks/eval_neural_extraction.py --split dev`                               |
| 智能体编排能力 (10)          | `medigraph/agents/kg_agent.py`、`medigraph/agents/qa_agent.py`、`medigraph/extraction/entity_linker.py`、`outputs/eval_entity_linking.json`                                  | 级联自动编排（L1 神经 / L1' 词典 / L2 LLM / L3 链接治理）；实体链接 in-KB**0.9727**、NIL 拒绝 **1.000**（KB 39,369 条）                                                                                                                             | `python demos/demo_task2_complete.py`                                                   |
| 结果质量 (10)                | `outputs/eval_neural_cmeie_dev.json`、`outputs/eval_neural_cmeie_v1_dev.json`、`outputs/eval_kg_qa.json`、`outputs/eval_ragas_kg_qa.json`、`outputs/calibration_report.json`、`docs/EVALUATION_PROTOCOL.md` | 实体 F1**0.7664** / 端到端三元组 F1 **0.5343**（held-out dev，53 关系无 gold 实体）；CMeIE-V1 严格 SPO-F1 **0.6146**；检索/溯源/安全拒答核心 1/2/3 跳 **1.000**、溯源 **1.000**、安全拒答 **1.000**（120 题，图谱自遍历自洽性检查，见下方口径说明）；真实回答质量另见 Ragas（faithfulness/answer_relevancy/context_precision/context_recall）；标定后 ECE **0.033** | `python benchmarks/eval_kg_qa.py`、`python benchmarks/eval_ragas_kg_qa.py`             |
| **NPU 算子优化 (+10)** | `NPU/NPU_source/ascendc/medical_fused_softmax_kernel.cpp`、`NPU/NPU_results/summary.json`、`NPU/ModelEngine开源项目贡献赛_NPU技术报告.md`                                      | **手写 Ascend C 融合算子**（温度校准+医疗标签约束+Softmax 融合单核）；纯核吞吐 **2.71×**、融合 **3.12×**、整卡能效 **3.00×**；含 profiler CSV、5 轮重复、能效采样、`source.sha256`、Validation PASS                                | `cd NPU/NPU_source && ./run_all.sh quick`（需 910B；结果已随仓于 `NPU/NPU_results/`） |

> 诚实边界：0.80 三元组 / 0.85 实体为**超公开 SOTA 的目标、未达**；0.534/0.543 已是"53 关系+无 gold 实体"困难口径下的中上水平、约词典基线 5×。全部数字实测未注水，口径定义见 `docs/EVALUATION_PROTOCOL.md`。

---

## 任务三：数据分析智能体与可视化（30 分）

| 评分项(分)              | 证据文件                                                                                                                                                                                                                 | 关键实测 / 交付                                                                                                                                             | 一条复现命令                                                                                                            |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 有效复用 (5)            | `medigraph/analysis/graph_profile.py`、`medigraph/analysis/schema_linking.py`、`outputs/task3_complete_report.json`                                                                                                | 直接接入任务二`graph_scaled.json` 与任务一处理产物，链路闭环                                                                                              | `python demos/demo_task3_complete.py`                                                                                 |
| 智能体能力 (10)         | `medigraph/analysis/nl2sql.py`、`outputs/eval_nl2sql.json`、`outputs/eval_nl2sql_nl2sql_hard_natural.json`、`outputs/eval_nl2sql_nl2sql_stress_128.json`、`benchmarks/nl2sql_*.json`                           | NL2SQL 执行准确率：人工**16/16=100%**（双库）、128 题压力集 **100%**（双库）、非模板 hard-natural **44/44=100%**（双库；从 14 题扩至 44 题，修复词表稀疏导致的巧合匹配与多处路由器缺陷，详见 `docs/EVALUATION_PROTOCOL.md`）；gold 带 SHA-256 | `python benchmarks/eval_nl2sql.py`  ·  `python benchmarks/eval_nl2sql.py --gold benchmarks/nl2sql_hard_natural.json` |
| 可视化与图谱相结合 (10) | `medigraph/analysis/viz.py`、`outputs/task3_complete_bar.html`、`outputs/task3_complete_line.html`、`outputs/task3_complete_pie.html`、`outputs/task3_complete_graph.html`、`outputs/task3_closed_loop.html` | 7 类 BI/图谱图表 + 点击钻取 + 文字洞察，产物 HTML 已落盘                                                                                                    | `python demos/demo_task3_analysis.py`                                                                                 |
| 工程实现 (5)            | `tests/test_task3_analysis.py`、`medigraph/analysis/relational.py`（只读/超时/行数保护）                                                                                                                             | 安全执行护栏 + 双库等价代理评测；测试覆盖                                                                                                                   | `python -m pytest tests/test_task3_analysis.py -q`                                                                    |

---

## 平台集成与提交合规（对应鼓励方向 / 评审规则）

| 关注点              | 证据文件                                                                                                                           | 一条复现 / 核验命令                             |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| DataMate 集成       | `integration/datamate/datamate_client.py`、`integration/datamate/upload_operators.py`、`datamate_ops/dist/*.zip`（5 个算子） | `python datamate_ops/build_zip.py`            |
| Nexent / MCP 集成   | `mcp_server/server.py`（743 行）、`integration/nexent_agents/*.json`（4 套 AgentConfig）                                       | `powershell -File scripts/start_mcp.ps1`      |
| A2A 集成            | `integration/a2a/a2a_server.py`（Agent Card）                                                                                    | `powershell -File scripts/start_a2a.ps1`      |
| 开源合规            | `LICENSE`(MIT)、`NOTICE`、`THIRD_PARTY_NOTICES.md`、`SECURITY.md`                                                          | —                                              |
| 提交完整性 / 可追溯 | `scripts/check_release.py`、`data/external_manifest.json`、`docs/MODEL_AND_DATA_CARDS.md`、`THIRD_PARTY_NOTICES.md`        | `python scripts/check_release.py`             |
| 总复现清单          | `outputs/reproduction_manifest.json`（命令 / 耗时 / 环境 / 产物 SHA-256）                                                        | `python scripts/reproduce_offline.py --quick` |

---

## 本作品实测硬指标汇总

| 指标                                                            |                                结果 | 证据文件                                    |
| --------------------------------------------------------------- | ----------------------------------: | ------------------------------------------- |
| 离线测试                                                        |                          315 passed | `tests/`                                  |
| L1 神经抽取 · 实体 F1 (CMeIE-V2 dev, held-out)                 |                    **0.7664** | `outputs/eval_neural_cmeie_dev.json`      |
| L1 神经抽取 · 端到端三元组 F1                                  |     **0.5343**（集成 0.5431） | `outputs/eval_neural_cmeie_dev.json`      |
| CMeIE-V1 · 严格 SPO-F1（> 公开 GPLinker 0.598 / CasRel 0.606） |                    **0.6146** | `outputs/eval_neural_cmeie_v1_dev.json`   |
| 实体链接 in-KB 准确率 / NIL 拒绝                                |            **0.9727 / 1.000** | `outputs/eval_entity_linking.json`        |
| 多跳 GraphRAG（120 题）检索/溯源/安全拒答核心 *                 |     **1.000 / 1.000 / 1.000** | `outputs/eval_kg_qa.json`                 |
| GraphRAG 真实回答质量（Ragas，独立标注集，含 LLM 表述层）*       |               见下方口径说明 | `outputs/eval_ragas_kg_qa.json`           |
| 自产知识图谱规模                                                | **43,566 实体 / 70,790 关系** | `outputs/kg_scale_report.json`            |
| 标定后预测级 ECE                                                |     **0.033**（标定前 0.535） | `outputs/calibration_report.json`         |
| NL2SQL：人工 16 / 压力集 128 / 非模板 44 题 <sup>†</sup>         |       **100% / 100% / 100%** | `outputs/eval_nl2sql*.json`               |
| 0.8B LoRA 编排 DAG 准确率 / 可执行率                            |             **0.973 / 1.000** | `finetune/outputs/eval_orchestrator.json` |
| NPU 融合算子：吞吐 / 能效                                       |           **3.12× / 3.00×** | `NPU/NPU_results/summary.json`            |

> 说明：所有 `outputs/*.json` 均为实测落盘；数据/模型边界与许可见 `docs/MODEL_AND_DATA_CARDS.md`、`NOTICE`。表中数字与 `README.md`「实测结果」一节、各技术报告一致，可交叉核对。本表为全库指标的唯一权威汇总。
>
> \* **口径说明（诚实降级标注）**：`eval_kg_qa.py` 的问题由图谱自身遍历（`neighbors`/
> `traverse_paths`）确定性生成，又用同一套遍历核心作答——即 gold 答案按构造必然是检索
> 结果的子集，这是**图谱遍历自洽性检查**，而非独立标注集上的测量，也**不经过**
> `QAAgent.answer()` 实际使用的 LLM 表述层。图外负例同理：虚构实体在图中不存在，
> `neighbors()` 返回空自然触发拒答，同样是构造性的。该行保留是因为它仍能验证检索
> 原语本身的正确性（1/2/3 跳邻居/路径遍历没有实现 bug），但不应作为"回答质量"的
> 独立证据。独立测量见下一行的 Ragas 结果：问题与参考答案/参考证据从
> `graph_scaled.json` 抽取并在生成时逐条核验（`benchmarks/build_kg_qa_human.py`），
> 由**真实调用** `QAAgent.answer()`（含 LLM 组合生成）产出的回答接受
> faithfulness / answer_relevancy / context_precision / context_recall 四项评分，
> 复现命令 `python benchmarks/eval_ragas_kg_qa.py`。小样本试点结果（8/40 题）、
> context_precision/recall 偏低的根因诊断（链式多跳检索带入中间实体的无关邻居）
> 见 `docs/EVALUATION_PROTOCOL.md`「一次实测结果」小节。
>
> † **NL2SQL 三个 100% 的口径（必须一起读，否则会被高估）**：128 题压力集由
> `benchmarks/build_nl2sql_stress.py` **按模板生成**，且 128/128 **全部由确定性模板路由
> 命中**（`llm_calls=0`）——问题与答案出自同一套模板，度量的是路由覆盖面与可复现性，
> **不是 LLM 的 Text-to-SQL 能力**；人工 16 题同样 16/16 走模板路由。真正有区分度的是
> **44 题非模板自然问句集**（子查询、跨表 JOIN、DISTINCT 计数、第 N 名、排除条件），
> 其中 **35 题由 LLM 生成、35/35 正确**。三个集合在定稿前都被用于定位并修复路由器缺陷
> （见 `docs/EVALUATION_PROTOCOL.md`「从 14 题到 44 题」），因此更接近**开发集**而非留出
> 测试集；补充的留出复测见同文件「留出集复测」小节。

---

## 端到端验收案例

2026-07-01 使用三份合成、非患者心血管代谢文本完成端到端验收，覆盖 DataMate 五算子、神经 GPLinker 建图、GraphRAG 问答、图谱驱动 NL2SQL 与网页展示。

| 环节        | 结果                                                                               | 证据                                                                                                   |
| ----------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| DataMate    | 任务`97bcd640-4b0a-40fc-8521-8b554c8aba0d`，3/3 成功，状态 `COMPLETED`         | `outputs/datamate_pipeline_summary.json`、`outputs/datamate_new_case_result/all_dest_files/`       |
| Nexent 建图 | 29 个实体、37 条有效三元组，校验通过率 100%                                        | `outputs/new_case_cardiometabolic_20260701.json`、`outputs/new_case_cardiometabolic_20260701.html` |
| 问答        | 证据三元组、confidence 与 source 已落盘                                            | `outputs/nexent_new_case_answer_qa.md`                                                               |
| 分析        | NL2SQL 人工集与压力集双库执行准确率 100%，会话含实际 SQL、数据血缘和 HTML 报告路径 | `integration/nexent/run_new_case.py`        |

复现命令：

```powershell
python integration/datamate/run_pipeline.py --input data/demo_cases/cardiometabolic_20260701 --max-files 3
python scripts/run_new_case.py
python integration/nexent/run_new_case.py
python integration/nexent/create_recording_conversation.py
```
