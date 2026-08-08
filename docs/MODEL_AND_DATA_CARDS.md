# 数据卡与模型卡

本文件集中说明数据来源、隐私边界、模型产物、训练/推理方式和适用限制。

## 1. 数据卡

### 1.1 用途

数据仅用于医疗实体/关系抽取、实体规范化、知识图谱问答和可复现分析演示，不用于临床诊断、处方或真实患者决策。

### 1.2 来源与角色

| 数据 | 在项目中的用途 | 是否为项目原创数据 |
| --- | --- | --- |
| CMeIE-V2 train | 神经 GPLinker 训练；L1' 词典/关系事实构建；实体链接术语库来源之一。 | 否 |
| CMeIE-V2 dev | 标定、完整 53 schema held-out 评测。 | 否 |
| CMeIE-V1 train/dev | 对照公开 GPLinker/CasRel 等方法的关系抽取基准。 | 否 |
| DIAKG | 糖尿病领域实体/关系、嵌套术语与自产图谱语料。 | 否 |
| CM3KG | 规范实体索引、结构化图谱基线和实体链接知识库来源。 | 否 |
| Pathology probe | 本体覆盖演示，验证病理域 Tumor/Biomarker/Gene/Morphology 等类型。 | 项目整理的探针样本 |
| `analytics.db` | KG 词表 + 固定 seed 合成就诊、处方、检查数据。 | 合成演示数据 |

原始公开数据位于项目上一级目录，`data/prep/` 只生成派生产物。第三方数据不随 MIT 源码被重新授权。

### 1.3 泄漏与评测边界

- 神经 GPLinker 与 L1' 词典只从 train 学习，dev 用于标定与评测；
- CMeIE-V1 官方 test 标签未公开，按惯例报告 held-out dev；与公开 test 数字对照时必须标注 dev/test 差异；
- CM3KG 是外部知识增强来源，报告中标注为 KB 增强，不写成纯闭卷模型；
- `outputs/graph_scaled.json` 为抽取管道自产图谱，`outputs/graph.json` 为 CM3KG 导入演示图谱，两者严格分列；
- 模板生成的 NL2SQL 压力集与人工自然问句集分开报告；
- pathology probe 只证明本体覆盖，不作为泛化质量结论。

### 1.4 隐私与偏差

仓库不应包含真实患者数据。`analytics.db` 为固定 seed 合成数据。若接入真实数据，应先完成合规审批、最小权限控制、PII 脱敏与人工复核，并禁止把原文标识符写入日志或提示词。

公开中文医疗数据中常见疾病、药物和常见关系占比更高，长尾类型、罕见病、非标准表达与跨句关系覆盖有限。CM3KG 中可能存在历史噪声、表格占位符和过时知识，链接结果不是医学权威编码。

## 2. 模型卡：MediGraph 神经联合抽取器（GPLinker）

### 2.1 模型身份

- 产物：`data/models/neural_extractor/`（主力 macbert-large）与 `data/models/neural_extractor_base/`（roberta-base 集成副模型）；
- 架构：GPLinker，基于 GlobalPointer 的联合实体—关系抽取；
- 编码器：主力 `hfl/chinese-macbert-large`，副模型 `hfl/chinese-roberta-wwm-ext`；
- 特点：单次前向联合解码 typed entity span 与 `(主体, 谓词, 客体)` 三元组，避免 NER→RE 级联误差传播；
- 代码：`medigraph/extraction/neural_gplinker.py`、`scripts/train_neural_extractor.py`、`benchmarks/eval_ensemble.py`。

### 2.2 训练与复现

```powershell
python scripts/download_encoder.py --repo hfl/chinese-macbert-large
python scripts/train_neural_extractor.py --encoder data/models/encoder/chinese-macbert-large `
  --out data/models/neural_extractor --epochs 10 --batch_size 6 --max_length 224
python benchmarks/eval_neural_extraction.py --split dev
python benchmarks/eval_ensemble.py --min_conf 0.75
```

训练数据为 CMeIE-V2 train；CMeIE-V2 官方 test 标签未公开，按惯例报告 held-out dev。

### 2.3 当前指标

全部为 CMeIE-V2 held-out dev（3,585 条）实测，四列同集同口径：

| 指标 | roberta-base | macbert-large（主力） | 双模型集成 | 词典基线 L1' |
| --- | ---: | ---: | ---: | ---: |
| 实体 Micro-F1 | 0.7611 | **0.7664** | 0.7601 | 0.3414 |
| 端到端三元组 F1（完整 53 关系） | 0.5261 | **0.5343** | 0.5431 | 0.1115 |

证据文件：`outputs/eval_neural_cmeie_base_dev.json`（roberta-base）、
`outputs/eval_neural_cmeie_dev.json`（macbert-large）、
`outputs/eval_ensemble_cmeie_dev.json`（集成，`triple_micro.conf_union`）、
`outputs/eval_fast_cmeie_dev.json`（词典基线）。各模型目录下的 `train_log.json`
记录训练曲线，其中的 `best_dev_triple_f1` 是**训练期子集口径**，与上表的全量 dev
口径不同，不能直接互相比较。

**延迟为什么不放进上表**：四份产物的单文本 P50 分别是 roberta-base 102.09 ms、
macbert-large 22.95 ms、集成 40.60 ms、词典基线 0.12 ms——**它们不是在同一设备上测的**，
并列会得出"大模型比 base 快 4 倍"这种不成立的结论。神经两栏跑在不同后端（本轮
roberta-base 复测明确落在 CPU，`device` 字段已记录；macbert 那一轮的产物没有记录
device，从量级判断是 GPU）。因此：**F1 两行是同集同口径、可直接比较；延迟必须按产物
各自的 `device` 字段单独读**。`benchmarks/eval_neural_extraction.py` 现已把 `device`
写入产物，后续复测不会再出现无法判定设备的延迟数字。同设备延迟对照见
`docs/PERFORMANCE.md`。

### 2.4 诚实边界

- 完整 53 关系、无 gold 实体的端到端三元组 F1 是困难口径，本仓 0.534/0.543 为实测值，不写成目标值；
- 公开 SOTA 对照需标注 dev/test 差异；
- 实体 F1 受 CMeIE 只标注参与关系实体的特性影响；
- 无 GPU 时系统自动回退 L1' 词典或 L2 LLM；
- 不适用于医学诊断或处方决策。

## 3. 模型卡：MediGraph 本地 L1 快速抽取基线

### 3.1 模型身份

- 产物：`data/models/fast_extractor.json`
- 方法：监督术语 Trie + 关系事实记忆 + schema/cue 解码
- 性质：确定性、非神经、CPU、无下载依赖
- 接口：字符 span 实体与端到端三元组

该产物不是 GLiNER、W2NER 或 OneRel，不使用“神经模型”名称描述它。

### 3.2 构建来源

- CMeIE-V2 train；
- DIAKG 标注文档；
- CM3KG 结构化图谱。

复现命令：

```powershell
python data/prep/build_fast_extractor.py
```

### 3.3 能力与限制

强项是亚毫秒延迟和完全离线复现；限制是训练词表外实体召回弱、关系事实方法对新实体对泛化有限、cue 解码可能产生语义近邻误配。默认 `auto` 模式会把低置信样本交给 L2。

未来 GLiNER/W2NER/OneRel 等 checkpoint 接入时，应复用以下契约：

```text
entity = {name,type,start,end,confidence,extractor,model_version}
triple = {head,head_type,relation,tail,tail_type,confidence,evidence}
```

接入新模型后必须重新执行完整 dev/test、延迟、ECE 与消融评测，并保留当前基线作对照。
