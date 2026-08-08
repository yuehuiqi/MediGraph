# 数据集说明

本目录存放医疗文本处理、知识图谱构建、问答和数据分析所需的可复现实验数据。数据由公开中文医疗数据集与病理教材片段加工得到，并通过脚本生成统一的文本语料、知识图谱和评测金标。

## 数据来源与用途

| 数据来源 | 用途 | 主要产物 |
| --- | --- | --- |
| CM3KG 结构化医学知识库 | 构建主知识图谱；生成本域受控 NER/RE 金标；支撑图谱问答和图谱分析。 | `kg/cm3kg_graph.json`、`gold/cm3kg_gold.json`、`outputs/graph.json` |
| CMeIE-V2 中文医学信息抽取数据 | 提供公开 NER/RE 评测样本；提供原始医疗文本语料。 | `gold/ner_re_gold.json`、`corpus/cmeie_*.txt` |
| DiaKG 糖尿病细粒度标注数据 | 提供垂直病种语料和额外实体关系金标。 | `gold/ner_re_gold.json`、`corpus/diakg_*.txt` |
| 中文病理教材片段 | 增加医疗文本清洗、切块和抽取场景的文本多样性。 | `corpus/pathology_*.txt` |
| 自建病理探针样本 | 覆盖 11 类实体和 14 类关系，验证病理域本体扩展能力。 | `gold/pathology_probe_gold.json`、`kg/ontology_showcase_graph.json` |

## 目录结构

```text
data/
├── corpus/                         # 60 篇原始医疗文本
├── gold/
│   ├── ner_re_gold.json             # CMeIE-V2 + DiaKG 公开基准金标
│   ├── cm3kg_gold.json              # CM3KG 受控金标
│   └── pathology_probe_gold.json    # 病理探针金标
├── kg/
│   ├── cm3kg_graph.json             # 主知识图谱
│   └── ontology_showcase_graph.json # 全本体展示图谱
└── prep/
    ├── ontology_map.py
    └── build_dataset.py
```

当前默认构建结果：

- `data/corpus/`：60 篇 `.txt` 文本；
- `outputs/graph.json`：7467 个实体、26784 条关系；
- `data/gold/ner_re_gold.json`：250 条公开基准样本；
- `data/gold/cm3kg_gold.json`：120 条受控金标样本；
- `data/gold/pathology_probe_gold.json`：16 条病理探针样本。

## 重建数据

在项目根目录执行：

```powershell
python data/prep/build_dataset.py
```

可通过参数调整生成规模：

```powershell
python data/prep/build_dataset.py `
  --max-diseases 1200 `
  --n-gold-cmeie 150 `
  --n-gold-diakg 100 `
  --n-corpus-cmeie 40 `
  --n-corpus-diakg 15 `
  --n-pathology 5
```

## 使用方式

数据处理 Demo：

```powershell
python demos/demo_task1_dataproc.py --input data/corpus --goal "清洗这批医疗文本、切块、抽取实体和关系三元组"
```

知识图谱问答：

```powershell
python demos/demo_task2_qa.py --question "高血压有哪些症状和推荐药物？"
```

图谱驱动分析：

```powershell
python demos/demo_task3_complete.py
```

抽取质量评测：

```powershell
python benchmarks/eval_extraction.py --gold cm3kg --out eval_extraction_cm3kg.json
python benchmarks/eval_extraction.py --gold public --out eval_extraction_public.json
python benchmarks/eval_extraction.py --gold pathology --out eval_extraction_pathology.json
```

完整公开基准评测会调用较多 LLM API；快速验收可使用 `--limit 2` 或 `--limit 30`。

## 评测口径

抽取评测同时给出两类口径：

- `strict`：实体名或三元组规范化后精确匹配；
- `relaxed`：实体类型一致，且实体边界允许包含关系，适合医学信息抽取中长短 span 不完全一致的场景。

CMeIE-V2 和 DiaKG 金标偏“关系驱动”，仅标注参与关系的实体，公开基准 strict 指标通常会低于受控 CM3KG 和病理探针结果。因此报告中同时保留 strict 与 relaxed，便于复现者核对不同数据源上的表现。
