# Qwen3.5-0.8B 医疗数据 DAG 编排模型

本目录实现一个小于 1B 参数的本地编排模型，把自然语言数据处理需求转换为
MediGraph/DataMate 算子 DAG。核心目标是让小模型负责稳定的流程规划，把大模型调用
留给真正需要语义理解的实体、关系抽取环节。

## 1. 核心产物

| 内容 | 路径 |
| --- | --- |
| 合成训练集 / 评测集 | `finetune/data/train.jsonl`、`finetune/data/eval.jsonl` |
| LoRA 适配器 | `finetune/outputs/qwen3p5-0p8b-orchestrator/` |
| 指标证据 | `finetune/outputs/eval_orchestrator.json` |
| 本地推理 | `finetune/infer.py` |
| OpenAI 兼容服务 | `finetune/openai_compatible_server.py` |
| API 调用 Demo | `demos/demo_finetuned_orchestrator_api.py` |

数据规模：1,126 条、19 类规范流程，train 1,014 / eval 112。训练数据包含
`max_chars` 500/800/1000/1500 和 `min_confidence` 0.6～0.9 等参数变化。

## 2. 实测指标

以下三组结果来自同一份 112 条 held-out 评测集：

| 系统 | 参数量 | DAG 准确率 | 可执行率 |
| --- | ---: | ---: | ---: |
| Qwen3.5-0.8B 原始模型 | 0.8B | 16.1% | 83.0% |
| **Qwen3.5-0.8B + LoRA** | **0.8B** | **97.3%** | **100.0%** |
| Qwen3.6-35B-A3B API | 35B（激活约 3B） | 74.1% | 96.4% |

唯一指标口径以 `finetune/outputs/eval_orchestrator.json` 为准。

## 3. 环境

```powershell
conda create -n meditune python=3.11 -y
conda activate meditune
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r finetune/requirements-train.txt
```

本地训练记录使用 RTX 5070 Ti Laptop 12 GB、bf16、LoRA、3 epoch，耗时约 60 分钟。

## 4. 完整复现

在项目根目录执行。

### 4.1 重新生成数据

该步骤会调用配置的大模型 API；只复现现有模型和指标时可跳过。

```powershell
conda activate meditune
python finetune/synth_data.py --per-pattern 60
```

### 4.2 训练 LoRA

基础模型为 `Qwen3.5-0.8B`：

```powershell
python finetune/train_lora.py --epochs 3 --batch 8
```

输出：

```text
finetune/outputs/qwen3p5-0p8b-orchestrator/
```

### 4.3 评测

```powershell
python finetune/eval_orchestrator.py
```

输出：

```text
finetune/outputs/eval_orchestrator.json
```

### 4.4 单条推理

```powershell
python finetune/infer.py `
  --goal "清洗病理文档、切块、抽取实体和关系、最后校验三元组"
```

## 5. OpenAI 兼容服务与 Nexent

### 5.1 启动服务

推荐使用项目统一启动脚本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_finetuned_model.ps1
```

也可以在 `meditune` 环境中直接运行：

```powershell
python finetune/openai_compatible_server.py `
  --host 0.0.0.0 `
  --port 18088 `
  --base <Qwen3.5-0.8B基础模型目录> `
  --adapter finetune/outputs/qwen3p5-0p8b-orchestrator
```

服务信息：

- 模型 ID：`qwen3p5-0p8b-orchestrator`
- 本机 API：`http://localhost:18088/v1/`
- Nexent Docker：`http://host.docker.internal:18088/v1/`

### 5.2 API 验收

```powershell
conda run -n meditune python demos/demo_finetuned_orchestrator_api.py
```

期望返回包含 `text_clean → chunker → medical_ner → medical_re →
triple_validator` 的 JSON DAG。

### 5.3 Nexent 模型配置

在 Nexent 模型管理中添加：

| 配置项 | 值 |
| --- | --- |
| Display Name | `qwen3p5-0p8b-orchestrator-local` |
| Model Name | `qwen3p5-0p8b-orchestrator` |
| Model Type | `llm` |
| Provider / Factory | `OpenAI-API-Compatible` |
| API Base URL | `http://host.docker.internal:18088/v1/` |
| API Key | `EMPTY` |
| Max Tokens | `4096` |

Nexent 自动生成的 `model_id` 与具体部署有关，不作为固定复现参数。

## 6. 接入 MediGraph

```python
from finetune.infer import LocalOrchestrator
from medigraph.agents.dataproc_agent import DataProcAgent

orchestrator = LocalOrchestrator(base_path, adapter_path)
agent = DataProcAgent(local_planner=orchestrator.plan)
```

这样 DAG 规划由本地 0.8B 模型完成，不调用外部规划 API。

## 7. 核心代码

| 文件 | 作用 |
| --- | --- |
| `orchestrator_prompt.py` | 算子目录、19 类流程模式和 DAG 约束 |
| `synth_data.py` | 合成自然语言需求与 DAG 配对 |
| `train_lora.py` | TRL + PEFT LoRA 训练 |
| `infer.py` | 本地模型加载和 DAG 推理 |
| `eval_orchestrator.py` | 三组模型的统一评测 |
| `openai_compatible_server.py` | OpenAI 兼容推理服务 |
| `api_planner.py` | 主系统调用本地服务的适配器 |

