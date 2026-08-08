---

## 问答结果汇总

基于图谱 `new_case_cardiometabolic_20260701.json`（29 实体 / 37 三元组，抽取器 neural_gplinker-1.0.0），以下为两个问题的严格图谱证据回答。

---

### 问题一：心力衰竭有哪些典型症状和检查依据？

#### 检查依据（直接三元组）

| 检查项目 | 关系 | 置信度 | 来源 |
|----------|------|--------|------|
| 胸部X线 | 需做检查 | 0.811 | new_case_cardiometabolic_input.txt |
| 脑钠肽检测 | 需做检查 | 0.71 | new_case_cardiometabolic_input.txt |

#### 典型症状

**图谱中不存在"心力衰竭 → has_symptom → 某症状"的直接三元组。** 但通过 2 跳推理路径（心力衰竭与 2型糖尿病/高血压 共享症状节点），可间接关联以下症状（非直接证据，不可作为确诊依据）：

| 症状 | 共享疾病 | 路径置信度 | 来源 |
|------|----------|-----------|------|
| 乏力 | 2型糖尿病 / 高血压 | 0.877 / 0.826 | new_case_cardiometabolic_input.txt |
| 活动后气促 | 2型糖尿病 / 高血压 | 0.870 / 0.815 | new_case_cardiometabolic_input.txt |
| 多饮 | 2型糖尿病 / 高血压 | 0.866 / 0.810 | new_case_cardiometabolic_input.txt |
| 多尿 | 2型糖尿病 / 高血压 | 0.860 / 0.805 | new_case_cardiometabolic_input.txt |
| 体重下降 | 2型糖尿病 | 0.853 | new_case_cardiometabolic_input.txt |
| 头晕 | 2型糖尿病 | 0.842 | new_case_cardiometabolic_input.txt |
| 夜间阵发性呼吸困难 | 2型糖尿病 | 0.833 | new_case_cardiometabolic_input.txt |
| 心悸 | 2型糖尿病 | 0.810 | new_case_cardiometabolic_input.txt |

> **注意**：以上症状均为多跳推理路径结果，置信度范围 0.81–0.88，图谱中无心力衰竭直接症状三元组，不能作为临床诊断依据。

---

### 问题二：高血压可能并发哪些疾病？

#### 直接并发疾病（1 跳三元组）

| 并发疾病 | 关系 | 置信度 | 来源 |
|----------|------|--------|------|
| 慢性肾病 | 并发 | 0.834 | new_case_cardiometabolic_input.txt |
| 脑卒中 | 并发 | 0.818 | new_case_cardiometabolic_input.txt |
| 糖尿病 | 并发 | 0.745 | new_case_cardiometabolic_input.txt |
| 糖尿病肾病 | 并发 | 0.745 | new_case_cardiometabolic_input.txt |
| 冠心病 | 并发 | 0.641 | new_case_cardiometabolic_input.txt |

> 另有一条"高血压 → 并发 → 高血压"（置信度 0.592），判定为数据噪声，不纳入有效结论。

---

### 图谱能力证据

- **图谱文件**：`new_case_cardiometabolic_20260701.json`
- **实体规模**：29 实体（Disease 9 / Symptom 9 / Drug 5 / Examination 6）
- **三元组规模**：37 条（has_symptom 24 / complication 6 / recommend_drug 5 / need_examination 2）
- **抽取器**：neural_gplinker/neural-gplinker-1.0.0
- **答案置信度**：问题一 high（0.88），问题二 medium（0.76）

---

### 下一步建议

1. **补充心力衰竭症状三元组**：当前图谱缺少心力衰竭的直接症状关联，建议补充相关临床文本后重新执行 `build_medical_kg` 以丰富症状覆盖。
2. **提升低置信度三元组**：冠心病并发关系置信度仅 0.641，可考虑引入更多循证文本或人工校验。
3. **图谱审计**：如需全面了解图谱质量，可调用 `inspect_medical_kg` 对 `new_case_cardiometabolic_20260701.json` 进行只读审计。