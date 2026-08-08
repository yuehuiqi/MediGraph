# 原创性声明（ORIGINALITY）

> 本文件说明本项目核心模块的原创设计点、其独立完成的过程依据，以及与
> `INSPIRATION_REGISTER.md` 中登记的公开借鉴内容之间的边界。

## 一、原创性判定原则

一个模块被列为"原创设计"，须满足：

1. 其具体实现（数据结构、算法组合方式、代码）由团队独立编写，不复制任何其他
   参赛项目或未经许可的第三方代码；
2. 若借鉴了公开研究方法或标准，仅借鉴其**问题定义/通用范式**，具体工程实现、
   参数选择、与本项目其他模块的组合方式由团队独立完成（详见
   `INSPIRATION_REGISTER.md` 的逐条边界）；
3. 面向本赛题的具体数据、Schema、评测协议、安全护栏均为本项目量身设计。

## 二、核心原创设计点

### 1. 统一算子协议（Operator Protocol）

`medigraph/operators/base.py` 定义的 `OperatorMeta + Registry + run(dict)->dict`
契约，使同一算子实现无需修改即可：本地直接调用、打包为 DataMate 市场算子
（`datamate_ops/`）、暴露为 Nexent MCP 工具（`mcp_server/server.py`）、被 A2A
Agent 编排（`integration/a2a/`）、以及本次新增的 HTTP 服务层（`service/`）复用。
这一"协议层吸收平台差异、算法实现作为唯一事实源"的设计不借鉴任何外部项目。

### 2. 置信度路由抽取级联

`medigraph/extraction/cascade.py` 与 `neural_gplinker.py` 组织的
"神经 GPLinker（主力）→ 词典 Trie（零依赖下界）→ LLM（难例修复）→ 实体链接/
Schema 校验（治理）"四级级联，路由阈值、双编码器融合权重、词典兜底触发条件均为
团队根据 CMeIE-V2 实测分布独立调参确定，不采纳任何已公开的路由策略实现。

### 3. DAG 执行器：分层并行 + 静态校验 + 节点超时

- `medigraph/agents/dag_executor.py` 的 `topological_layers` 分层调度、
  "层内快照读取、层末统一合并"的并发写入隔离策略、节点级超时（通过一次性
  `ThreadPoolExecutor` 实现可放弃等待）均为团队设计，仅算法骨架（拓扑排序）
  借鉴 Kahn 算法这一图论通用方法。
- `medigraph/agents/plan_validator.py` 的执行前静态校验（沿拓扑序传播可用
  payload key、比对 `input_schema.required`、一次性报全部错误）为团队原创设计，
  解决"LLM 生成的计划跑到一半才发现缺输入"的具体问题。

### 4. SQL 安全护栏：AST 结构判定 + 双引擎纵深防御

`medigraph/analysis/sql_guard.py` 用 sqlglot 解析 SQL 为 AST 后做结构级白名单
判定（而非关键词黑名单），`medigraph/analysis/pg_relational.py` 额外用连接级
`read_only` 属性 + 服务端语句超时做第二层防御。护栏设计（判定规则、绕过测试
用例）为团队根据实际攻防场景独立设计。

### 5. 中文数字归一化：双层门控

`medigraph/analysis/numerals.py` 的量词门控（仅在数字后紧跟计量单位或前接
排序前缀时才改写）与词表掩码（保护含数字的医学术语不被误改写）为团队根据
本项目分析库 7,465 条真实词表值反向验证后设计的具体方案，解决"朴素中文数字
转换会破坏'血液生化六项检查''二十五味松石丸'等医学术语"的具体问题。

### 6. 分析库词表精选（`derive_vocab` 疾病筛选）

`medigraph/analysis/relational.py` 的 `derive_vocab(max_diseases, anchors)`
解决"图谱疾病词表过大导致合成就诊记录按疾病稀释、字面值过滤类 NL2SQL 问题
命中空集巧合匹配"这一具体数据完整性问题。筛选规则（按药物/检查条目数排序
+ 强制锚点疾病）为团队诊断问题根因后独立设计。

### 7. SSE 流式：重放式续传

`service/stream.py` 的流式续传语义（LLM 生成不可确定性重放，因此"续传"实现为
基于 `stream_id` 缓冲区的**已产出内容重放**而非重新生成）为团队根据 LLM 场景
的具体约束设计；缓冲区上限、LRU 淘汰、`replay_truncated` 标注均为团队实现。

### 8. Ascend C 融合算子

`NPU/NPU_source/ascendc/medical_fused_softmax_kernel.cpp`（温度校准 + 医疗
标签约束 + Softmax 融合为单一设备 Kernel）为团队手写 Ascend C 实现，不借鉴
任何公开算子实现；对照实验设计（同卡 torch_npu 基线、多轮统计、Profiler、
实测功耗）为团队独立设计的评测协议。

## 三、问题发现过程的诚实说明

本项目 2026-07 至 2026-07-31 期间的多项修复（中文数字 bug、分析库词表稀疏导致
的巧合匹配、NL2SQL 路由器 Unicode 大小写 bug、README/EVIDENCE_MAP 数字漂移等）
均由团队对**自身代码库**的独立测试与审查发现——具体方法是直接查询数据库真实
内容、对比金标准与实际 SQL 执行结果、反向验证边界条件——而非来自外部报告的
提示。每项修复对应的诊断过程、复现步骤与验证测试均保留在代码提交与测试文件中，
可逐项核对。

## 四、维护说明

新增核心设计点时，在本文件补充一条记录，说明其原创部分、依赖的公开方法（若有，
须同步登记到 `INSPIRATION_REGISTER.md`）及验证方式。
