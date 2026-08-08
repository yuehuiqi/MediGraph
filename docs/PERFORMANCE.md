# 服务层性能与调度实测

> 本文只记录 **服务化与存储层改造** 相关的实测数字：HTTP 服务吞吐、SSE 流式感知延迟、
> DAG 分层并行调度、PostgreSQL 索引与连接池、pgvector ANN 检索、Redis 结果缓存。
> 抽取质量、图谱规模、NL2SQL 准确率与 NPU 指标不在本文范围，
> 见 [EVIDENCE_MAP.md](EVIDENCE_MAP.md) 与 [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)。
>
> 所有数字均由仓内脚本现场产出并落盘到 `outputs/bench_*.json`，可直接复核。

## 测试环境

| 项 | 值 |
| --- | --- |
| 平台 | Windows 11，Python 3.11（`.venv-neural`） |
| 服务 | `service/main.py`（uvicorn，单进程） |
| 分析库 | SQLite `outputs/analytics.db` |
| LLM | SiliconFlow `deepseek-ai/DeepSeek-V4-Pro`（外部 API，延迟受网络与服务商影响） |

---

## 1. HTTP 服务吞吐与延迟

复现：

```powershell
$env:CACHE_ENABLED="false"; python service/main.py            # 另开一个终端
python scripts/bench_api_load.py --concurrency 16 --requests 320
```

打到确定性 NL2SQL 路由（模板快路径，**不产生 LLM 调用**），因此测量的是**服务链路本身**：
中间件、指标埋点、请求校验、Schema-Linking、SQL AST 校验、受限 SQLite 执行。

| 指标 | 实测（关缓存基线） |
| --- | ---: |
| 并发 | 16 |
| 请求总数 / 成功 | 320 / 320 |
| 错误率 | **0.0%** |
| 吞吐 | **530.2 req/s** |
| 延迟 p50 | **26.19 ms** |
| 延迟 p90 | 42.21 ms |
| 延迟 p95 | 45.76 ms |
| 延迟 p99 | **49.85 ms** |
| 延迟 max | 52.17 ms |

证据：`outputs/bench_api_load.json`（该文件的 `cache_events.cache_active` 字段
自证本轮缓存未生效，见 §6）

**口径说明**：这条路径不含 LLM 调用。走 LLM 的路由（抽取 L2、图谱问答、非模板 NL2SQL）
其延迟由外部服务商决定，不受本服务链路约束，不应与上表混读。单请求实测参考：

| 路由 | 单请求耗时 | 说明 |
| --- | ---: | --- |
| `POST /api/v1/analysis/nl2sql`（模板命中） | 7.93 ms | 纯本地 |
| `POST /api/v1/extract`（`backend=fast`） | 7.20 s | 词典 NER + LLM 关系抽取 |

---

## 2. SSE 流式：感知延迟

复现：

```powershell
python service/main.py
python scripts/bench_stream_ttft.py --repeats 2
```

同样 6 次问答分别走 `POST /api/v1/kg/qa`（阻塞 JSON）与 `POST /api/v1/kg/qa/stream`（SSE）。

| 指标 | p50 | 含义 |
| --- | ---: | --- |
| 阻塞式首次可见输出 | **10.293 s** | 阻塞式在答案生成完毕前不返回任何内容，所以它的总延迟**就是**首次可见输出时间 |
| 流式首字节（`meta` 帧） | **0.017 s** | 客户端可立即渲染会话与状态 |
| 流式首个答案 token | **8.369 s** | 答案正文开始出现 |

| 对比 | 倍数 |
| --- | ---: |
| 首字节 | **约 605×** 更早 |
| 首个答案 token | **1.23×** 更早（提前 1.92 s） |

证据：`outputs/bench_stream_ttft.json`

**诚实结论**：流式**没有**缩短总生成时间，它改善的是感知延迟，而且改善分两段：

- 首字节从 10.3 s 降到 17 ms，这一段收益是确定且巨大的；
- 首个答案 token 只快 1.23×，因为 **首 token 延迟的主要成本不在生成，而在检索阶段**：
  `QAAgent.answer()` 先对问题做一次**阻塞式 LLM NER** 定位锚点实体，再做图谱多跳遍历，
  之后才进入生成。这段前置开销无法被流式掩盖。

因此"下一步优化目标是检索阶段（把问题 NER 换成本地词典/神经路径，或与遍历并行），
而不是流式本身"。本文不把 1.23× 包装成更大的数字。

**观测到的方差**：单条最慢样本（"冠心病的临床表现有哪些？"）首 token 达 24.9–45.5 s，
慢于同题阻塞式的 19.4–20.9 s。原因是该题证据子图更大、生成更长（约 370 个 delta），
且外部 API 延迟本身抖动明显。样本量仅 6，`bench_stream_ttft.json` 保留逐条明细。

### 协议选型：SSE 而非 WebSocket

| 理由 | 说明 |
| --- | --- |
| 单向即可 | 本场景只有服务端向客户端推 token，客户端不回写，用不到 WebSocket 的双向能力 |
| 纯 HTTP | 无需 Upgrade 握手，可穿过常规反向代理与企业网关 |
| 协议自带续传 | `id:` 帧 + `Last-Event-ID` 请求头是规范内语义，浏览器还自带自动重连；WebSocket 需自行实现心跳与重连 |
| 实现成本 | 显著更低，且可用 `curl` 直接调试 |

配套细节：响应头带 `X-Accel-Buffering: no` 与 `Cache-Control: no-transform`，
否则 nginx 之类的中间层会缓冲帧，流式的早首字节收益会被抹掉。

### 续传语义（重要边界）

LLM 生成不可确定性重放，所以这里的"续传"是**重放已产出内容**，不是重新生成：
每条流按 `stream_id` 缓存 delta，重连时携带 `Last-Event-ID` 与 `X-Stream-ID` 只补发该序号之后的部分。
缓存有上限（64 条流 / 每条 64k 字符，LRU 淘汰），因此这是**断线短窗口恢复，不是持久日志**；
超出预算时 `meta` 帧的 `replay_truncated` 字段会如实标注。

---

## 3. DAG 分层并行调度

复现：

```powershell
$env:EXTRACTION_BACKEND="fast"
python scripts/bench_dag_parallel.py --repeats 3 --simulate-latency 0.5
```

对比两种 DAG 形状，各自跑串行与分层并行：

| 形状 | 节点 | 层数 | 各层宽度 |
| --- | ---: | ---: | --- |
| `linear`（对照组） | 8 | 8 | `[1,1,1,1,1,1,1,1]` |
| `branched` | 8 | 6 | `[1,3,1,1,1,1]` |

`linear` 是经典 ETL 链（清洗→质检→脱敏→切块→NER→链接→RE→校验），每层只有一个节点，
**并行在原理上不可能带来收益**，所以它是验证测量有效性的对照组。
`branched` 把「质检 / PII 脱敏 / 切块」三个互不依赖的算子放在同一层。

以每个算子调用注入 **+0.5 s** 固定延迟测量调度器（理由见下）：

| 形状 | 串行 p50 | 分层并行 p50 | 加速 |
| --- | ---: | ---: | ---: |
| `linear`（对照组） | 4.00 s | 4.00 s | **1.00×** |
| `branched` | 4.00 s | 3.00 s | **1.33×**（省 1.00 s） |

结果与理论完全吻合：8 节点 × 0.5 s = 4.0 s 串行；`branched` 6 层 × 0.5 s = 3.0 s。
对照组稳定在 1.00× 说明测量本身没有引入虚假加速。

证据：`outputs/bench_dag_parallel.json`

**为什么用注入延迟而不是真实端到端计时**：

- 用离线词典后端（`EXTRACTION_BACKEND=fast`）时，整条流水线在**毫秒级**完成
  （实测两种形状串行/并行均为 0.00 s），调度差异根本无法测量；
- 用 LLM 后端时每个节点耗时数秒，但外部服务商的延迟抖动远大于调度收益，
  且每次测量都消耗 API 额度（首次尝试即遭遇连续 `Request timed out`）。

注入固定延迟是在**真实 DAG 形状**上隔离调度器的做法。该数字在报告中标注为
`synthetic_latency_s`，**不得**当作端到端流水线耗时对外引用。真实收益随单节点延迟增长，
而生产环境下单节点延迟由 LLM 调用主导——这正是选线程池（而非进程池）的原因：算子是 IO 密集型，
GIL 在等待网络时会释放。

并发正确性由离线测试锁定（`tests/test_dag_parallel.py`）：
4 个各 sleep 0.3 s 的算子同层执行，实测总耗时 < 0.9 s 且观测到实际重叠。

### 并发下的 payload 竞争

同层节点**读取同一份 payload 快照**，各自输出先单独收集，**整层结束后统一合并**。
若边完成边合并，两个并发节点写同一个 key 会互相覆盖，结果取决于完成顺序。
`tests/test_dag_parallel.py::test_layer_snapshot_isolation` 直接断言"兄弟节点的输出
不会泄漏进同层其他节点的输入"。

### 节点超时（诚实边界）

新增节点级 `timeout`（可在 DAG 节点上声明，或用执行器默认值）。超时按普通失败处理，
复用既有的重试 / fallback / 失败依赖跳过链路，实测 0.2 s 超时可在 5 s 内让执行器脱离
一个 sleep 30 s 的算子。

**边界要讲清楚**：Python 无法强杀线程，所以超时是**放弃等待**而非终止算子——
DAG 不再阻塞，但真正卡死的算子仍占用一个线程。要彻底约束需要进程级隔离，
本轮明确不做（见改造计划的决策记录）。

---

## 4. 可观测性

| 能力 | 实现 |
| --- | --- |
| 存活 / 就绪 | `GET /healthz`（不触碰依赖，避免依赖异常引发重启风暴）、`GET /readyz`（逐依赖明细，降级时返回 503 但仍列出**哪一项**失败） |
| 指标 | `GET /metrics`，Prometheus 文本格式，53 条 series |
| 请求关联 | `X-Request-ID` 中间件生成或透传入站值，经 `ContextVar` 贯穿日志；响应头回带 |
| 日志 | JSON 行格式（`python-json-logger`），uvicorn 自身日志一并接管，全进程单一机器可读流 |

指标命名遵循 Prometheus 约定（计数器 `_total`、耗时直方图 `_seconds`、一律基本单位）：

```
medigraph_http_requests_total{method,route,status}
medigraph_http_request_duration_seconds{method,route}
medigraph_http_in_flight_requests
medigraph_llm_call_duration_seconds{mode}          # blocking | stream
medigraph_llm_time_to_first_token_seconds          # 流式独有，与总延迟分开统计
medigraph_llm_tokens_total{kind}                   # prompt | completion
medigraph_llm_errors_total
medigraph_cache_events_total{cache,outcome}        # P2.3 起填充
```

`route` 标签取**路由模板**而非原始路径，避免路径参数把时间序列基数打爆
（`tests/test_service_api.py::test_metrics_label_uses_route_template_not_raw_path` 锁定）。

TTFT 单独建一个直方图而不是复用总延迟：流式改善的正是这个量，混在一起就看不出来了。

---

## 5. PostgreSQL 分析后端：索引与连接池

复现（需要 `medigraph-pg` 容器，见 `.env.example`）：

```powershell
python scripts/bench_pg_indexes.py --visits 200000
```

SQLite 仍是零配置默认；`ANALYTICS_BACKEND=postgres` 切换后，NL2SQL 生成的
SQLite 方言 SQL 经 sqlglot **AST 转译**为 PG 方言在连接池上执行。两引擎装载
**完全相同的行**（`relational.generate_rows` 抽出为纯函数，重构后与原实现
逐字节一致，`tests/test_p2_backends.py` 亦断言双引擎查询结果一致）——
因此跨引擎对比是真正的逻辑等价检验，而不是两份随机数据的比较。

### 5.1 索引（20 万就诊 / 27.5 万处方 / 29.8 万检验，COPY 装载 4.97 s）

| 查询形状 | 无索引 p50 | 有索引 p50 | 加速 | 计划变化 |
| --- | ---: | ---: | ---: | --- |
| 点过滤 `disease='高血压' AND age>60` | 9.06 ms | **0.11 ms** | **85.5×** | Seq Scan → Bitmap Heap Scan |
| 药物点查 `drug='二甲双胍'` | 10.30 ms | **0.03 ms** | **355×** | Seq Scan → Bitmap Heap Scan |
| 图谱边查 `head+relation` | 1.30 ms | **0.03 ms** | **42×** | Seq Scan → Bitmap Heap Scan |
| 按科室 GROUP BY（全表） | 21.04 ms | 21.49 ms | 1.0× | 仍 Seq Scan |
| 月度趋势 `substr(visit_date,1,7)`（全表） | 28.90 ms | 28.91 ms | 1.0× | 仍 Seq Scan |
| 异常检验 JOIN（全表） | 55.78 ms | 54.92 ms | 1.0× | 仍 Seq Scan |

证据：`outputs/bench_pg_indexes.json`（含每条查询的 EXPLAIN ANALYZE 计划节点）。

**诚实口径**：

- 选择性点查从索引获益巨大（计划从顺序扫描变为位图堆扫描，有 EXPLAIN 佐证）；
  **全表聚合本来就要扫全表，索引帮不上**——两类结果都如实列出，不挑好看的报。
- 之所以在 20 万行规模测量：600 行的演示库上顺序扫描本来就赢，
  诚实的结果只能是"索引是噪声"。索引价值是表规模的函数，这句话必须说全。
- 索引选择依据是 NL2SQL 路由**实际产出**的查询形状（复合索引
  `(disease, age)` 让年龄过滤留在索引扫描内），不是拍脑袋加的。

### 5.2 连接池（200 条短查询）

| 方式 | p50 | p99 |
| --- | ---: | ---: |
| 每查询新建连接 | 13.27 ms | — |
| 连接池复用 | **2.00 ms** | — |

**p50 提速 6.6×**——差值就是被池摊销掉的每次连接建立成本。

### 5.3 安全护栏（与 SQLite 执行器同构的纵深防御）

- 事务级只读：`conn.read_only = True` 使**当前**事务只读（会话级
  `default_transaction_read_only` 只对下一个事务生效，在 psycopg 隐式开启事务
  的模式下永远轮不到当前语句——实测发现该坑并以回归测试锁定：DELETE 必须收到
  `cannot execute ... in a read-only transaction`，而不是碰巧撞上外键约束）；
- 服务端 `statement_timeout`（实测 1 s 超时正确取消 `pg_sleep(10)`）；
- 行数上限（fetchmany 越界即拒）；
- 只读属性用 finally 归还前复位，不泄漏给池中下一个使用者（建库/建索引共用此池）。

---

## 6. Redis 结果缓存

复现（需要 `medigraph-redis` 容器，见 `.env.example`）：对 `/api/v1/extract`
以相同 body 连续请求两次，观察 `elapsed_ms` 与 `/metrics` 中的
`medigraph_cache_events_total`。

| 路径 | 冷（实算） | 热（缓存命中） |
| --- | ---: | ---: |
| `POST /api/v1/extract`（`backend=fast`，含神经 RE） | 1496.9 ms | **2.3 ms** |

**键设计**是这里的重点：`medigraph:{命名空间}:{sha256(输入)}:{模型指纹}`。
模型指纹由 LLM provider/model + 抽取后端 + 路由阈值哈希而成——**换模型或改后端
自动全量失效**，TTL 解决不了这类"缓存的是已下线模型的结果"的陈旧性。
NL2SQL 键还包含执行引擎（sqlite/postgres），双后端结果互不串用；失败响应不缓存。

降级策略：Redis 不可达时退化为不缓存（带 30 s 退避，不会每个请求都付一次连接
超时），**永不因缓存故障使请求失败**；`/readyz` 如实报告缓存态但不因此判死。
命中率可观测：`medigraph_cache_events_total{cache,outcome}`。

缓存开启后重跑 §1 的负载测试：**577.5 req/s、p50 21.68 ms**（同机同轮对比关缓存
530.2 req/s / 26.19 ms，约 +8.9%）。两轮分别落盘为
`outputs/bench_api_load_cached.json` 与 `outputs/bench_api_load.json`，
各自的 `cache_events` 字段记录了实际命中情况（命中轮 **hit 329 / miss 1**，
关缓存轮无任何缓存事件），可当场核对是哪一轮。

口径说明：该轮 320 请求除首个外全部命中缓存，所以这是**命中路径**的吞吐，不是混合
负载的吞吐。**这条路径上缓存的收益本来就小**——确定性 NL2SQL 只有 8 ms，省掉它换来
的是个位数百分比；真正的收益在上表 1497 ms → 2.3 ms 的抽取路径。早期版本在此处记录
过 340.7 → 444.9（+30%）的对比，那两轮是在不同负载状态下先后测的，且两次都写进同一个
`bench_api_load.json`（后写覆盖先写），无法自证是哪一轮；`--out` 参数与上述
`cache_events` 字段就是为堵这个洞加的。

---

## 7. pgvector HNSW 向量检索

复现：

```powershell
python scripts/bench_vector_search.py --n 100000 --queries 100                          # 主口径
python scripts/bench_vector_search.py --n 100000 --queries 100 --distribution gaussian  # 对抗下界
```

`LocalVectorStore` 是 numpy 全量矩阵乘：精确（recall 恒为 1）但 **O(N)** 且矩阵
常驻 API 进程内存。`PgVectorStore`（同接口）把向量放进 pgvector HNSW 索引。
基准：10 万 × 1024 维（Qwen3-Embedding-0.6B 宽度）、100 条查询、
ground truth 由 numpy 精确 top-10 给出，`ef_search` 为查询时的召回/延迟旋钮。

### 7.1 主口径：聚类分布（模拟真实文本嵌入结构）

构造：1000 个主题中心的混合分布，同主题余弦 ≈ 0.67、跨主题 ≈ 0
（真实语料块按主题聚类的形状）；查询按同一配方生成（问题会嵌入到其主题附近）。

| 配置 | recall@10 | p50 | p99 |
| --- | ---: | ---: | ---: |
| HNSW `ef_search=10` | 0.839 | 5.21 ms | 8.43 ms |
| HNSW `ef_search=20` | 0.918 | 4.74 ms | 6.18 ms |
| **HNSW `ef_search=40`** | **0.990** | **5.66 ms** | **7.21 ms** |
| HNSW `ef_search=100` | 1.000 | 5.82 ms | 7.41 ms |
| HNSW `ef_search=200` | 1.000 | 10.19 ms | 13.16 ms |
| numpy 精确基线（同轮） | 1.000 | 16.42 ms | 19.12 ms |

**ef_search=40 时：p50 快 2.9×、p99 快 2.7×，recall@10 保持 0.99**（同一轮
同机对测）。曲线整体发布而非只报一个点——ef_search 就是旋钮，"正确"设置取决于
应用要求的召回和可承受的延迟。

证据：`outputs/bench_vector_search.json`（HNSW `m=16, ef_construction=64`；
建索引 57.99 s，需要 `SET maintenance_work_mem='1GB'`——默认 64 MB 下图装不进
维护内存、退化为磁盘路径，实测同规模 >8 分钟未完成，该设置已内置于
`PgVectorStore.create_index()` 并注释说明）。

### 7.2 对抗下界：i.i.d. 高斯（如实公布）

| 配置 | recall@10 | p50 |
| --- | ---: | ---: |
| HNSW `ef_search=40` | **0.033** | 7.50 ms |
| HNSW `ef_search=200` | **0.137** | 16.92 ms |
| numpy 精确 | 1.000 | 6.64 ms |

i.i.d. 高维向量下两两余弦全部集中在 0 附近（1024 维时 σ≈0.031），
"最近邻"与第 1000 名几乎无差距——这是 **ANN 图索引的最坏情况输入**，HNSW 召回
崩塌是该机制的固有边界而非实现 bug。保留发布（`bench_vector_search_gaussian.json`）
的原因：说明主口径的聚类假设是结果成立的前提；真实嵌入若接近无结构，索引不可用。

### 7.3 诚实边界

- **当前生产语料只有数千 chunk**，numpy 暴力扫描本来就是亚毫秒且严格更准——
  索引在语料增长后才回本（numpy O(N) vs HNSW ~O(log N)，且 10 万×1024 维矩阵
  已占 400 MB 进程内存，放进 PG 还换来服务端并发）；
- 两轮 numpy 基线分别为 6.94 / 16.42 ms（机器负载不同），所有加速比**只在
  同轮内对比**，不跨轮拼数字；
- 合成分布毕竟是代理：换成真实医疗语料嵌入后应重测 recall 曲线再定 ef_search。

---

---

## 8. 复现清单

```powershell
# 启动服务
python service/main.py

# P1 三项基准
python scripts/bench_api_load.py --concurrency 16 --requests 320
python scripts/bench_stream_ttft.py --repeats 2
$env:EXTRACTION_BACKEND="fast"; python scripts/bench_dag_parallel.py --repeats 3 --simulate-latency 0.5

# P2 存储层基准（需要 medigraph-pg / medigraph-redis 容器，见 .env.example）
python scripts/bench_pg_indexes.py --visits 200000
python scripts/bench_vector_search.py --n 100000 --queries 100

# 相关离线测试（PG/Redis 不可达时自动跳过对应用例）
python -m pytest tests/test_service_api.py tests/test_dag_parallel.py `
                 tests/test_plan_validator.py tests/test_sql_guard.py `
                 tests/test_p2_backends.py -q
```

产物：`outputs/bench_api_load.json`、`outputs/bench_stream_ttft.json`、
`outputs/bench_dag_parallel.json`、`outputs/bench_pg_indexes.json`、
`outputs/bench_vector_search.json`（+ `_gaussian.json` 对抗下界）。
