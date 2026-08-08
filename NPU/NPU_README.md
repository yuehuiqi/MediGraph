# MedFusion-Softmax NPU 优化

本目录是 MediGraph ModelEngine 项目的昇腾 NPU 算子优化模块，对应比赛任务二的“NPU 算子优化实现”加分项。

核心实现包含：

- FP32 `[N,32]` 行级稳定 Softmax Ascend C 算子；
- `温度校准 + 医疗标签 Mask + Softmax` 单 Kernel 融合算子；
- AscendCL/C ABI 运行时和 Python `ctypes` 接口；
- CPU、`torch_npu`、Ascend C 的正确性、吞吐、延迟、能效和 msProf 对比；
- 从干净构建到自动校验、结果汇总与归档的一键复现。

完整设计、优化过程和实验分析见
[《ModelEngine 开源项目贡献赛 NPU 技术报告》](./ModelEngine开源项目贡献赛_NPU技术报告.md)。

## 1. 正式结果

正式运行 ID：`20260627_212142_full`
硬件：Ascend 910B3（64 GB HBM）
软件：CANN 8.3.RC1、PyTorch 2.7.1、`torch_npu` 2.7.1
测试配置：FP32 `[200000,32]`、200 次吞吐迭代、200 次延迟采样、5 轮重复、`BlockDim=40`、`temperature=1.35`
自动校验：`Validation: PASS`

### 1.1 纯 Softmax

| 系统               |    吞吐（rows/s） | 吞吐 CV |                P50 |                P99 |
| ------------------ | ----------------: | ------: | -----------------: | -----------------: |
| CPU / PyTorch      |          26.864 M |  30.08% |           2.971 ms |          60.021 ms |
| NPU /`torch_npu` |           1.258 B |   0.19% |           0.213 ms |           0.236 ms |
| NPU / Ascend C     | **3.412 B** |   2.26% | **0.100 ms** | **0.117 ms** |

Ascend C 相对 `torch_npu`：

- 吞吐提升 **2.71×**；
- P50/P99 分别降低 **53.0%/50.3%**；
- 相对 CPU 参考的最大绝对误差为 **1.49×10⁻⁷**。

CPU 数据受共享服务器调度影响较大，主要用于数量级对照；核心性能结论以同一块 910B3 上的 `torch_npu` 与 Ascend C 对比为准。

### 1.2 端到端延迟

口径为 `pageable H2D + kernel + D2H`，并复用已分配设备内存。

| 系统          |                Mean |                 P50 |                 P99 |
| ------------- | ------------------: | ------------------: | ------------------: |
| `torch_npu` |           5.1578 ms |           5.2557 ms |           6.4079 ms |
| Ascend C      | **3.4260 ms** | **3.3718 ms** | **5.4515 ms** |

### 1.3 MedFusion-Softmax 融合算子

融合语义：

```text
softmax(logits / temperature + additive_label_mask)
```

| 系统                     |    吞吐（rows/s） | 吞吐 CV |                P50 |                P99 |
| ------------------------ | ----------------: | ------: | -----------------: | -----------------: |
| `torch_npu` 三算子组合 |           1.012 B |   0.10% |           0.279 ms |           0.310 ms |
| Ascend C 单 Kernel 融合  | **3.156 B** |   0.41% | **0.127 ms** | **0.145 ms** |

融合实现相对 `torch_npu` 组合基线：

- 吞吐提升 **3.12×**；
- P50/P99 加速 **2.21×/2.13×**；
- 整卡 rows/J 提升 **3.00×**；
- 扣除空闲功耗后的动态 rows/J 提升 **2.70×**；
- 最大绝对误差为 `0`，最大行和误差为 `2.38×10⁻⁷`，禁用标签概率为 `0`。

msProf 实测纯算子和融合算子的 Task Duration 分别为 **58.24 μs** 和 **62.66 μs**。融合后每核 Vector FOPS 增加 **29.3%**，Task Duration 仅增加 **7.59%**，说明新增语义主要在 UB 内完成。

> 上述数字均来自 `NPU_results/summary.json` 及其关联原始结果，不是人工估算值。

## 2. 目录结构

```text
NPU/
├── NPU_README.md
├── ModelEngine开源项目贡献赛_NPU技术报告.md
├── NPU_source/
│   ├── ascendc/       # 两个 Ascend C Kernel、Host 运行器和构建脚本
│   ├── benchmark/     # 吞吐、延迟、能效及对比基准
│   ├── integration/   # Python/C ABI 集成与正确性校验
│   ├── scripts/       # 全流程编排、汇总和归档脚本
│   └── run_all.sh     # 统一复现入口
└── NPU_results/
    ├── summary.json
    ├── run_config.json
    ├── environment.json
    ├── repeats/
    ├── profiler/
    ├── logs/
    ├── source.sha256
    └── 其他原始 JSON 与设备状态文件
```

两份 Markdown 的职责：

- `NPU_README.md`：快速了解、复现入口和证据导航；
- `ModelEngine开源项目贡献赛_NPU技术报告.md`：完整方案、创新点、实验方法、结果分析和边界说明。

`NPU_results/` 不再保存重复的 Markdown 报告；`summary.json` 是机器可读的权威汇总，原始 JSON、日志和 Profiler 文件用于追溯。重跑时 `artifacts/runs/<RUN_ID>/REPORT.md` 只是一份随单次运行生成的临时快照，不属于正式提交文档。

## 3. 环境要求

- Linux AArch64；
- Ascend 910B3 或兼容环境；
- CANN 8.3.RC1；
- Python 3.11；
- PyTorch 2.7.1、`torch_npu` 2.7.1；
- NumPy 1.26.4；
- CMake 3.16.5、GCC 7.3.0；
- 可用的 `npu-smi` 与 `msprof`。

默认 Python 环境为：

```text
/home/ma-user/anaconda3/envs/PyTorch-2.7.1
```

若路径不同，可在运行前设置：

```bash
export PYTORCH_ENV=/path/to/your/python/env
```

## 4. 一键复现

在仓库根目录执行：

```bash
cd NPU/NPU_source
./run_all.sh quick
```

`quick` 模式适合演示：执行干净构建、集成正确性、单轮纯算子/融合基准和自动汇总。

正式全量复现：

```bash
cd NPU/NPU_source
./run_all.sh full
```

`full` 模式额外执行：

- 纯算子与融合算子五轮重复；
- 端到端五轮测试；
- 纯算子与融合算子能效测试；
- 两个 Kernel 的 msProf；
- 环境、设备状态和源码校验记录；
- 结果汇总、Validation 与归档。

成功标志：

```text
FULL_RUN_OK
Validation: PASS
exit code = 0
```

每次运行写入独立目录：

```text
NPU_source/artifacts/runs/<timestamp>_<quick|full>/
```

`NPU_source/artifacts/latest` 和 `NPU_source/results` 指向最近一次运行。仓库随附的正式结果固定保存在 `NPU_results/`，便于离线评审。

## 5. 证据导航

| 内容                          | 文件                                                      |
| ----------------------------- | --------------------------------------------------------- |
| 权威汇总、加速比和 Validation | `NPU_results/summary.json`                              |
| 正式参数                      | `NPU_results/run_config.json`                           |
| 软件与硬件环境                | `NPU_results/environment.json`                          |
| 纯算子五轮                    | `NPU_results/repeats/pure/run_*.json`                   |
| 端到端五轮                    | `NPU_results/repeats/e2e/run_*.json`                    |
| 纯算子能效                    | `NPU_results/energy_compare.json`                       |
| 融合五轮                      | `NPU_results/fused_compare_repeated.json`               |
| 融合能效                      | `NPU_results/fused_energy_compare.json`                 |
| 纯算子 Profiler               | `NPU_results/profiler/pure/`                            |
| 融合 Profiler                 | `NPU_results/profiler/fused/`                           |
| 构建和运行日志                | `NPU_results/logs/`                                     |
| 实验前后设备状态              | `NPU_results/npu_smi_before.txt`、`npu_smi_after.txt` |
| 参与运行的源码校验            | `NPU_results/source.sha256`                             |

## 6. 开源许可

项目根目录提供 MIT `LICENSE`；第三方软件与工具链遵循各自许可证。
