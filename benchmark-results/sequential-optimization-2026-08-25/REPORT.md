# MoeEngine 顺序优化与真实 DSV4 验证报告

> 历史测量档案：项目标识与本地路径已按 MoeEngine 发布名称规范化；此次整理未重新运行测量。参见[档案说明](../README.md)。

日期：2026-08-25  
设备：MacBook Air，Apple M5（4P+6E），24 GB 统一内存  
系统：macOS 26.5.2  
模型：DeepSeek V4 Flash 0731，156,712.8 MB `.mollm` 包  
默认测试配置：CPU、4 compute threads、8 SSD I/O workers、1 GiB 专家缓存、MoeEngine-GDSF、`n_ctx=4096`、temperature 0。

## 结论

这轮没有靠“换个 benchmark”制造数字，而是逐项实现、同二进制开关 A/B、失败即回退。最终保留三项：

1. decode 期间对刚完成 I/O 的专家重新成批，减少小 GEMV 的线程池 barrier；
2. 限制同时在途的 speculative I/O，8 个 worker 默认最多 6 个跑预取，至少为真实缺页留 2 个；
3. router rank 学习从 128 个样本提前到 32 个样本收敛，更早停止低命中 tail-rank 预取。

最终 64-token 持续测试达到 **2.06 token/s（TPOT 486.31 ms）**。测试前后系统已有 swap 占用均为 2669.69 MB，`Swapouts` 均为 959503，增量均为 **0**：本次推理没有把 SSD 当虚拟内存。

Metal 与 I8MM 都做了真实实现和测试，但没有默认启用：当前形状下它们比已有 CPU SDOT 路径更慢。

## 按顺序的实验结果

### 1. MXFP4 I8MM / SMMLA

实现了独立 `-march=armv8.6-a+i8mm` translation unit 和运行时 dispatch，数值测试通过。直接 row-major 权重布局的 SMMLA 仍比现有四行 SDOT 内核慢：

| 形状（4 threads） | SDOT p50 | I8MM p50 | 结果 |
|---|---:|---:|---:|
| M1, K4096, N4096 | 0.1309 ms | 0.1405 ms | I8MM 慢约 7.3% |
| M1, K2048, N4096 | 0.0644 ms | 0.0701 ms | I8MM 慢约 8.9% |

结论：保留为 `MOLLM_MXFP4_I8MM=1` 的实验开关，默认继续用 SDOT。要让 SMMLA 真正占优，需要在模型转换阶段生成原生 8-row tile 权重布局，不能只在热循环里临时重排。

### 2. 真实 shape profile 与 ready-wave regrouping

真实 DSV4 profile 显示，MXFP4 routed expert 是最大的可控矩阵族；同一层内的 batch=2..4 明显比逐专家 dispatch 更高效。初始扫描之后仍有专家在前一批计算期间完成 I/O，原实现会把它们逐个执行。

加入 ready-wave 重新分组后：

| 指标 | 修改前 | 修改后 | 变化 |
|---|---:|---:|---:|
| 16-token TPOT | 486.65 ms | 476.15 ms | -2.16% |
| 直接 MXFP4 single calls | 2078 | 1606 | -22.7% |

尝试把 batched GEMV 改为静态 N-shard 后，batch=2/3 内核约慢 9%，已回退。

### 3. demand-worker quota

旧调度只保证 demand job 在“队列里”优先；如果 8 个 worker 都已进入不可抢占的 speculative `pread()`，新 demand 仍会等待。新调度限制 speculative in-flight 数量，默认保留约 25% worker 给 demand。

两对 16-token 同二进制 A/B：

| 策略 | TPOT 两次 | 均值 | decode I/O wait 均值 |
|---|---:|---:|---:|
| 旧：8/8 worker 可跑预取 | 486.62 / 478.61 ms | 482.62 ms | 3019.9 ms |
| 新：最多 6/8 跑预取 | 481.31 / 477.37 ms | 479.34 ms | 3000.5 ms |

收益约 0.68%，幅度小但两对方向一致，并消除了真实的设备侧优先级反转。`MOLLM_SSD_DEMAND_WORKERS=0` 可恢复旧行为用于 A/B。

### 4. 合并 data/scales 读取

尝试把每个专家的 4 个 component job 合成 gate/down 两个 tensor job。结果明确变差：

| 指标 | 4 component jobs | 2 tensor jobs | 变化 |
|---|---:|---:|---:|
| TPOT | 480.80 ms | 527.03 ms | +9.6%（更慢） |
| decode I/O wait | 3017 ms | 3735 ms | +23.8% |

原因：这台机器上的瓶颈不是 syscall 数，而是设备队列深度和首个专家完成时间。4 个 component job 能并行占满 I/O 队列；合并后并行度下降。该修改已完全回退。

### 5. 更快的 router-rank 学习

现有策略最终会把 DSV4 的 6 个预测 rank 裁到前 3 个，但 128 个全局反馈样本约等于前三个 decode token 都在读取完整 tail。门槛改为 32 后，从第二个 token 起即可收敛；仍要求 rank 命中率至少 80%，并至少保留一半候选。

结构性结果（15 decode token）：

| 指标 | 128 samples | 32 samples | 变化 |
|---|---:|---:|---:|
| prefetch-origin load | 28.18 GB | 24.83 GB | -11.9% |
| total logical load | 51.06 GB | 49.57 GB | -2.9% |
| unused prefetch | 3.49 GB | 2.10 GB | -39.9% |

反序的干净 A/B 为 477.82 → 475.55 ms（约 0.48%）；另一控制轮出现 514 ms I/O 尾延迟，而 32-sample 两轮稳定在 475–478 ms。因此报告只采用保守的约 0.5% 吞吐收益，主要价值是减少写实的 SSD 流量和尾延迟风险。`MOLLM_SSD_PREFETCH_MIN_SAMPLES=128` 可恢复旧门槛。

### 6. 整段 routed-MoE Metal

当前 Metal 路径已经不是逐 matmul 往返：hidden quantize、gate/up、SwiGLU、intermediate quantize、down 在一个 GPU command 中提交，CPU shared expert 与其并行。真实 DSV4 结果仍明显落后：

| 后端 | TPOT | decode throughput | 645 个 MoE 层 |
|---|---:|---:|---:|
| 最终 CPU | 约 476–486 ms | 2.06–2.10 tok/s | 短 trace：1.174 s / 129 层 |
| routed-MoE Metal hybrid | 669.01 ms | 1.49 tok/s | 8.141 s / 645 层；短 trace：1.905 s / 129 层 |

3-token trace 进一步拆分：

- CPU routed expert compute：165 ms；
- Metal GPU routed compute：801 ms，单是 GPU 内核就慢约 4.8 倍；
- Metal routed wall：1856 ms，GPU 之外还有约 1055 ms 的 I/O event、command-buffer wait 和 CPU/GPU 边界。

结论：M=1、top-k=6 的小 GEMV 目前更适合 CPU。Metal 只有在整个 DSV4 block/graph、KV state、shared expert 与路由都常驻 GPU，且能跨层异步流水时才值得重新评估；当前 hybrid 不应成为默认路径。

## 最终持续测试

命令核心参数：64 generated tokens、63 decode tokens、4 threads、8 I/O workers、1 GiB cache、MoeEngine-GDSF、CPU。

| 指标 | 最终值 |
|---|---:|
| Decode throughput | **2.06 token/s** |
| TPOT | **486.31 ms** |
| TTFT | 3519.74 ms |
| Decode wall time | 30.637 s |
| Peak RSS | 14,590.9 MB |
| Expert cache resident | 1069.5 MB |
| Decode I/O wait | 11.690 s（185.56 ms/token） |
| Logical load | 193,788.6 MB（3076.0 MB/token） |
| Demand-origin load | 101,446.6 MB（1610.3 MB/token） |
| Prefetch-origin load | 92,342.1 MB（1465.7 MB/token） |
| Useful prefetch | 85,416.7 MB |
| Unused prefetch | 6,831.7 MB |
| Swap used before / after | 2669.69 / 2669.69 MB |
| Swapouts before / after | 959503 / 959503 |

这里的“swap 增量为 0”与“系统从未使用 swap”不是一回事。测试开始前 macOS 已有 2.67 GB 历史 swap；可证事实是本轮推理没有新增 swap，也没有把专家权重交给虚拟内存系统管理。模型权重仍由 MoeEngine 的显式 1 GiB cache、GDSF admission/eviction 和 direct expert reads 管理。

## 验证

- Portable build：成功；
- 49/49 CTest：全部通过；
- Metal operator parity：全部通过；
- I8MM 开/关的 matmul 数值测试：通过；
- 最终生成文本正常且确定性 prompt 输出一致；
- 64-token 持续测试 swapout 增量：0。

## 下一步（不应与本轮结果混为一谈）

1. 在模型转换器中生成 MXFP4 原生 8-row I8MM tile，再决定 SMMLA 是否值得；
2. 把 ready-wave 的 batch-size、I/O wait 与 route rank 纳入在线 cost model，而不是固定阈值；
3. 若继续 Metal，目标必须是整个 DSV4 block/graph 常驻与跨层流水，不再做每层同步 hybrid；
4. 用固定冷却窗口、随机化 A/B 顺序和至少 5 次重复生成可发表的置信区间。

