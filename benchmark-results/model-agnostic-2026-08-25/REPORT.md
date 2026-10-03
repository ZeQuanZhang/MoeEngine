# MoeEngine 去模型专属策略：Qwen / DeepSeek V4 实测

> 历史测量档案：项目标识与本地路径已按 MoeEngine 发布名称规范化；此次整理未重新运行测量。参见[档案说明](../README.md)。

日期：2026-08-25  
设备：MacBook Air `Mac17,3`，Apple M5（4P+6E），24 GB 统一内存  
系统：macOS 26.5.2 (25F84)  
电源：AC 连接  

## 结论

本轮已经把 MoeEngine 调度层中按模型名称选择的性能策略拆除。保留下来的
模型分支只负责数值正确性和架构差异，例如 Qwen GDN 与 DeepSeek 压缩注意力；
缓存、预取、slot-map 和自动容量不再查询模型名称。

最终结果：

- **Qwen3.6-35B-A3B：26.80 tok/s**（最终二进制，官方 short case，
  516 generated tokens，`stop=endOfTurn`）；
- **DeepSeek V4 Flash 0731：1.87 tok/s**（64 generated / 63 decode，
  CPU，4 compute threads，8 I/O workers）；
- 最终 Qwen 与 DSV4 进程均报告 0 swaps；
- 最终 DSV4 区间系统 `Swapouts` 为 **1116130 → 1116130**，没有新增换出。

通用化对 Qwen 没有表现出可辨认的短提示损失；DSV4 当前实测低于上一轮
2.06 tok/s 的专门调优结果约 9%，但两轮提示和整机热状态不完全相同，不能把
全部差值归因于某一代码改动。

## 拆除了什么

### Mference / Qwen 路径

1. 删除 Qwen3.6 专属的 96/32/16 自动缓存档位。
2. 自动档位改为只读取包几何与机器资源：expert stride、routed layer 数、
   experts/layer、core weights 大小和物理内存。
3. 删除 DeepSeek V4 专属的 shadow-prefetch 默认开启；所有模型默认不做投机
   读取，实验模式仍可显式启用。
4. 删除 slot-map 的 `family == qwen36` 门槛，改用图能力条件：无 sandwich
   norm、非 linear layer、top-k 几何兼容并且 runtime 能提供 slot map。

在本机 Qwen 包上，一槽跨 40 层的成本为 70,778,880 bytes；资源公式给出约
45 槽的预算，再向下落到公共允许档位 **32 slots**，约 2.11 GiB routed-expert
cache。这个结果没有使用 `qwen` 字符串。

### Palm / DSV4 路径

1. 删除为 43-layer DSV4 手调的固定 32-sample 预取裁剪门槛。
2. 改为 99.99% Wilson Bernoulli upper confidence bound：只要某个 rank 仍有
   统计可能达到 80% usefulness 就继续探索；否则裁掉。允许连 rank 0 一起
   关闭，不再强制保留“至少一半候选”。
3. 增加 `--ssd-cache-mb auto`，不读取模型名称。
4. 初版 `auto = physical RAM / 8` 在本机得到 3072 MiB。真实 A/B 证明它少读
   但不更快，而且整个失败实验阶段出现系统换出风险，因此最终改成
   `min(physical RAM / 8, 1 GiB)`。这个保守上限适用于无法在构造 runtime 前
   获得 core/sidecar 完整工作集的 Palm 前端。

## 仍然保留的通用机制

- Router-aware GDSF admission/eviction 与 ghost refault history；
- resident/ready expert wave regrouping；
- speculative I/O 并发上限以及 demand worker reserve；
- hit-first execution 与异步 demand fill；
- capability-gated GPU slot map；
- 显式模型文件读取，权重不交给 OS swap 管理。

这些机制依据路由观测、缓存压力、I/O 类型或图能力工作，不依据模型 ID。

## Qwen 正式结果

模型：`Qwen3.6-35B-A3B` 4-bit `.gturbo`  
模型 manifest SHA-256：
`bfaa17cbca7755969c9a1e02216b5bb28f55701529c0f07dafbc07080517badd`  
最终 `MferenceCLI` SHA-256：
`3c7746c04a68ddd2d7bd95e66430fc0a3ab5f5b7322b5551511dbda426c6f963`

公共 community prompts，temperature 0.2、top-k 64、top-p 0.95、context
4096、max-new 1024、固定 seeds、`--expert-cache-slots auto`：

| Case | Warm-up | Measured | Prompt | Generated | Stop |
|---|---:|---:|---:|---:|---|
| short-explanation | 25.822 | 26.225 tok/s | 62 | 516 | endOfTurn |
| medium-review | 24.505 | 24.658 tok/s | 426 | 688 | endOfTurn |
| long-synthesis | 20.659 | 13.243 tok/s | 2940 | 668 | endOfTurn |

long measured 是六轮连续负载末端的明显 fanless thermal outlier；输出仍与其
warm-up 逐字节一致，不应把 13.24 tok/s 解释成算法稳定速度。

最后把 slot-map 的模型名门槛改成能力门槛并重编后，额外复测 short：

```text
[stop=endOfTurn prefill=62tok/2.37s new=516tok decode=19.26s tok/s=26.798]
```

最终 short 输出 SHA-256：
`aed1d8fba029e982c238ef6e83cdc71e8fdd5863397bdf808804b077a9471c32`，
与修改前后以及历史 Qwen campaign 一致。

三个正式 case 的每个进程都报告 0 swaps，峰值 RSS 为约 2.25-2.42 GiB。

## DSV4 正式结果

模型：DeepSeek V4 Flash 0731，156,712.8 MB `.mollm`  
最终 `mollm_bench` SHA-256：
`cae41df811ac3c970924a7e5e0d2b1802e5645ff04561a8745b69937cb49190b`

最终策略：CPU、4 compute threads、8 I/O workers、MoeEngine-GDSF、
`--ssd-cache-mb auto`（解析为 1024 MiB）、temperature 0、64 generated
tokens、无内置 warm-up。

| 指标 | 最终值 |
|---|---:|
| Decode throughput | **1.87 tok/s** |
| TPOT | 533.41 ms |
| TTFT | 13,775.84 ms |
| Decode wall | 33.605 s |
| Cache capacity reported | 1,073.7 MB |
| Peak RSS (runtime sampler) | 15,852.5 MB |
| Logical expert load | 199,123.0 MB |
| Demand-origin load | 102,395.8 MB |
| Prefetch-origin load | 96,727.2 MB |
| Decode I/O wait | 13,558.3 ms |
| Process swaps | 0 |
| System Swapouts before / after | 1116130 / 1116130 |
| System swap used before / after | 3013.25 / 2997.25 MiB |

最终输出与所有同提示控制轮相同。64-token 吞吐测试按上限停止，
`hit_eos=false` 是预期行为，不是生成错误。

### 失败的 3 GiB 通用预算实验

为了避免只挑有利数字，保留失败结果：

| 顺序块 | 3 GiB auto | 1 GiB fixed |
|---|---:|---:|
| auto → fixed | 1.79 | 1.77 tok/s |
| fixed → auto | 1.93 | 1.76 tok/s |
| 两轮均值 | **1.775** | **1.850 tok/s** |

3 GiB 将 logical load 从约 199.0 GB 降到 173.7 GB（约 -12.7%），但平均
吞吐反而低约 4.1%。整个综合 campaign 的 Swapouts 从 978618 增至
1116130；因为中间未在每一轮之间采样，不能严格把全部增量归因到某一个
进程，但 3 GiB locked-cache 是唯一显著扩大工作集的实验。最终 1 GiB
复测单独证明了零新增 swapout，因此 3 GiB 路线已经退出 `auto`。

## 验证

### Palm

- Portable release build：成功；
- 全量 CTest：49/49 通过；
- 最终 auto-cap 修改后的 focused tests：2/2 通过；
- Wilson policy 覆盖：无样本继续探索、统计无望时裁 tail、允许裁掉完整
  prefix；
- 物理内存 auto-cap 覆盖 24 GiB 与 4 GiB host；
- 最终真实 DSV4：生成成功、process swaps 0、system swapouts delta 0。

### Mference

- Release `MferenceCLI` build：成功；
- RuntimeConfiguration：4/4 通过；
- ResidencyAutoProfile：3/3 通过；
- CLIArguments：26/26 通过；
- 最终 slot-map Metal parity：1/1 通过；
- 最终真实 Qwen short：`stop=endOfTurn`、输出 hash 不变。

完整 Swift suite 共 1043 tests，首次在受限沙箱中运行时有 466 个环境失败，
原因是 Metal device 不可见和 Gemma tokenizer 禁止联网；相关本轮逻辑测试均
单独通过，Metal slot-map parity 随后在允许 Metal 的环境中通过。这里不把
环境失败包装成“全量测试通过”。

## 命令

Mference release build（系统 Swift 6.3.3 与 SDK interface 6.3.2 对齐）：

```bash
env SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
  CLANG_MODULE_CACHE_PATH=/path/to/cache/clang \
  SWIFT_MODULECACHE_PATH=/path/to/cache/swift \
  swift build --disable-sandbox -c release --product MferenceCLI \
  -Xswiftc -interface-compiler-version \
  -Xswiftc 'Apple Swift version 6.3.2 effective-5.10 (swiftlang-6.3.2.1.2 clang-2100.0.123.2)'
```

Qwen final short：

```bash
.build/release/MferenceCLI \
  --model /path/to/MoeEngine/artifacts/qwen3.6-35b-a3b-4bit.gturbo \
  --messages-file docs/benchmark-prompts/real-generation-v1/short-explanation.json \
  --max-new 1024 --max-context 4096 --temperature 0.2 \
  --top-k 64 --top-p 0.95 --seed 20260721 \
  --expert-cache-slots auto --verify trusted-receipt
```

DSV4 final：

```bash
./build_portable/mollm_bench \
  --package /path/to/MoeEngine/artifacts/deepseek-v4-flash-0731/mollm/deepseek_v4_flash_native.mollm \
  --prompt-file /path/to/MoeEngine/vendor/Mference/benchmark-results/model-agnostic-2026-08-25/dsv4/prompt.txt \
  --max-new-tokens 64 --n-ctx 4096 --threads 4 --device cpu \
  --ssd-cache-mb auto --ssd-io-workers 8 --ssd-policy moeengine-gdsf \
  --temperature 0 --warmup 0 --output kv
```

## Protocol deviations

- Qwen 使用 trusted install receipt，避免每个新进程重复 SHA-256 扫描整个
  expert pool；模型 manifest 与 executable hash 均已记录。
- Qwen 完整三-case campaign 在删除最后一个冗余 Qwen family guard 前完成；
  对 Qwen 来说 guard 前后分支条件相同。最终二进制另跑 short，输出 hash
  不变且达到 26.798 tok/s。medium/long 没有在最终 hash 上重跑。
- DSV4 使用 Palm 的既有 64-token sustained protocol，不是 Mference public
  community protocol，因此不可把两模型 tok/s 直接当作同工作量横向排名。
- 两个 vendored runtime 都没有可用 `.git` metadata；以源码文件 hash、模型
  hash 和 executable hash 固定身份。

## Raw evidence

这些路径标识历史实验工作区中的原始证据，本次发布未包含这些目录。保留目录名用于对应实验记录，不作为当前仓库内链接。

- Qwen：`/path/to/MoeEngine/vendor/Mference/benchmark-results/model-agnostic-2026-08-25/qwen/`
- DSV4：`/path/to/MoeEngine/vendor/Mference/benchmark-results/model-agnostic-2026-08-25/dsv4/`

