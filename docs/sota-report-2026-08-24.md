# MoeEngine / DeepSeek-V4-Flash-0731：重复 CPU 实验审查

测试日期：2026-08-24（Asia/Shanghai）

## 实验范围与结论

本报告记录单台无风扇 MacBook Air（Apple M5，24GB 统一内存）上的三个
独立进程实验。使用 156,712,820,637 字节的 DeepSeek-V4-Flash-0731 原生
FP4/FP8 混合精度权重包，batch 1、贪心解码、无 MTP；routed experts
按 Router 结果从 SSD 读取，不要求整个模型驻留内存。

128-token 测试的最佳 decode 结果为 **2.13 token/s**，三次运行的中位数为
**1.57 token/s**。此前 15 个 decode tokens 测得的 **2.34 token/s** 仅是
短突发结果，不能代表长时间服务吞吐。三轮数据说明这组硬件与参数下能够完成
生成，同时存在显著运行间波动；它们不构成跨硬件、跨精度或跨运行时的性能排名。

## 可复述的测量结果

> 在一台 24GB Apple M5 MacBook Air 上，以 4GiB 专家 RAM 缓存和 SSD
> 专家存储运行 156.7GB 原生混合 FP4/FP8 权重包。三个独立进程分别生成
> 128 tokens，decode 最佳值为 2.13 tok/s，中位数为 1.57 tok/s，
> peak RSS 为 16.7–17.1GB。结果仅适用于下述配置与测量协议。

在完成全模型 logit 或困惑度对齐之前，不能声称与官方实现质量完全相同。
权重格式、内核单测和确定性输出是不同层次的检查，不能单独证明全模型统计质量等价。

## 测试系统

| 项目 | 值 |
|---|---|
| 机器 | MacBook Air `Mac17,3` |
| SoC | Apple M5，10 CPU 核（4P + 6E） |
| 统一内存 | 24GB |
| 散热 | 无风扇 |
| 电源 | AC，电池 100% |
| macOS | 26.5.2，build `25F84` |
| 编译器 | Apple clang 21.0.0，arm64 |
| 构建 | CMake `Release` |
| 模型文件 | `deepseek_v4_flash_native.mollm` |
| 模型文件大小 | 156,712,820,637 bytes（约 146.0GiB） |
| 模型配置 | 43 层，256 routed experts，top-6，FP4 experts，FP8 dense |
| 推理二进制 SHA-256 | `25e2c0916f4e816e0d1e73e89965b0604d7596d233b62af65d5e86fda29c1297` |
| `config.json` SHA-256 | `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023` |

为避免额外扫描 146GiB 文件，本次没有重新计算整个 `.mollm` 的 SHA-256；
精确字节数、配置哈希和二进制哈希共同作为本次本地复现标识。

## 测试口径

- prompt：`给我讲一个故事`（8 tokens）
- 输出：128 tokens，其中首 token 计入 TTFT，后 127 tokens 计入 decode
- 采样：temperature 0，贪心
- context：4096
- CPU threads：4
- SSD I/O workers：8
- 专家策略：`moeengine-adaptive`
- 专家 RAM cache：4096MiB（运行时实际 4295.0MB）
- dense mmap：锁定 1211.5MB
- CPU sidecars：锁定 9358.9MB
- MTP：关闭
- compact `wo_a` sidecar：关闭，使用吞吐更快的 FP16 展开布局
- 每轮使用全新进程；用 `caffeinate -i` 防止休眠
- 第一轮到第二轮静置 45 秒；第二轮到第三轮静置约 180 秒
- 不清理 macOS 文件页缓存，因此这些是“进程冷启动”，不是可证明的
  “物理 SSD 冷缓存”结果
- 模型只从内置 NVMe 读取；没有使用或写入 T7，没有启用自定义系统 swap，
  没有记录大型 trace

## 三轮原始结果

| 指标 | Run 1 | Run 2 | Run 3 | 中位数 |
|---|---:|---:|---:|---:|
| 静置时间 | 初始状态 | 45s | 180s | — |
| load | 19.400s | 18.350s | 19.833s | 19.400s |
| TTFT | 3.796s | 4.182s | 3.872s | 3.872s |
| TPOT | **470.32ms** | 784.10ms | 637.10ms | **637.10ms** |
| decode | **2.13 tok/s** | 1.28 tok/s | 1.57 tok/s | **1.57 tok/s** |
| peak RSS | 17.147GB | 17.047GB | 16.655GB | 17.047GB |
| expert wait/token | 170.79ms | 65.81ms | 126.13ms | 126.13ms |
| logical expert load/token | 2081.93MB | 2080.56MB | 2081.30MB | 2081.30MB |
| acquired expert bytes/token | 3449.29MB | 3449.29MB | 3449.29MB | 3449.29MB |
| prediction accuracy | 76.7% | 76.7% | 76.7% | 76.7% |

三轮合计 381 个 decode tokens，用时 240.224 秒，加权吞吐约
**1.59 token/s**。最佳与最差速度相差约 40%，因此不能用单个短跑数字代表
这台无风扇机器的长期服务能力。

此前相同参数的 16-token 测试达到 2.34 token/s（427.7ms TPOT）。它证明
了冷机短突发能力，但由于只有 15 个 decode tokens，本报告不把它作为持续
吞吐主指标。

## 正确性与确定性

- 三次生成文本逐字一致，均以 `## 会唱歌的旧钟表` 开头，并在第 128 个
  token 停在相同位置。
- 三轮的 `expert_bytes_acquired_mb_per_decode_token` 完全一致：3449.290752MB。
- Router 的逐层预测准确率序列一致：
  `0.981,0.940,0.874,0.752,0.604,0.448`。
- 完整 CPU 测试套件通过 48/48；紧凑和展开 `wo_a` 路径还包含 FP16
  subnormal 精确性测试。

确定性输出只能证明三次运行走出了稳定的相同路径。它不能替代官方 logit
向量、长语料困惑度或任务集评测。

## 运行间波动分析

本次实验未能在无特权环境下读取本机热状态传感器。下面依据算子计时分析
退化发生的位置，无法直接确认温度、频率或功耗变化。

Run 2 的专家等待比 Run 1 **降低 61.5%**（170.79 → 65.81ms/token），但
总 TPOT 反而 **增加 66.7%**。同一时间：

| decode 算子平均耗时 | Run 1 | Run 2 | 变化 |
|---|---:|---:|---:|
| MATMUL | 0.291ms | 0.572ms | +96.6% |
| DSV4_GROUPED_LINEAR | 0.978ms | 1.970ms | +101.4% |
| DSV4_SPARSE_ATTN | 0.315ms | 0.808ms | +156.5% |

仅用 SSD 变慢无法解释这一组计时变化：专家等待减少，而计算算子耗时增加。
热状态或功耗限制是可能原因，但本次没有直接遥测来区分它们。180 秒静置后的
Run 3 为 1.57 token/s，仍低于 Run 1 的 2.13 token/s。

因此建议分别报告：

- 冷机/短突发峰值：2.34 token/s（15 decode tokens）；
- 最佳 128-token 持续值：2.13 token/s；
- 三轮 128-token 中位数：1.57 token/s；
- 本组三轮中的最低值：1.28 token/s，该轮距前一轮静置 45 秒。

## 其他公开结果的历史参考

下表保留实验时收集的参考数字及链接，未在同一设备和测试工具中复现，也不代表
对外部页面当前内容的重新核验。模型精度、SoC、内存、计算后端和测试协议不同，
不能据此判断本项目领先或落后；对缓存策略的判断应使用相同后端内的受控比较。

| 类别 | 公开结果 | 与本项目是否同口径 |
|---|---:|---|
| 本报告：M5 Air 24GB + 156.7GB 原生混合精度 + SSD experts | 最佳 2.13；中位数 1.57 tok/s | 本机三轮观测 |
| M5 Pro 64GB，mixed 4/8-bit 质量路径，峰值内存 17.7GB | 2.04 tok/s 中位数；跨任务 1.52--2.30 | 接近精度/内存，不同硬件和运行时 |
| M5 Pro 24GB，2-bit 动态量化，约 91GB | 最高 4.8 tok/s | 不同 SoC、精度和模型大小 |
| M5 Pro 64GB，ds4 Q2 SSD streaming | 11.4 tok/s | 不同 SoC、Q2、Metal 后端 |
| M3 Ultra 256GB，全驻留 oMLX | 约 41--42 tok/s | 完全不同内存/带宽等级 |

公开来源：

- [mixed 4/8-bit SSD streaming 项目](https://github.com/kk-r/dsv4-streaming)
- [24GB M5 Pro、2-bit 实验帖](https://www.reddit.com/r/LocalLLaMA/comments/1vdbix4/deepseekv4flash_284b_on_53gb_of_memory/)
- [oMLX 社区性能表](https://omlx.ai/benchmarks/performance?page=0)
- [DeepSeek 官方模型页](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)

## 复现命令

先按 [mollm 构建说明](mollm-integration.md) 准备后端与模型包，在仓库根目录运行。
历史实验的构建目录为 `vendor/Palm-Infra/build_moeengine`；以下使用当前说明中的
`build/palm`。构建时配置、模型包与运行参数都应一并记录。确保没有设置最大压缩模式：

```bash
env -u MOLLM_DSV4_WOA_COMPACT_SIDECAR \
  caffeinate -i build/palm/mollm_bench \
  --package artifacts/deepseek-v4-flash-0731/mollm/deepseek_v4_flash_native.mollm \
  --prompt '给我讲一个故事' \
  --max-new-tokens 128 \
  --n-ctx 4096 \
  --threads 4 \
  --temperature 0 \
  --ssd-cache-mb 4096 \
  --ssd-io-workers 8 \
  --ssd-policy moeengine-adaptive \
  --warmup 0 \
  --profile \
  --output kv
```

公开复现时应记录每轮完整 stdout、运行前后的 `vm_stat`、AC/电池状态和静置
时间。运行顺序与散热状态可能影响结果；缺少这些记录时，无法将观测差异
可靠地归因于策略。

## 尚未完成的验证

1. 官方实现对齐：固定短 prompt 的逐 token logit/top-k 对比，或完整
   WikiText-2 困惑度。
2. 更严格热态协议：隔夜冷机首轮，以及连续 10 分钟稳态窗口；若允许，采集
   官方功耗/频率遥测。
3. 完整测量源码与日志追溯：本次测量时未记录可公开引用的工作树 Git commit，
   只有二进制和关键源码哈希。后续发布的提交不能追溯性地视作该次运行的精确源码版本。
4. 使用同一 prompt、tokenizer、上下文和输出长度，在同一台机器上 A/B
   Mference Q2 或 ds4 Q2，才能量化精度与速度交换。

本报告支持的结论限定为：**这组配置在 24GB M5 Air 上完成了原生混合精度
权重的生成实验；最佳 128-token decode 为 2.13 token/s，三轮中位数为
1.57 token/s。** 长期性能、质量等价与相同条件下的外部实现比较仍需单独验证。
