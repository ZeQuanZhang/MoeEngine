# DeepSeek-V4-Flash-0731：M5 Air 24GB / 0-swap 测量记录

> 历史测量档案：项目标识与本地路径已按 MoeEngine 发布名称规范化；此次整理未重新运行测量。参见[档案说明](../README.md)。

测试日期：2026-08-25（Asia/Shanghai）

## 结论

本轮记录了两类观察：

1. 所有成功取得 `/usr/bin/time -l` 页面的运行记录中，
   进程级 `swaps` 字段都是 0；专家权重由 Router 选择后，
   从 `.mollm` 文件显式读取到应用管理的 RAM cache。
2. 在当前已经存在历史 swap 的桌面会话中，不能证明“整台 Mac 在推理期间
   完全没有任何 swap I/O”。最低内存配置的一轮做到了零新增 swapout，复跑
   仍出现 76 页（约 1.19 MiB）的系统后台换出。因此不能把本轮宣传为可重复的
   **whole-system strict 0-swap**。

本报告测量的条件为 `Mac17,3`、M5 Air 24GB、原生 FP4/FP8
DeepSeek-V4-Flash-0731、native top-6、无 MTP、SSD experts。公开对比资料
使用的硬件、量化与协议不同，下面将这些差异与测量结果一起列出，
不据此推断跨设备或跨运行时的性能排名。

## 系统与模型

| 项目 | 值 |
| --- | --- |
| 设备 | MacBook Air `Mac17,3` |
| SoC | Apple M5，10 CPU cores（4P + 6E） |
| 内存 | 24 GB unified memory |
| 系统 | macOS 26.5.2，build `25F84` |
| 电源 | AC，电池 100% |
| 模型 | 官方 `DeepSeek-V4-Flash-0731` |
| 本地模型 | `deepseek_v4_flash_native.mollm` |
| 文件大小 | 156,712,820,637 bytes |
| 精度 | 原生 MXFP4 routed experts + FP8 dense/core |
| 结构 | 43 layers，256 routed experts，native top-6 |
| 二进制 SHA-256 | `25e2c0916f4e816e0d1e73e89965b0604d7596d233b62af65d5e86fda29c1297` |

## 固定测试口径

- prompt：`给我讲一个故事`（8 tokens）
- 输出：128 tokens；127 个 decode tokens
- greedy：temperature 0
- context：4096
- CPU threads：4
- SSD I/O workers：8
- replacement：`moeengine-adaptive`
- MTP：关闭
- 每轮全新进程，`caffeinate -i`
- 每轮前后记录 `vm.swapusage`、`vm_stat`；进程用 `/usr/bin/time -l`
- 不清理 macOS 文件页缓存

这里把两种“0 swap”分开：

- **进程级**：`/usr/bin/time -l` 的 `swaps=0`。
- **整机严格级**：本轮前后 `Swapouts` 增量必须精确为 0，并且最好从
  `vm.swapusage used=0` 的干净启动开始。本次会话开始时已经有 906.25 MiB
  历史 swap，因此只能测增量，不能证明绝对零。

## 结果

| Run | CPU sidecar | Expert cache | Decode | TPOT | Peak footprint | 进程 swaps | Swapout 增量 | swap used 变化 | 严格零新增换出 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | default 9.36 GB | 4 GiB | 1.41 tok/s | 710.20 ms | 未取得 | 未取得 | +65,432 pages | +966.37 MiB | 否 |
| 2 | default 9.36 GB | 2 GiB | 1.30 tok/s | 769.30 ms | 15.58 GB | 0 | +41,772 pages | +620.69 MiB | 否 |
| 3 | compact 7.92 GB | 1 GiB | 0.99 tok/s | 1008.91 ms | 13.18 GB | 0 | **0 pages** | -24.00 MiB | 是，单轮 |
| 4 | default 9.36 GB | 1 GiB | **1.13 tok/s** | 885.42 ms | 14.62 GB | 0 | **0 pages** | -8.00 MiB | 是，单轮最快 |
| 5 | compact 7.92 GB | 2 GiB | 0.94 tok/s | 1059.52 ms | 14.13 GB | 0 | **0 pages** | -8.00 MiB | 是，单轮 |
| 6 | default 9.36 GB | 1 GiB | 1.26 tok/s | 795.25 ms | 14.52 GB | 0 | +1,116 pages | +9.44 MiB | 否，复跑 |
| 7 | compact 7.92 GB | 1 GiB | **1.47 tok/s** | 682.25 ms | 13.18 GB | 0 | +76 pages | -54.81 MiB | 否，复跑接近零 |

从当前数据可作出的最强但不夸大的表述：

- **精确零新增 swapout 的单轮最高值：1.13 tok/s。**
- **最低内存配置的冷却复跑：1.47 tok/s，进程 swaps=0，但整机新增约
  1.19 MiB swapout，不能写成数学意义的 0。**
- 默认 4 GiB/2 GiB 高速档会明显迫使 macOS 把其他匿名页换出；它们不能用于
  严格整机 0-swap 声明。
- 先前 2.13 tok/s 的 4 GiB 最佳 128-token 结果仍是本机原生精度性能候选，
  但当时没有做系统级前后 swapout 审计，不能回填成严格 0-swap 纪录。

## 输出一致性

七轮生成文本完全一致，提取出的 128-token continuation SHA-256 均为：

`10813773b72df315a9c05dcd778c44e5e9a3796570fb4b4de5cf221e6ec003bb`

这证明 cache 大小和 exact compact sidecar 没有改变本轮 greedy 输出。它不替代
全模型 logits、困惑度或任务集对齐。

## 公开对比边界（检索于 2026-08-25）

| 项目 | 公开结果 | 与本轮实验的条件差异 |
| --- | ---: | --- |
| DeepSeekV4SSD | M5 Pro 64GB，约 6.41--6.5 tok/s | 更强 SoC、主动散热、64GB；未披露本口径系统 swap delta |
| dsv4-streaming full-quality | M5 Pro 64GB，4--8GB cache 中位数约 2.00--2.04 tok/s | 64GB M5 Pro；24-token protocol，不是 128-token Air 复测 |
| Anemll Flash-MoE | M5 Max，native top-6，短 prompt 约 5.5 tok/s | M5 Max 与更大 slot bank；不同硬件等级 |
| Mference DSV4 | M3 Ultra 256GB，2-bit DQ 约 5.3--6.1 tok/s | 2-bit checkpoint、256GB page cache、不同硬件 |

来源：

- 官方模型：<https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731>
- DeepSeekV4SSD：<https://github.com/yanun0323/deepseek_ssd>
- dsv4-streaming：<https://github.com/kk-r/dsv4-streaming>
- Anemll Flash-MoE：<https://github.com/Anemll/anemll-flash-llama.cpp/blob/master/docs/flashmoe-layer-major-dedup-single-sidecar.md>
- Mference：<https://github.com/NeelM0906/Mference>

## 严格 0-swap 复测还需要哪些条件

当前桌面会话已有历史 swap，且前两轮高内存测试已经把后台应用页面换出；后续
配置不是从相同的干净系统状态开始，因此只看增量会受运行顺序影响。严格方案是：

1. 重启，确认 `vm.swapusage used=0`、`Swapins=0`、`Swapouts=0` 或记录干净基线；
2. 不启动额外应用，每个候选配置在独立重启后作为第一轮运行；
3. 先验证 compact + 1 GiB，再验证 default + 1 GiB；
4. 每个配置至少三次独立冷启动，报告中位数，不采用单个冷机峰值；
5. 同时保存进程 `swaps=0` 和整机 `Swapouts delta=0` 两份证据。

在完成这个 clean-boot protocol 前，结论限定为：**已记录运行的进程级 swaps
字段均为零。在 M5 Air 24GB + 156.7GB 原生 DSV4-0731 测试中，有单轮实现
零新增系统换出，复跑仍出现少量新增换出；尚未证明可重复的整机严格 0-swap。**
