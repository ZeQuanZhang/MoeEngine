# MoeEngine

MoeEngine 是面向 MoE 本地推理研究的专家权重驻留与缓存实验项目，供缓存策略、模型存储和推理后端开发者使用。它根据 Router 选中的专家、实际载入代价和访问历史，研究在有限内存中保留哪些权重，以及何时从存储重新读取。

项目提供 Python 参考实现与实验 CLI、接入 mollm 的 C++ 驻留优化器，以及 llama.cpp 的 GGUF 专家页实验补丁。Router 负责专家选择，MoeEngine 管理驻留，计算后端负责模型运算与文本生成。当前版本为研究原型。

## 可以用它做什么

- **比较缓存策略。** 在相同的随机、Zipf、Markov 或 bursty 访问轨迹上，对比 LRU、LFU、Generic、Router-GDSF 与离线 Belady，观察命中率、读取量和加载延迟。
- **检查专家存储。** 以 `(layer_id, expert_id)` 定位专家，读取打包文件、已适配的 safetensors 或分片 GGUF 中对应的权重范围。
- **验证缓存生命周期。** 通过按字节记账的缓存、使用期间的 pin 保护和带代际校验的 `ExpertLease`，检查载入、使用、释放与淘汰之间的关系。
- **接入真实推理。** 在已有计算后端内测试驻留策略，并记录固定模型、设备和运行参数下的吞吐与内存表现。

## 快速开始

准备 Git 和 Python 3.11 或更新版本。先运行一个无需模型权重的合成实验：

```bash
git clone https://github.com/ZeQuanZhang/MoeEngine.git
cd MoeEngine
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

moeengine benchmark \
  --pattern bursty --tokens 20 \
  --layers 4 --experts-per-layer 32 --top-k 2 \
  --expert-size-mib 1 --cache-gib 0.01 \
  --policies lru,lfu,generic,gdsf,belady
```

Windows PowerShell 可用 `py -3 -m venv .venv` 创建环境，用 `.venv\Scripts\Activate.ps1` 激活；安装命令相同，多行 CLI 示例需合并为一行执行。

成功运行后，终端会打印五种策略的结果行，每行的 `cache` 为 `10.24 MiB`。各列表示：

| 列 | 含义 |
|---|---|
| `policy` / `cache` | 策略名称与本次配置的缓存容量 |
| `hit rate` / `faults` | 命中比例与需要载入专家的次数 |
| `loaded` | 累计载入的专家字节数 |
| `p95 fault` | 由合成存储带宽和固定延迟计算的加载延迟 P95 |

这些指标用于比较策略在构造访问序列上的行为。真实存储吞吐与模型生成速度分别由文件读取实验和推理后端实验测量。命令完成后自动退出；修改参数即可再次运行，追加 `--json` 可获得机器可读结果。退出虚拟环境使用 `deactivate`。

### 运行检查

GGUF 读取和完整 Python 测试需要额外依赖：

```bash
python -m pip install -e ".[gguf]"
python -m unittest discover -s tests -v
```

CI 配置包含 Python 3.11、3.12、3.13 的测试任务，以及独立 C++ 优化器测试。独立 C++ 测试需要 CMake 3.14 或更新版本和 C++17 编译器：

```bash
cmake -S native -B build/native
cmake --build build/native
ctest --test-dir build/native --output-on-failure
```

## 选择实验路径

| 目标 | 输入与执行范围 | 主要输出 |
|---|---|---|
| 研究缓存与访问局部性 | 生成合成轨迹，配置专家大小和载入代价 | 策略命中、读取量、占用与模型化延迟 |
| 检查真实权重读取 | 对已适配模型中指定层的前 N 个专家重复读取 | 专家字节、校验值、缓存结果与存储吞吐 |
| 评估本地生成 | 准备兼容权重，构建对应推理后端，执行真实 Router 与计算图 | 特定协议下的生成吞吐、延迟和内存记录 |

合成回放可直接使用 Python 核心。真实文件读取需要本地 checkpoint，并满足适配器的张量布局要求：safetensors 和 GGUF 的专家定位分别由 [checkpoint 适配器](src/moeengine/checkpoint_store.py) 和 [GGUF 适配器](src/moeengine/gguf_store.py) 实现。更换模型时应核对张量命名、专家维度和量化元数据。

查看符合当前适配器字段约定的模型配置与 safetensors 索引：

```bash
moeengine inspect-config /path/to/model/config.json
moeengine inspect-index /path/to/model/model.safetensors.index.json
```

以下示例对指定层的前 16 个专家重复读取三轮：

```bash
moeengine benchmark-real /path/to/model \
  --layer 0 --working-set 16 --rounds 3 \
  --cache-mib 128,256 --policy lru
```

该构造序列用于检查文件寻址与缓存容量。safetensors 和打包文件路径使用 `os.pread`；macOS 上可追加 `--no-os-cache`，为源文件启用 `F_NOCACHE`。操作系统与设备缓存状态仍需通过单独的测量协议控制。模型权重由使用者另行准备。

## 已记录的实验结果

以下两组实验均使用 **24GB Apple M5 MacBook Air** 和 CPU 推理后端。各组的模型、缓存协议和计时口径分别列出，便于复核结果对应的条件。

### Qwen：87.24GiB GGUF 的专家页驻留

**2026-08-28 · Qwen3.8 Flash Next UD IQ4_XS · llama.cpp + Accelerate/BLAS · 10 线程。**

使用 mmap、零 GPU 层、无 warmup。每种配置在**同一进程中重复 3 次，每次 decode 32 tokens**，共 96 tokens，期间保留缓存历史；各进程在专家 bank 登记时执行一次页回收 reset，基线采用相同处理。

| 配置 | 专家页名义预算 | 96 tokens 累计逻辑命中率 | Decode 均值 ± 样本标准差 |
|---|---:|---:|---:|
| macOS mmap 基线，无 MoeEngine 淘汰 | 未设策略上限 | — | **2.26 ± 0.24 tok/s** |
| MoeEngine，24 slots/bank | 2.60 GiB | 66.05% | **0.83 ± 0.50 tok/s** |
| MoeEngine，96 slots/bank | 10.39 GiB | 81.43% | **2.24 ± 0.15 tok/s** |

96 slots 的平均吞吐接近系统基线，24 slots 出现反复读取。表中预算按 `55.432 GiB × slots / 512` 计算，描述专家页策略的名义容量。报告记录的受控 CPU 对照与生成检查区间内，系统 swapout 计数增量为 **0**，权重仍按需从存储读取。完整配置和专家范围读取校验见 [Qwen 真实推理报告](docs/qwen4exp-real-runtime-2026-08-28.md)。

### DeepSeek：156.7GB 原生 FP4/FP8 权重的重复生成

**2026-08-24 · DeepSeek-V4-Flash-0731 · mollm CPU · 4 计算线程、8 I/O workers。**

使用 `moeengine-adaptive`、4GiB 专家缓存、4096 context、batch 1、贪心解码、无 MTP。三轮各启动**独立进程，生成 128 tokens**；首 token 计入 TTFT，后 127 tokens 计入 decode。轮间分别静置 45 秒和约 180 秒，保留 macOS 文件页缓存。

| 运行 | Decode | 峰值 RSS |
|---|---:|---:|
| Run 1 | **2.13 tok/s** | 17.147 GB |
| Run 2 | 1.28 tok/s | 17.047 GB |
| Run 3 | 1.57 tok/s | 16.655 GB |
| 三轮中位数 | **1.57 tok/s** | **17.047 GB** |

该配置完成了三轮生成，吞吐存在明显波动。第二轮专家等待减少，计算算子耗时增加；报告保留了相应计时，温度或频率变化缺少直接遥测。完整协议见 [DeepSeek 重复 CPU 实验](docs/sota-report-2026-08-24.md)。

这两组结果以对应日期的报告存档；其原始运行日志未随仓库分发。仓库另附 [2026-08-25 实验归档](benchmark-results/README.md)，其中 [DeepSeek 后续实验目录](benchmark-results/dsv4-0731-m5air-0swap-2026-08-25/) 保存了 stdout、stderr 和部分系统计数器。该批资料对应后续实验，应按各自日期和配置使用。

## 接入推理后端

| 后端 | MoeEngine 接入点 | 当前策略或实验范围 |
|---|---|---|
| mollm / Palm-Infra | `MoeSsdCache` 的专家请求与驻留决策 | native 基线、LRU、LFU、adaptive、Generic、Router-GDSF |
| llama.cpp 实验补丁 | CPU `MUL_MAT_ID` 的文件映射专家 bank | frequency-biased LRU、页预取与回收建议 |
| Mference 实验源码 | 独立 Swift 推理后端 | 独立构建与实验记录，按该后端的模型格式和缓存协议运行 |

模型计算图、数值内核、分词和生成由各上游后端提供。缓存策略比较需要固定后端、权重、输入与运行参数。上述 Qwen GGUF 结果使用 llama.cpp 的文件页策略，Router-GDSF 的接入与验证见 [策略说明](docs/generic-optimizer.md)。

仓库包含 Palm-Infra 和 Mference 源码快照。llama.cpp 与早期 DeepSeek C 后端使用锁定上游版本及补丁准备：

```bash
# 准备锁定的 llama.cpp 源码和实验补丁
python3 tools/prepare_backends.py llama.cpp
```

该命令获取后端源码并校验补丁；构建与模型准备分别进行。mollm 的 CPU/Metal 构建、策略参数与模型包准备见 [接入说明](docs/mollm-integration.md)，llama.cpp 的 CPU 构建与运行协议见 [GGUF 实验报告](docs/qwen4exp-real-runtime-2026-08-28.md)，Mference 的独立入口见其 [README](vendor/Mference/README.md)。各原生路径有自己的工具链和平台要求。

## 设计与验证范围

`ExpertCache.acquire` 返回的 `ExpertLease` 会固定 Python 缓存槽位，消费者使用后释放；代际编号用于检查租约对应的槽位版本。Python 参考实现同步执行载入，容量记账在新对象载入、旧对象淘汰之后收敛，过渡期间可能同时持有新旧权重。

缓存计数、专家槽位和进程内存分别描述不同范围。原生推理还需要 dense 权重、KV 状态、工作区和读取缓冲；Apple Silicon 上的应用缓冲与系统文件页共享统一内存。`MOEENGINE_EXPERT_CACHE_SLOTS` 表示每层、每个 projection bank 的专家数量，文件页回收通过操作系统建议实现，实际驻留页与进程 RSS 受操作系统及其他分配共同影响。

`generic` 与 `gdsf` 是在线启发式，Belady 使用完整未来访问序列作为离线参照。当前实验覆盖合成回放、真实专家文件读取和特定配置下的短文本生成；全模型 logit、困惑度与任务质量对齐，以及长上下文、并发负载和跨平台真实模型表现仍需各自验证。

## 文档与许可

| 文档 | 内容 |
|---|---|
| [架构与实现范围](docs/architecture.md) | 专家地址、租约、存储与内存记账 |
| [Generic / Router-GDSF 策略](docs/generic-optimizer.md) | 驻留评分、路由复用与原生接口 |
| [mollm 接入与构建](docs/mollm-integration.md) | CPU/Metal 构建、策略配置与模型包准备 |
| [按日期记录的实验](docs/benchmarks.md) | 合成、存储和原生推理的历史测量 |
| [早期架构提案](docs/codex-spec.md) | 项目初期的设计与研究计划 |

第一方代码采用 [MIT License](LICENSE)。第三方源码与补丁保留各自许可和版权说明，固定版本与来源记录见 [第三方说明](THIRD_PARTY_NOTICES.md)。模型权重按其发布方的许可使用。
