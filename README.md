# MoeEngine

面向混合专家模型（MoE）的专家权重驻留与缓存实验。MoeEngine 使用模型 Router
给出的专家选择、实际加载代价与访问历史，研究有限内存下的专家读取、保留和淘汰。

这是**研究原型**。Python 参考实现用于验证缓存与策略；原生实验把策略接入已有推理后端。
Router 决定调用哪些专家，MoeEngine 管理专家权重的驻留，计算后端执行模型运算。
项目尚不提供生产推理服务，也未完成全模型质量等价、长上下文与并发负载验证。

## 已实现的内容

- 以 `(layer_id, expert_id)` 定位专家，映射到文件偏移、长度和可选校验值。
- 支持合成存储、打包文件 `pread`、safetensors 专家切片，以及已验证布局的分片 GGUF。
- 按字节记账的缓存、使用期间的 pin 保护，以及带 generation 校验的 `ExpertLease`。
- LRU、LFU、Generic 和 Router-GDSF 在线策略；Belady 仅用于离线对照。
- 随机、Zipf、Markov、bursty 合成轨迹，命中率、读取字节、缓存占用与加载延迟统计。
- C++ 驻留优化器及 mollm 接入；另有 llama.cpp 文件页驻留实验。

### 三类实验，各自回答不同问题

| 路径 | 输入与机制 | 可以观察的结果 |
|---|---|---|
| Python 合成回放 | 生成访问序列，使用配置的加载代价 | 策略行为与模型化成本；不代表真实推理速度 |
| Python 真实文件读取 | 对指定层的前 N 个专家重复读取 | 专家切片、缓存命中与存储吞吐；不执行文本生成 |
| 原生推理接入 | 后端实际 Router 与计算图 | 特定模型、设备、参数和运行协议下的端到端结果 |

## 快速开始

需要 Python 3.11 或更新版本。完整 Python 测试包含 GGUF 测试，因此先安装 `gguf` extra：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[gguf]"
python -m unittest discover -s tests -v
```

GitHub CI 配置覆盖 Python 3.11、3.12、3.13 以及独立 C++ 优化器测试；不包含模型下载或完整推理基准。

Windows PowerShell 的激活命令为 `.venv\Scripts\Activate.ps1`。真实文件读取使用
`os.pread`，原生后端和 macOS 文件缓存选项有各自的平台要求；它们不属于跨平台验证承诺。

安装后可直接运行合成实验，无需下载模型：

```bash
moeengine benchmark \
  --pattern bursty --tokens 200 \
  --layers 43 --experts-per-layer 256 --top-k 6 \
  --expert-size-mib 12 --cache-gib 2,4,8 \
  --policies lru,lfu,generic,gdsf,belady
```

### 仅体验标准库核心

Python 核心不要求第三方运行依赖。未安装 extra 时，可在仓库根目录用以下命令体验
合成回放；这不包含完整测试套件和 GGUF 读取：

```bash
PYTHONPATH=src python3 -m moeengine benchmark \
  --pattern bursty --tokens 20 \
  --layers 4 --experts-per-layer 32 --top-k 2 \
  --expert-size-mib 1 --cache-gib 0.01 \
  --policies lru,lfu,generic,gdsf,belady
```

## 检查模型与读取专家

只读取模型配置，不加载权重：

```bash
moeengine inspect-config /path/to/model/config.json
moeengine inspect-index /path/to/model/model.safetensors.index.json
```

以下示例需要自行准备兼容的 checkpoint。`--working-set 16 --rounds 3` 对指定层的
前 16 个专家重复读取三轮，不模拟真实 Router 的选择：

```bash
moeengine benchmark-real /path/to/model \
  --layer 0 --working-set 16 --rounds 3 \
  --cache-mib 128,256 --policy lru
```

macOS 可追加 `--no-os-cache`，对源文件启用 `F_NOCACHE`。该选项不能证明 SSD
硬件缓存为冷态，也不能将文件读取吞吐换算成模型生成速度。模型权重不随仓库发布；模型使用
条件由其发布方决定。

## 原生后端与策略

| 后端 | MoeEngine 接入点 | 策略边界 |
|---|---|---|
| mollm / Palm-Infra | `MoeSsdCache` 的专家请求与驻留决策 | native 基线、LRU、LFU、adaptive、Generic、Router-GDSF |
| llama.cpp 实验补丁 | CPU `MUL_MAT_ID` 的文件映射专家 bank | frequency-biased LRU 与页预取/回收建议；不使用 Router-GDSF |
| Mference 实验源码 | 独立 Swift 推理后端 | 独立构建与实验记录；不等同于上述 C++ 策略接入 |

模型计算图、数值内核、分词和生成由对应上游后端提供。MoeEngine 的缓存或策略结果需要在
同一后端、权重、输入与运行参数下比较；不同后端的吞吐差异不能归因于缓存策略。

仓库保留 Palm-Infra 与 Mference 的源码快照。llama.cpp 和早期 C 后端通过固定上游版本
及补丁准备；脚本只下载后端源码，不下载模型权重：

```bash
# 可只准备 llama.cpp；省略名称则准备所有补丁后端
python3 tools/prepare_backends.py llama.cpp
```

上游版本与改动来源见 [第三方说明](THIRD_PARTY_NOTICES.md)。构建独立 C++ 优化器测试
不需要模型权重：

```bash
cmake -S native -B build/native
cmake --build build/native
ctest --test-dir build/native --output-on-failure
```

完整 mollm 构建与策略参数见 [mollm 接入说明](docs/mollm-integration.md)。
llama.cpp 的 GGUF 实验协议见 [真实推理实验](docs/qwen4exp-real-runtime-2026-08-28.md)。

## 如何理解内存与性能结果

- **逻辑容量不是进程或系统内存硬上限。** Python 参考缓存先加载新对象再淘汰旧对象；
  临时峰值可能超过已记账容量。原生路径还包含 dense 权重、KV、工作区、文件页和读取缓冲。
- llama.cpp 的 `MOEENGINE_EXPERT_CACHE_SLOTS` 是每层、每个 projection bank 的专家数量，
  不是字节数。它对干净的文件映射页给出回收建议，物理回收仍由操作系统决定。
- `generic` 与 `gdsf` 是在线启发式。`belady` 需要完整未来访问序列，只适合作为离线参照。
- 历史数据分别包含合成回放、文件读取、短文本生成和重复推理。比较前应对齐模型精度、缓存
  预算、输出长度、缓存冷热、线程、运行顺序及散热状态；短跑峰值不代表长期吞吐。
- 已有输出一致性和内核测试不替代全模型 logit、困惑度或任务质量评测；当前数据不支持
  跨模型通用加速、硬物理内存上限或领先性能的结论。

## 文档

- [架构与实现边界](docs/architecture.md)
- [Generic / Router-GDSF 策略](docs/generic-optimizer.md)
- [mollm 原生接入与构建](docs/mollm-integration.md)
- [按日期记录的实验结果](docs/benchmarks.md)
- [DeepSeek 重复 CPU 实验审查](docs/sota-report-2026-08-24.md)
- [GGUF 专家页驻留实验](docs/qwen4exp-real-runtime-2026-08-28.md)
- [早期架构提案（历史文档）](docs/codex-spec.md)

第一方代码采用 [MIT License](LICENSE)。第三方源码与补丁所涉及的上游代码保留各自许可，
模型权重另按模型发布方的许可使用。
