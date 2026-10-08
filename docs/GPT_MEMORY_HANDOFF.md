# GPT MEMORY - MCNPU / Ornith-1.5-9B-MLX-4bit

> 【GPT 的记忆开头】
>
> 这是 GPT 为本仓库下一次继续开发准备的长期接班记忆文件。
> 阅读本文件后，必须先恢复这里记录的目标、边界、已完成工作和下一步，不要根据旧聊天里的模糊印象重新猜模型格式。
> **本文件开头这个标记固定保留：这是 GPT 的记忆。**
>
> **当前唯一主目标是 Android 手机上的 Ornith-1.5-9B-MLX-4bit/model.safetensors，约 4.69 GB；不是 GGUF。GGUF/llama.cpp 代码只作为旧基线和参考保留。**

---

# 0. 接班时先看这里

## 当前目标

把 MCNPU 做成 Android 本地推理运行时，专门适配：

- 模型：Ornith-1.5-9B-MLX-4bit
- 文件：model.safetensors
- 格式：Safetensors
- 量化：MLX 4-bit
- 平台：Android
- 加速目标：Qualcomm QNN / HTP / NPU
- 硬约束：运行时 RAM <= 4 GiB
- 当前不是通用模型加载器项目。
- 当前只把这个模型完整跑通；未来其他模型必须单独增加适配。

## 最终目标

完成真实的：
1. Safetensors 文件解析；
2. MLX 4-bit 权重解码；
3. Ornith/Qwen3.5 模型拓扑执行；
4. Android mmap/按需读取；
5. CPU fallback；
6. QNN/HTP/NPU 加速的矩阵运算路径；
7. tokenizer / prompt / prefill / decode；
8. KV 与 24 个线性注意力/GDN 层状态管理；
9. 4 GiB RAM 峰值控制；
10. 真实文本生成；
11. 完整诊断、benchmark、回归测试。

**绝不能用假输出、固定字符串、随机输出或 probe 成功冒充模型已经能推理。**

# 1. 模型事实，绝对不要改错

目标文件：
Ornith-1.5-9B-MLX-4bit/model.safetensors

约 4.69 GB。

配置基线：
- 32 decoder layers
- hidden size = 4096
- vocab size = 248320
- native context = 262144
- 8 个 full-attention layers
- 24 个 linear-attention / GDN layers
- full attention 使用 4 KV heads
- head dimension = 256
- 4-bit MLX 量化
- group size = 64
- 量化结构涉及 packed weights / scales / biases

## 重要格式区别

**MLX 4-bit 绝对不能当成 GGUF Q4_K_M。**

禁止：
- 用 GGUF header parser 读取 Safetensors；
- 把 Safetensors tensor offsets 当 GGUF tensor table；
- 直接调用 llama.cpp GGUF loader；
- 把 MLX 4-bit packed weight 当成 GGUF Q4_K_M block；
- 没有真实 metadata 就猜量化布局。

# 2. Android / 内存硬约束

**运行时 RAM 必须控制在 4 GiB 以内。**

正确方向：
- Safetensors 使用 mmap / RandomAccessFile + page-wise 访问；
- 避免完整 weight copy；
- 一次只准备当前 layer / 当前算子的工作集；
- CPU staging buffer 重用；
- QNN/NPU tensor buffer 重用；
- KV / recurrent state 单独预算；
- 记录 RSS/PSS；
- 记录 native heap、Java heap、mapped resident pages；
- 记录峰值。

禁止一次性创建装下整个 4.69GB 文件的 byte[] 或 direct buffer。

# 3. 当前仓库状态

仓库：
https://github.com/bslsjdk/mcnpu

当前 main HEAD：
eb14eba9dc4d13c82837d9821531b6fc184d18f5

当前分支：
- main：主开发线
- bonsai2-pq2-baseline：旧 Bonsai/PQ2 baseline

当前没有发现已经完成 MLX/Safetensors runtime 的独立分支。

# 4. 已完成的 MLX 修改

## 4.1 Ornith15MlxProbe.java

路径：
app/src/main/java/bslsjdk/mcnpu/Ornith15MlxProbe.java

作用：严格的 Safetensors metadata-only probe。

功能：
- 读取 Safetensors header length；
- 限制 header 最大大小；
- 解析 JSON header；
- 枚举 tensor；
- 检查 dtype / shape / data_offsets；
- 检查 offsets 不越界；
- 检查 layer 0～31；
- 检查 embedding / lm_head 维度；
- 检查 packed U32 weight；
- 检查 scales；
- 检查 biases；
- 输出文件和 data 区域统计。

它不会读取整个 4.69GB 权重数据。

成功前缀：OK ORNITH15_MLX_PROBE/1。

注意：probe 只是格式/结构验证，不是推理器。

# 5. Ornith15Runtime.java

路径：
app/src/main/java/bslsjdk/mcnpu/Ornith15Runtime.java

已经修改：
- .safetensors 进入 Ornith15MlxProbe；
- 不会把 Safetensors 交给旧 llama.cpp/GGUF loader；
- probe 失败直接返回错误；
- probe 成功明确标记 MLX 已验证但 executor 尚未连接；
- generate/执行路径不会把验证成功冒充模型已加载；
- reload 时清掉旧 MLX 状态。

# 6. GGUF / llama.cpp 的正确定位

旧 baseline/reference 文件：
- app/src/main/cpp/ornith_runtime.cpp
- app/src/main/cpp/ornith_kv.cpp
- app/src/main/java/bslsjdk/mcnpu/Ornith15Probe.java
- app/src/main/java/bslsjdk/mcnpu/Ornith15MemoryPlanner.java

这些可参考 runtime 生命周期、KV 规划、内存测量、Android 集成。

**不要把 GGUF parser/quant decoder 直接套到 MLX Safetensors。**

建议后续使用独立命名：
- ornith15_safetensors.cpp
- ornith15_mlx_quant.cpp
- ornith15_mlx.cpp
- Ornith15MlxRuntime.java
- Ornith15MlxMemoryPlanner.java

# 7. QNN / HTP / NPU 基础设施

核心 native 文件：
app/src/main/cpp/mcnpu.cpp

主要功能：
- 加载 QNN；
- 查找 HTP provider；
- backendId = 6；
- backend/device/context；
- QNN graph；
- ADD；
- MATMUL；
- MATMUL16；
- MATMUL8；
- binary transport；
- graph cache；
- shape bucket；
- diagnostics；
- runtime locking；
- CPU affinity；
- QNN/RPC diagnostics。

已经确认的环境能力：QNN HTP backend 可用，backendId=6，HTP_V73，persistent service 和 binary IPC 已建立。

# 8. HTP shape bucket 经验

HTP 不是任意 M/K/N 都稳定支持。M、K、N 都需要落入实测支持 bucket。

之前稳定测试过 32³、64³、128³、256³、512³、1024³，典型 NPU execution 约 1.7～2.4 ms，bad=0。16³ 曾出现 scale mismatch。

新的 MLX matmul dispatcher 必须：
- 维护支持 bucket；
- 对 M/K/N 做 padding 或分块；
- 不假设任意 shape；
- 记录 padding；
- 输出裁剪回原 shape；
- 处理 scale；
- graph cache；
- 限制 graph 总量。

# 9. NPU Service / IPC

主要文件：
- NpuRuntime.java
- NpuService.java
- NpuServiceClient.java
- NpuKeepAlive.java
- NpuWakeReceiver.java
- NpuBinFile.java
- NpuBigAdd.java

核心原则：MCNPU service 持有 QNN/HTP 生命周期。

不要每个 token 都重新加载 QNN、创建 backend 或销毁 graph。

模型 runtime 应保持稳定的 binary tensor transport；JSON 只用于 diagnostics/debug。

# 10. CMake

路径：app/src/main/cpp/CMakeLists.txt

当前：
- native library = mcnpu
- C++17
- QNN include 必须存在；
- llama.cpp 是可选 baseline；
- build 会记录 MCNPU_BUILD_ID；
- section GC / visibility 优化。

后续加入独立 MLX native source，但不要删除 GGUF baseline。

# 11. Java/UI/Android 层

主要文件：
- MainActivity.java
- ChatActivity.java
- ShizukuHelper.java
- activity_main.xml
- activity_chat.xml
- AndroidManifest.xml

建议最终链路：
Activity -> Ornith15MlxRuntime -> native MLX adapter -> NpuServiceClient -> QNN

# 12. Ornith 长上下文内存基线

相关文档：
- docs/ORNITH15_MEMORY_BASELINE.md
- docs/ORNITH15_MEMORY_POLICY.md

8 个 full-attention layer 的 FP16 K+V baseline：32768 bytes/token。

所以：
- 64K FP16 KV ≈ 2 GiB
- 128K FP16 KV ≈ 4 GiB
- 262K FP16 KV ≈ 8 GiB

4 GiB RAM 下不能用普通 FP16 KV 直接做 262K。

24 个 GDN/linear-attention layers 使用固定 recurrent state，是长上下文的重要性质。

# 13. 长上下文验收

Gate A：64K 真 context、无静默截断、RAM <= 4 GiB、质量和速度记录。

Gate B：128K 同样要求。

Gate C：262144 只有前两阶段稳定后再做。

最终需要更紧凑的 KV/state 表示或其他经过测量的方案。

# 14. 下一步：真实 MLX 适配

## Step 1：取得真实 Safetensors metadata

必须拿目标 model.safetensors 的真实 header，记录 tensor names、dtype、shape、offsets、metadata、quantization companion tensors 和 layout。

如果真实 metadata 与当前 probe 假设不同，修改 probe；不要修改现实来迁就代码。

## Step 2：Safetensors reader

建议：app/src/main/cpp/ornith15_safetensors.cpp

功能：open、header、metadata、tensor lookup、bounds、dtype、shape、zero-copy view、page-aligned read、线程安全。

## Step 3：MLX 4-bit decoder

建议：app/src/main/cpp/ornith15_mlx_quant.cpp

必须从真实 tensor layout 确认 packed word 排列、group size、scale/bias dtype、group 维度、weight layout、transpose convention、dequant 数学、permutation。

## Step 4：CPU golden path

先不接 NPU。实现 packed MLX weight + scales + biases -> FP16/FP32 matrix，并用小 tensor 做 known-value、random、edge、zero、max/min、scale、bias 测试。

只有 CPU decoder 可信后才允许接 NPU。

# 15. NPU 加速路线

不要一上来把全部 4-bit 权重解码成完整 FP16 模型。

优先研究：
packed 4bit -> chunk dequant / quant -> HTP MATMUL

第一阶段可以接受小块 FP16/INT8 staging，但 staging buffer 必须有固定上限并复用。

# 16. Layer streaming

推荐：
Safetensors mmap -> current layer tensor view -> MLX 4-bit decode -> reusable staging buffer -> QNN HTP graph -> output buffer -> next layer

禁止：整个模型先 decode 到 RAM。

# 17. NPU graph cache

建议 key：operation + M + K + N + dtype + quant mode。

必须有 bounded cache、固定 bucket、eviction、context lifetime 管理、graph creation 统计、cache hit/miss 和 memory budget。

# 18. Prefill / Decode

Prefill：token 多、矩阵大、适合 NPU、适合批量。

Decode：通常 batch 小、每次 token 少，不能为了 NPU 强行制造巨大延迟。

最终需要独立 prefill dispatcher 和 decode dispatcher。

# 19. RAM 预算

必须实时记录：Java heap、native heap、mapped RSS、model resident pages、QNN buffers、staging buffers、KV、recurrent state、tokenizer、temporary allocations、peak RSS、peak PSS。

目标是 peak RSS < 4 GiB，并保留安全余量。

# 20. Tokenizer

必须确认目标模型自己的 tokenizer 文件和 chat template。

必须支持 tokenizer、BOS/EOS、special tokens、chat template、prompt formatting、stop conditions、sampling。

不要因为 GGUF 有 tokenizer metadata 就假定 MLX 模型完全一样。

# 21. Sampling

第一阶段支持 greedy、temperature、top-p、top-k、repetition penalty。

sampling 不应成为 NPU 主任务。

# 22. 诊断

每次测试必须能打印：
- MCNPU_BUILD_ID
- MODEL=Ornith-1.5-9B-MLX-4bit
- FORMAT=SAFETENSORS
- QUANT=MLX_4BIT
- CONTEXT
- RSS/PSS
- backendId=6
- HTP_V73
- graph cache hit/miss
- M/K/N
- prefill_ms / decode_ms
- bad
- fallback
- staging_bytes
- peak_rss

否则后续无法可靠判断 APK 到底是哪一版。

# 23. 必须保留的 NPU 底座

不要因为目标从 GGUF 转 MLX 就删除：
- mcnpu.cpp
- QNN service
- binary IPC
- HTP diagnostics
- graph bucket/cache
- NPU keepalive
- Shizuku path
- NPU shape tests
- Bonsai/PQ2 baseline。

# 24. 禁止再次出现的错误

1. 把 model.safetensors 当 GGUF。
2. 把 MLX 4-bit 当 Q4_K_M。
3. 把 probe OK 当 inference OK。
4. 把 4.69GB 模型整体解码进 RAM。
5. QNN MATMUL 能跑就宣称整个模型已经 NPU 加速。
6. 没有真实 metadata 就猜 decoder。
7. 只测 Java heap，不测 native RSS/PSS。
8. 当前阶段为了通用化同时适配几十个模型。

# 25. 借鉴、依赖和相关仓库

## llama.cpp
https://github.com/ggml-org/llama.cpp
用途：当前旧 Ornith GGUF runtime 的基础和参考，包括 GGUF runtime、KV、Android integration。它不是当前 MLX Safetensors 主格式。

## MCNPU
https://github.com/bslsjdk/mcnpu
本项目自己的主仓库。

## Minecraft/Fabric NPU 客户端
https://github.com/bslsjdk/mcjavanpu
MCNPU service 的 Minecraft/Fabric IPC 使用方，不是模型 loader。

## Qualcomm QNN / QAIRT
QNN 是 Snapdragon HTP/NPU 的外部 SDK/API 依赖。当前仓库使用其 headers/libraries，不应把未确认来源的 Qualcomm SDK 源码说成已经复制进仓库。

**注意：除上述明确记录的来源外，不要在以后凭印象声称某段代码“抄自某仓库”。如果没有 commit、文件或明确来源证据，就写成“参考/依赖”，不要写成“复制”。**

# 26. 关键历史提交

Ornith GGUF baseline：
- fbfe731：Ornith 1.5 GGUF memory probe
- 55b85a：long-context memory planner
- a94d561：page codec 对齐
- c0d3a28：KV estimates / validation
- fff862：Q8Q4 memory estimate
- 2987f68：pinned llama.cpp Ornith backend
- df84dd0：local Ornith generation baseline
- dcc0c59：Ornith runtime boundary
- b672bce：Ornith runtime bridge
- ab317c6：KV codec size_t 修复
- 7329e3c：Java escaped newline 修复
- 313b718：GGUF string terminator 修复
- a3a1681：llama.cpp model params 修复

这些历史提交只用于理解 GGUF baseline 的来龙去脉，不得因此把 GGUF 当当前目标。

# 27. 本轮实际落库

新增：
app/src/main/java/bslsjdk/mcnpu/Ornith15MlxProbe.java

修改：
app/src/main/java/bslsjdk/mcnpu/Ornith15Runtime.java

目的：
- Safetensors 与 GGUF 分流；
- Safetensors metadata validation；
- 防止误调用 GGUF executor；
- 明确 executor 尚未连接；
- reload 状态清理。

修改后 GitHub Actions 已自动启动 Android build。

# 28. 下一次 GPT 接手时的严格执行顺序

不要重新讨论项目方向。

1. 确认最新 GitHub Actions build。
2. 检查 Ornith15MlxProbe 与真实 model.safetensors metadata。
3. 获取真实 tensor layout。
4. 实现 ornith15_safetensors.cpp。
5. 实现 ornith15_mlx_quant.cpp。
6. 建立 CPU golden decoder。
7. 小 tensor 逐元素验证。
8. 接入 Ornith15MlxRuntime。
9. 实现 layer streaming。
10. 接 QNN HTP MATMUL。
11. 实现 shape bucket + graph cache。
12. 实现 prefill。
13. 实现 decode。
14. 实现 KV / GDN recurrent state。
15. 接 tokenizer/chat template。
16. 4GB RAM 压力测试。
17. 64K gate。
18. 128K gate。
19. 最后攻 262144 context。

# 29. 最终架构目标

Android Chat/UI
  -> Ornith15MlxRuntime
  -> native MLX adapter
  -> Safetensors Reader
  -> MLX 4-bit Weight View
  -> Layer Streaming
  -> CPU fallback / QNN HTP Dispatcher
  -> Shape Bucket
  -> Graph Cache
  -> Snapdragon HTP/NPU

内存控制横跨 Model mmap、KV/recurrent state、NPU buffers、CPU staging、Tokenizer 和 runtime objects，由 Memory Planner 统一预算。

# 30. 最终成功标准

- [ ] 直接读取真实 model.safetensors
- [ ] 不依赖转换成 GGUF
- [ ] MLX 4-bit decoder 正确
- [ ] CPU golden test 通过
- [ ] Android mmap/streaming 正常
- [ ] 不整包复制 4.69GB
- [ ] 运行时 RAM <= 4 GiB
- [ ] QNN HTP 初始化成功
- [ ] 关键 GEMM/MATMUL 真正在 HTP 执行
- [ ] CPU fallback
- [ ] prefill 正确
- [ ] decode 正确
- [ ] KV/state 正确
- [ ] tokenizer 正确
- [ ] chat template 正确
- [ ] 真实生成模型输出
- [ ] 不是假输出
- [ ] 64K 通过
- [ ] 128K 通过
- [ ] 262144 最终目标完成或有明确、可测量的剩余瓶颈
- [ ] 完整诊断日志
- [ ] build ID 可追踪
- [ ] GGUF baseline 仍可回归，但不影响 MLX 主路径

# 31. 最重要的接班原则

> **先验证真实格式，再写 decoder；先 CPU 正确，再接 NPU；先控制内存，再追求速度；先完成单模型，再谈多模型。**

当前主线只有一个：

Ornith-1.5-9B-MLX-4bit/model.safetensors -> Android -> MLX 4-bit -> QNN HTP/NPU -> <=4 GiB RAM -> 真实生成

这就是下一次 GPT 接手后必须继续完成的主任务。

---
*文件性质：GPT 长期接班记忆 / 项目状态快照*
*最后更新：2026-10-08*