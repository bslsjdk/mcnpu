
# GPT-5.6 Luna：MCNPU 代码审查与 NPU 优化路线报告

作者：GPT-5.6 Luna
日期：2026-10-03
审查对象：bslsjdk/mcnpu 当前 main 分支
性质：代码仓库级架构审查 / 优化建议

> 本报告基于当前仓库实际代码结构、QNN/HTP 调用方式以及已经暴露出的实测约束。实现时必须以设备实测为最终依据。

## 0. 核心结论

当前项目已经有一个正确的基础：Android Foreground Service 持有 QNN/HTP V73，并通过 127.0.0.1:38761 给 Minecraft 侧提供 NPU RPC。

但目前本质上仍是“NPU RPC 服务 + MatMul 实验场”，还不是完整的 Minecraft 世界生成加速器。

目标应该明确为：

“尽可能保持 Minecraft 26.3 原版世界生成结果，同时把原版 CPU 重计算尽可能搬到 NPU。”

推荐最终数据流：

Minecraft Chunk 请求
→ 9×9 工作集调度
→ 跨 Chunk 数学特征收集
→ HTP m/k/n Shape Planner
→ 批量 NPU Job
→ 结果 Buffer
→ 轻量 CPU Chunk 组装
→ 普通 Minecraft Chunk

## 1. 当前代码最重要的问题

### 1.1 Java 看起来并行，native 实际被全局锁串行

NpuService 使用 Executors.newFixedThreadPool(8)，但 native 使用全局 gRuntimeMutex。

runMatMulInt8Buf、runBatchXform 等路径都会进入这把锁。

因此当前实际上是：

8 个 Java worker
→ 1 个 native 全局锁
→ 一次只进入一个 QNN 操作

不要继续扩大 Java worker 数量来“提高 NPU 并行”。

推荐改成：

IPC 接入
→ NPU Job Queue
→ 一个长期存在的 NPU worker
→ QNN Context

然后由这个 worker 做 batch、shape 合并、优先级和 buffer 复用。

### 1.2 当前每个 request 都新建 Socket

NpuServiceClient.request() 每次都会创建、connect、读写并关闭 Socket。

对于测试没问题，对于 Chunk 世界生成非常不划算。

如果一个 Chunk 被拆成几十甚至几百个 NPU job，IPC 建连和 Java byte[]/解析开销可能成为主要瓶颈。

建议保留当前 TCP 文本接口作为诊断/兼容接口，同时新增持久 binary data channel：

CONNECT → HELLO → 多个 binary job → 多个 binary result → KEEPALIVE

不要每个 tile 都重新 connect。

## 2. MatMul bucket 逻辑

当前 native MM_BUCKETS：

32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536

bucketize(m)、bucketize(k)、bucketize(n) 分别取桶。

这个方向正确。不要改成只对 m 取桶。

但是必须增加 tensor byte-size 安全检查。

例如 65536 × 65536 = 4,294,967,296 bytes，已经碰到 32-bit uint 上限，更不用说手机实际内存。

不能只用“v <= 65536”判断 shape 合法。

必须统一增加 checkedTensorBytes(rows, cols, elementSize)，检查：

- size_t 乘法溢出
- uint32 dataSize 上限
- 实际允许的 native 工作内存
- 当前 Job 的内存预算

## 3. Padding 是目前非常大的效率风险

例如：

33 → 64

如果三个维度都发生这种情况：

33³ → 64³

计算量约放大 7.3 倍。

不要取消 bucket，因为 HTP 的桶约束是真实硬件约束。

正确解决方法是跨 Chunk 合桶。

例如多个相同类型的 33/31/30 等任务，不要各自执行，而是进入 Batch Builder，尽可能组成 64/128/256 等桶。

这就是 9×9 工作集存在的核心价值之一。

## 4. 9×9 SuperChunk 的正确定位

9×9 不应该变成 Minecraft 的一个巨型 Chunk。

它应该只是工作集：

81 个普通 Chunk
→ Scheduler
→ Feature Collector
→ Batch Builder
→ NPU

按照以下维度重新分组：

- feature type
- DensityFunction 阶段
- 输入/输出 shape
- 优先级
- 空间位置
- 玩家运动方向

例如 81 个 Chunk 可以拆成：

Height jobs
Noise jobs
Cave jobs
Biome jobs

然后各自形成 NPU batch。

## 5. 576 MiB Dense Matrix 应该彻底禁止

98,304 density points × 6,144 columns ≈ 576 MiB 的 dense int8 矩阵，是一种很差的表示方式。

它把局部邻域关系强行展开成 dense matrix。

不要继续沿着这个方向优化。

正确方向：

- tiled processing
- streaming
- implicit neighborhood
- buffer reuse
- cross-chunk batching
- operator fusion

如果一个局部计算只有 8 个输入和 1 个输出，不要为了“必须让 NPU 亮”而单独调用 NPU。

真正应该搬的是 CPU 热点中的大批量、规则数学计算。

## 6. 当前 native 的临时 vector 应该改掉

runMatMulInt8Buf() 当前会创建：

Ap
Bp
Cpad

高频世界生成会不断触发 native allocation、memcpy 和 heap 抖动。

推荐：

ScratchBufferPool

至少复用：

A buffer
B buffer
C buffer

进一步做 double buffering：

CPU prepare B
NPU execute A

然后交换。

目标是把：

CPU prepare + NPU execute

尽量变成：

max(CPU prepare, NPU execute)

而不是简单相加。

## 7. QNN shared buffer 值得验证

Qualcomm 官方 QAIRT 文档提供 HTP shared-buffer 机制，目标之一就是减少 host CPU 与 HTP 之间的数据复制。

因此应该实际检查当前 QNN stack 是否支持适合本项目的：

QNN_MEM_TYPE_CUSTOM
+
QNN_HTP_MEM_SHARED_BUFFER

不要直接假设一定更快。

必须做 A/B benchmark：

普通 client buffer
vs
shared buffer

分别记录：

host prepare
copy in
graphExecute
copy out
total
peak memory

## 8. Graph cache 是另一个高风险点

当前 MAX_CACHED_GRAPHS = 8。

超过后会 contextFree + contextCreate，并清空 graph cache。

对于 Minecraft 这种 shape 很多的 workload，会产生 cold graph、graph finalize 和 context reset 的延迟尖峰。

不要简单把 8 改成 64。

推荐：

有限 graph + Shape Planner + 任务重排

让真实 workload 主要集中到少数高频 bucket。

必要时预热最常用 shape。

## 9. Shape Planner 应该成为核心模块

进入 native 前先计算：

requested(m,k,n)
→ bucket(m), bucket(k), bucket(n)

同时计算：

paddingRatio
computeRatio
estimatedInputBytes
estimatedOutputBytes
cacheHit

然后 Scheduler 决定：

- 立即执行
- 等待极短时间收集更多同类 Job
- 与其他 Chunk 合并
- 使用已有 graph
- 拒绝危险的大 shape

这层是 9×9 batching 的核心。

## 10. 世界生成应该分成两层数据

### Layer A：NPU-friendly

只传数学数据：

- 坐标
- seed 派生参数
- noise 输入
- density
- height
- cave feature
- biome feature
- 其他连续数学量

优先考虑 int8/int16/fp16，具体由精度实验决定。

### Layer B：Minecraft-native

最后才转换：

density / height / biome
→ BlockState / ChunkSection / Palette

NPU 不应该接触：

- Chunk Java 对象
- BlockState 对象
- NBT
- Palette
- Java object graph

NPU 只负责数学。

## 11. 原版相似度必须成为硬指标

目标不是随便生成一个世界。

目标是：

同 Seed、同坐标，尽可能接近 Minecraft 26.3 原版。

建立自动比较：

Original Generator
vs
NPU Generator

连续数据比较：

- height error
- density error
- noise error
- RMSE
- max error

离散数据比较：

- biome mismatch
- block mismatch
- cave mismatch
- Chunk boundary mismatch

性能比较：

- CPU ms/chunk
- NPU ms/chunk
- IPC ms/chunk
- memcpy ms/chunk
- peak native memory
- queue wait
- 1% low
- 0.1% low

## 12. 世界生成推荐调度

Player
→ Chunk Request
→ 9×9 Workset
→ Feature Collector
→ Shape Planner
→ Batch Builder
→ NPU Queue
→ HTP Graph
→ Result Buffer
→ Lightweight Chunk Builder
→ Normal Minecraft Chunk

优先级建议：

当前 Chunk：100
附近 Chunk：80
玩家移动方向：60
其余 9×9：30

具体数字不是硬编码要求，应该根据实际玩家移动速度和生成吞吐调整。

## 13. 不要把“全世界 NPU 化”理解成“每一行代码都必须 NPU”

应该 NPU 化的是 CPU 热点。

适合 NPU：

- Noise
- 大批量连续数学
- 向量变换
- 大规模 density feature
- 大规模 height feature
- 可融合的重复函数
- 跨 Chunk 同构计算

应该主要留在 CPU：

- Chunk 对象创建
- BlockState 写入
- Palette
- NBT
- Java 集合
- 分支密集逻辑
- 状态机
- Chunk 保存
- 最终 Minecraft API 调用

## 14. 当前仓库缺少真正的 Terrain 上层

当前仓库主要提供：

MATMUL
MATMUL16
MATMUL8
XFORM
以及 IPC/QNN runtime。

这些只是底层 primitive。

下一阶段应该增加：

NpuJob
NpuBatch
NpuShape
NpuBuffer
TerrainWorkset
TerrainResult
TerrainBatchManager
SuperChunkScheduler
DensityBatcher
NpuTerrainPipeline
ChunkResultAssembler

Minecraft 侧只提交 TerrainJob，不应该自己操心 m/k/n。

## 15. 推荐模块边界

Java：

NpuTerrainScheduler
NpuBatchPlanner
NpuIpcChannel
TerrainWorkset
TerrainJob
TerrainResult

Native：

QnnRuntime
GraphCache
ShapePlanner
ScratchBufferPool
NpuExecutor
TerrainKernel

Minecraft/Fabric：

TerrainHook
DensityCollector
ChunkResultAssembler

Minecraft 层不应该依赖 QNN 细节。

## 16. 必须增加 telemetry

每个 Job 至少记录：

job_id
chunk_x
chunk_z
feature_type
m
k
n
bucket_m
bucket_k
bucket_n
padding_ratio
queue_wait_us
prepare_us
copy_in_us
graph_execute_us
copy_out_us
assemble_us
total_us
input_bytes
output_bytes
native_peak_bytes
cache_hit
graph_create_us

最终必须能回答：

NPU actual compute 占多少？
IPC 占多少？
memory copy 占多少？
CPU prepare 占多少？
CPU assemble 占多少？
queue wait 占多少？
graph creation 占多少？

没有这套数据，优化基本就是蒙眼修车。

## 17. 开发顺序

### P0：安全

- tensor byte-size overflow
- 超大 bucket allocation guard
- graph create/finalize 失败状态恢复
- 临时 buffer 生命周期
- context reset 后 cache 状态

### P1：数据通道

- persistent IPC
- binary job protocol
- batch submit
- batch result

### P2：内存

- ScratchBufferPool
- double buffering
- 减少 memcpy
- 验证 shared buffer

### P3：调度

- 9×9 Workset
- priority queue
- ShapePlanner
- cross-chunk batching

### P4：Minecraft 26.3

直接读取真实 bytecode/source：

DensityFunction
→ NoiseChunk
→ DensityVolume
→ Chunk

找到真正占 CPU 时间的连续数学阶段。

### P5：一致性

建立 Original vs NPU 自动比较。

### P6：最终优化

根据 telemetry 决定：

- 哪些函数融合
- 哪些留 CPU
- 哪些进入 NPU
- batch 多大
- 9×9 是否足够
- 是否需要更大工作集

## 18. 最终架构

Minecraft 26.3
→ Terrain Hook
→ 9×9 Workset Scheduler
→ Feature Collector
→ Shape Planner
→ Batch Builder
→ Persistent NPU Channel
→ Native NPU Executor
→ Scratch / Double Buffer
→ Terrain Result
→ Lightweight Chunk Assembler
→ Normal Minecraft Chunk

## 19. 最重要的一句话

**保留原版生成规则和 Seed 语义，把连续、规则、可批量的重计算抽出来，在 9×9 工作集内跨 Chunk 聚合，按 HTP 的 m/k/n 桶一次性批量送入 NPU，用复用 buffer 和持久 IPC 消灭搬运成本，最后只用 CPU 做轻量 Minecraft Chunk 组装。**

这样才同时满足：

- 世界观感尽可能接近原版
- NPU 真正承担 CPU 热点
- 内存不会被 Dense Matrix 撑爆
- Minecraft 仍然看到正常 Chunk
- 玩家移动时可以提前生成
- Seed 保持确定性
- 系统可以通过 telemetry 持续优化

## 外部依据

本报告关于 QNN HTP shared buffer、QNN tensor 生命周期和 context/graph 能力的外部依据来自 Qualcomm AI Runtime 官方文档：

- QNN HTP Shared Buffer Tutorial
- QNN Tensor API
- QNN General Overview

## 给后续 AI 实现者

**这是 GPT-5.6 Luna 的代码审查报告。**

后续 DeepSeek、元宝或其他实现 AI 修改仓库时：

1. 先读取当前最新代码。
2. 对照本报告验证当前状态。
3. 每完成 P0/P1/P2 项目必须有实测数据。
4. 不要用“理论上更快”替代 benchmark。
5. 不要删除已有可用 HTP 路径来换取架构重写。
6. Minecraft 26.3 真实 bytecode/source 优先级高于旧版本经验。
7. 所有 NPU terrain 优化都必须同时记录 CPU baseline。
8. 所有内存优化都必须记录 peak native memory。
9. 最终以 Minecraft 实际 Chunk 生成吞吐、卡顿情况和原版相似度验收。

**报告结束。**


---

# 20. 第二轮 GPT 代码审查：当前 main 分支新增发现

本节是在读取当前 main 分支实际 mcnpu.cpp、NpuRuntime.java、NpuService.java、NpuServiceClient.java 后追加。

## 20.1 【P0】JNI 入口在安全检查前就可能分配数 GB 内存

当前 nativeMatMulInt8Buf 的顺序是：

1. 检查 m/k/n > 0。
2. 检查 mmSizeAllowed。
3. 读取 Java byte[] 长度。
4. 直接构造 A、B、C 三个 vector。
5. 之后才进入 runMatMulInt8Buf 的 mmShapeSafe。

因此 m=k=n=65536 时，入口可能先尝试构造约 4 GiB A、4 GiB B、4 GiB C，然后才执行 64 MiB tensor budget。

这是当前代码最危险的内存安全问题之一。

必须在任何 vector allocation 前检查：

- uint64_t m*k、k*n、m*n
- Java array 长度
- bucket size
- per-tensor byte budget
- whole-request memory budget

之后才允许分配。

## 20.2 【P0】graphCount 在 graph 创建失败时可能错误增长

当前创建 graph 前就执行 g.graphCount++。

如果 graphCreate、tensorCreateGraphTensor、graphAddNode 或 graphFinalize 失败，代码会 erase C++ map entry，但没有同步减少 graphCount。

更重要的是，QNN 2.27 下 graph 由 context 持有，erase map 不等于释放 QNN graph。

建议：

1. graph construction 全部成功后再正式计入 cache。
2. 任意 construction failure 都把 context 标记为 dirty。
3. 如果 QNN 没有 graphFree，则在安全点完整 context reset。
4. reset 后清空 graph maps 和 graphCount。

## 20.3 【P0】CPU affinity 绑错线程

initRuntime 中 sched_setaffinity(0, ...) 只作用于调用 initRuntime 的当前线程。

当前 QNN 初始化发生在 NpuService.serverLoop 线程，但真正的 matMulInt8Buf 等请求来自 clients 线程池。

因此当前注释声称的“让服务远离 prime core”实际上只约束初始化线程，不能保证真正执行 QNN request 的线程远离 prime core。

而 HTP/DSP 执行本身也不是由 sched_setaffinity 控制的。

正确方案：

NPU Job Queue + 专用 NPU worker 后，在 worker 启动时设置 affinity，然后只从该 worker 进入 QNN。再做 affinity on/off 实测。

## 20.4 【P1】NpuRuntime 的 synchronized 让 Java 线程池也基本失去并行意义

NpuRuntime 的 init、status、smoke、add、matMul、matMulFp16、matMulInt8、xform、matMulInt8Buf、shutdown 等大量方法都是 synchronized。

因此：

8 handlers
→ NpuRuntime.class monitor
→ 一次只允许一个调用

之后 native 又有 gRuntimeMutex。

也就是说当前存在 Java class lock + native runtime mutex 双重串行。

不要直接删除同步。正确方向是：

IPC handlers
→ NpuJobQueue
→ 唯一 NPU execution worker
→ native QNN

控制面 STATUS/PING 与数据面执行分开。

## 20.5 【P1】accept loop 与 client handler 共用同一个 8-thread pool

当前 clients = newFixedThreadPool(8)。

serverLoop 本身也通过 clients.execute 运行，每个 accepted socket 又通过 clients.execute(handle)。

因此这 8 个线程同时承担 accept supervisor 和 client handler。

一旦改成持久连接，最坏情况是：

1 个 accept worker + 7 个长期 client handler。

第 8 个连接只能排队。

必须拆分：

- accept/supervisor：1 thread
- client IO：独立 bounded pool
- NPU execution：1 dedicated worker

如果 Minecraft 使用单一持久连接，client IO 甚至可以只有少量线程。

## 20.6 【P1】stall restart 存在 epoch / serverLoop 竞态窗口

onStartCommand 检测 stall 后会提前把 serverLoopStarted 设为 false，然后重新 ensureServerLoop。

旧 serverLoop 线程此时未必已经退出。

于是新 worker 可能在旧 worker 清理期间启动并尝试 bind 38761，产生 bind collision、共享 server 状态竞争和 listener 状态短暂不一致。

正确方式：

request stop
→ close server
→ 等旧 worker 自己 finally
→ finally 设置 serverLoopStarted=false
→ supervisor 再启动新 worker

不要在旧 worker 真正退出前提前清零运行标志。

## 20.7 【P1】当前 binary path 仍然存在多重内存复制

当前 SUBMITBIN_MATMUL8 路径大致是：

socket
→ Java byte[] A/B
→ native vector A/B
→ padding Ap/Bp
→ QNN client buffer
→ Cpad
→ native C
→ JNI byte[]
→ Java result
→ socket

这对于高频 terrain workload 仍然偏重。

目标：

persistent socket
→ pooled native/direct buffer
→ QNN buffer
→ NPU
→ pooled output
→ socket

至少先用 scratch pool 消灭每次 vector allocation。

## 20.8 【P1】binary header 应统一采用 64-bit 解析和内存预算

handleSubmitBinMatMul8 当前最终还是以 int m/k/n/alen/blen 创建 Java byte[]。

协议层应先解析 long，再统一检查：

- dimension cap
- tensor byte cap
- whole-job byte cap
- result byte cap

检查通过后再 cast 到 int。

## 20.9 【P1】loadRuntime 初始化失败路径需要统一 cleanup

dlopen、provider lookup、backendCreate、deviceCreate、contextCreate 任一步失败时，都应该进入统一 cleanupRuntimeLocked。

必须确保：

backend/device/context/logger/dlopen

不会因为中途失败而泄漏。

否则重复初始化可能积累资源。

## 20.10 【P1】contextFree 返回值没有检查

resetContextLocked 直接调用 contextFree，然后把 g.context 设为 null，再创建新 context。

必须记录 contextFree rc。

如果释放失败，应把 runtime 标记为 unhealthy，并考虑完整 QNN stack restart，而不是默认旧 context 已经释放。

## 20.11 【P2】首次 MatMul8 calibration 仍会制造 latency spike

首次 shape 会：

NPU execute
→ CPU 4 rows × n × k dot product
→ median
→ scaleEff

这已经比旧版全矩阵 calibration 好很多，但如果 terrain 运行时不断出现新 shape，仍会周期性出现第一次执行延迟尖峰。

推荐：

- 预热阶段 calibration。
- 每个 bucket 保存 calibration result。
- 正式 terrain 实时路径禁止临时 calibration。
- 未校准 shape 不进入实时路径。

## 20.12 【P2】MatMul8 固定 scale 不能直接作为 terrain 的最终量化方案

当前实验使用固定 scaleA/scaleB/scaleC，再通过首次运行的 scaleEff 修正。

正式 terrain pipeline 应采用 feature-specific quantization。

Noise、Density、Height 等 feature 应分别统计：

- min/max
- scale
- zero point
- saturation rate
- quantization error

不要把 MatMul benchmark 的量化参数直接当成整个世界生成器的量化方案。

## 20.13 【P2】JNI output 仍有额外复制

当前结果存在 Cpad → Cout → tmp → NewByteArray → SetByteArrayRegion 的多级复制。

后续 pooled/direct output path 至少应该消灭其中一到两次复制。

## 20.14 第二轮问题优先级

必须先修：

1. JNI 入口提前分配巨大 vector。
2. graph construction failure 后的 graph/context 资源处理。
3. graphCount failure path。
4. contextFree 返回值检查。

接下来修：

5. NpuRuntime 全局 synchronized 串行。
6. accept/client/NPU 共用线程池。
7. CPU affinity 绑错线程。
8. stall restart 竞态。
9. binary path 多重 copy。
10. persistent IPC 与 execution queue 解耦。

然后优化：

11. calibration warmup。
12. feature-specific quantization。
13. JNI output copy。
14. shared buffer A/B。
15. terrain batch planner。

## 20.15 第二轮结论

当前项目已经从“能不能调用 HTP”进入“能不能让 Minecraft 高频 workload 持续、稳定、低内存地吃到 HTP”的阶段。

因此下一阶段不要继续无限增加测试命令。

应严格按照：

安全边界
→ 生命周期
→ 线程模型
→ 数据通道
→ buffer 复用
→ batch planner
→ terrain

推进。

尤其在正式加入 terrain 之前，必须先消灭 JNI 巨大分配风险。


---

# 21. 第三轮 GPT 全量复核：修复状态、隐藏 bug、性能瓶颈

本轮重新读取当前 main 分支实际代码，重点验证报告中的问题是否真的已经落地，并继续检查 JNI、QNN 生命周期、Android Service、IPC、构建层和性能路径。

## 21.1 修复状态总表

| 项目 | 当前状态 | 结论 |
|---|---|---|
| m/k/n 独立 bucket | 已实现 | ✅ |
| native tensor byte budget | 已实现于 runMatMulInt8Buf | ⚠️ JNI 入口仍在检查前分配 |
| 65536 巨型 tensor 防护 | 部分实现 | ❌ |
| graph cache 上限 | 已实现 | ⚠️ failure path 仍有问题 |
| graphCount | 未完全修复 | ❌ |
| contextFree rc 检查 | 未修复 | ❌ |
| 初始化失败 cleanup | 未完全修复 | ❌ |
| CPU affinity | 代码存在 | ❌ 绑的是 init/serverLoop 线程 |
| Java synchronized | 仍存在 | ❌ 数据面双重串行 |
| 8-thread client pool | 仍存在 | ❌ accept 与 client 共池 |
| persistent IPC | 未实现 | ❌ 每次 request 都新建 Socket |
| binary IPC | 已实现 | ⚠️ 仍有多级复制 |
| calibration cache | 已实现 | ⚠️ 首次 shape 仍可能尖峰 |
| feature-specific quantization | 未实现 | ❌ |
| ScratchBufferPool | 未实现 | ❌ |
| double buffering | 未实现 | ❌ |
| Terrain abstraction | 未实现 | ❌ |
| Minecraft 26.3 terrain hook | 当前仓库未实现 | ❌ |

因此当前 main 不能标记为“P0 已全部修完”。它已经有明显的安全防线，但关键边界仍未闭合。

## 21.2 【P0 仍存在】JNI nativeMatMulInt8Buf 的安全检查顺序错误

当前入口仍是 GetArrayLength → vector A/B/C allocation → runMatMulInt8Buf → bucketize/mmShapeSafe。

m=k=n=65536 时，进入 mmShapeSafe 前仍可能尝试：A 约 4 GiB、B 约 4 GiB、C 约 4 GiB。该问题不是单纯性能问题，可能直接导致进程 OOM。

必须改成：m/k/n 64-bit multiplication → bucketize → bucket tensor byte check → original input byte check → whole-job byte budget → Java array length validation → 最后 allocation。

并增加单 Job 总预算，例如 A+B+C+padding scratch 不得超过 JOB_MAX_BYTES。

## 21.3 【P0 仍存在】graph construction failure 会留下脏 context

当前多个路径仍采用 graphCount++ → graphCreate → tensorCreate → graphAddNode → graphFinalize。失败时只是 erase map entry，但 QNN graph 并不会因此消失。

必须增加 contextDirty 状态。任意 graph construction failure 都应标记 dirty，并在安全点执行 contextFree → contextCreate → maps.clear → graphCount=0。否则 C++ map 看起来为空，QNN context 内仍可能持有失败 graph。

## 21.4 【P0 仍存在】graphCount 与真实 QNN graph 数量可能脱钩

graphCount++ 发生在 construction 成功之前。失败一次就会虚增一次，连续失败会提前触发 MAX_CACHED_GRAPHS 和不必要的 context reset。

建议使用 pendingGraph，只有 graphFinalize 成功后才正式计入 cache。由于 QNN 没有 graphFree，failure 仍应使 contextDirty=true 并 reset。

## 21.5 【P0/P1】context reset 必须成为完整状态机

当前 resetContextLocked 没检查 contextFree rc，也没有完整的 runtime health 状态。

建议状态：READY、DIRTY、RESETTING、FAILED，并记录 contextFree rc、contextCreate rc、reset duration、reset count、reason、reset 前 graph count。

## 21.6 【P1】当前 8 线程并没有产生真正的 NPU 并行

Java FixedThreadPool(8) 最终仍会进入 native gRuntimeMutex，而且 NpuRuntime 方法本身大量 synchronized。因此执行模型仍是多线程排队进入一个 QNN critical section。

最终应该改为 IPC IO → bounded job queue → one NPU worker → QNN。不要继续靠增加 Java worker 数量提高所谓 NPU 并行。

## 21.7 【P1】client pool 在 persistent IPC 后会成为硬瓶颈

当前 clients = newFixedThreadPool(8)，serverLoop 自己也运行在该 pool，每个 socket handler 又占一个 worker。持久连接情况下最坏是 1 个 accept worker + 7 个长期 handler，第 8 个连接排队。

建议拆成 accept/supervisor、bounded client IO、dedicated NPU worker 三层。Minecraft 单一长期连接时不需要 8 个 client worker。

## 21.8 【P1】stall restart 仍可能双 worker

onStartCommand 检测 stall 后会提前把 serverLoopStarted 设为 false，而旧 worker 未必已经进入 finally。此时新旧 worker 有重叠窗口，可能争抢 38761 和 server 状态。

正确顺序：request stop → close server → 唤醒旧 worker → 等旧 worker finally → serverLoopStarted=false → supervisor 启动新 worker。禁止外部线程提前清零运行标志。

## 21.9 【P1】NpuRuntime 全 synchronized 应拆控制面和数据面

init、status、smoke、add、matMul、matMulFp16、matMulInt8、xform、matMulInt8Buf、shutdown 仍大量 synchronized。PING/STATUS 不应该等待大型 terrain job。

最终拆成 Control Plane：PING、STATUS、health、metrics；Data Plane：submit、batch submit、result。数据面统一排队，控制面读取原子状态快照。

## 21.10 【P1】binary path copy chain 仍然过长

当前路径仍接近 socket → Java byte[] → native vector → Ap/Bp → QNN client buffer → Cpad → Cout → tmp → Java byte[] → socket。

对于用户要求的 400+ FPS、约 2.5 ms/frame 预算，这些搬运必须单独计时。最终 telemetry 至少拆出 IPC read、Java→native、padding、QNN execute、output copy、native→Java、IPC write。

## 21.11 【P1】persistent IPC 尚未落地

NpuServiceClient.request() 每次 new Socket → connect → request → response → close。正式 terrain 每个小 job 都付连接成本。

应保留当前 request() 作为诊断/兼容路径，同时增加长期 NpuChannel，使用 CONNECT → HELLO → 多个 JOB/RESULT，并加入 request id 防止异步 batch 后结果错配。

## 21.12 【P2】native allocation 仍是热路径问题

runMatMulInt8Buf 每次可能创建 Ap、Bp、Cpad，JNI 入口还创建 A、B、C、tmp。高频 terrain workload 下会持续 native heap churn。

必须引入 ScratchBufferPool，至少复用 A、B、Ap、Bp、Cpad、output。进一步做 double buffering，使 stage time 接近 max(prepare, execute)，而不是 prepare + execute。

## 21.13 【P2】8 个 graph cache 不能简单无限增大

超过 8 个 graph 就 context reset 会产生 cold-start spike。正确方向不是盲目把 8 改成 64，而是统计 shape frequency、预热 top-N、Shape Planner 合并低频 shape，并让实时 Chunk 尽量只使用已预热 shape。

## 21.14 【P2】首次 calibration 仍可能进入实时路径

首次 bucket 会执行 host-side calibration。虽然已经比旧版轻很多，但仍可能在第一次遇到新 shape 时产生延迟尖峰。

正式 terrain path 应在 startup warmup 阶段完成 calibration，实时路径只使用已校准 bucket。

## 21.15 【P2】固定量化参数只能作为 benchmark

当前 scaleA、scaleB、scaleC 与 runtime calibration 适合证明 HTP 能跑，但不能直接作为正式 terrain 参数。Noise、Density、Height、Cave 应分别统计 range、scale、zero point、saturation、RMSE、max error，并验证确定性。

## 21.16 【P2】CMake 没有显式 release 优化策略

当前 CMake 明确写出的编译选项主要是可见性、section、Wall/Wextra 和 gc-sections，没有显式 release O2/O3。Android release 构建可能通过外部配置提供优化，但 benchmark 应固定构建类型和参数，避免环境差异。不要未经 benchmark 就无脑 O3。

## 21.17 【P2】当前仓库没有 Terrain 上层代码

当前仓库实际 Java/native 主要是 MainActivity、NpuRuntime、NpuService、NpuServiceClient、ShizukuHelper、mcnpu.cpp。没有真正的 Minecraft 26.3 terrain collector、batch scheduler 或 chunk assembler。

所以 9×9、Density、Noise、Chunk batching 目前仍是目标架构，不应写成已实现。真正接入层仍需要在 bslsjdk/mcjavanpu 或新的 Fabric integration 中落地。

## 21.18 【P3】必须按 1 帧预算验收

用户实际目标约 400+ FPS，即约 2.5 ms/frame。因此不能拿单次 graphExecute 数字当成整个系统延迟。

验收应至少记录：queue wait、IPC、prepare、copy、graphExecute、assemble、total。正常情况下 Minecraft 主线程不等待 NPU，目标是 ≤1 frame 预算，最坏 ≤2 frame，并持续观察 1%/0.1% low。

## 21.19 本轮“已修/未修”结论

已真正落地：m/k/n 三维独立 bucket、native tensor byte budget、padding、binary IPC、graph cache 上限、calibration cache、QNN context 级 graph 生命周期思路、loopback listener 自检、foreground service 基础结构。

尚未真正落地：JNI allocation guard 前置、graph failure dirty-context recovery、graphCount 正确计数、contextFree rc、init cleanup、dedicated NPU worker、正确 affinity、persistent IPC、scratch pool、double buffering、warmup calibration、feature-specific quantization、terrain scheduler、cross-chunk batching、Minecraft 26.3 terrain hook、end-to-end telemetry。

## 21.20 本轮最终优先级

### P0
1. JNI pre-allocation guard。
2. graph failure dirty-context recovery。
3. graphCount 修正。
4. contextFree rc。
5. init cleanup。

### P1
6. dedicated NPU worker。
7. accept/client/NPU 线程拆分。
8. stall restart 生命周期修复。
9. persistent IPC。
10. binary protocol request-id。

### P2
11. ScratchBufferPool。
12. double buffering。
13. warmup calibration。
14. shared-buffer A/B。
15. end-to-end telemetry。

### P3
16. TerrainWorkset。
17. 9×9 scheduler。
18. cross-chunk batching。
19. feature quantization。
20. original-vs-NPU deterministic comparison。

## 22. 给 DeepSeek / 元宝的交叉检查要求

后续任何修改必须同时提交：修改前路径、修改后路径、bug 根因、为什么新代码能证明问题消失、benchmark 数据、内存峰值数据。涉及 QNN context/graph 必须记录 rc；涉及线程模型必须记录 queue wait 和 end-to-end latency；不能只改注释、README 或 report 就声称修复。

重点复查关键词：std::vector<int8_t> A、g.graphCount++、contextFree(、sched_setaffinity(、newFixedThreadPool(、synchronized、new Socket(、new byte[、Base64、Cpad、scaleEff。

## 23. 本轮结论

当前 main 已经能实证 Android 服务调用 QNN/HTP V73，但距离 Minecraft 世界生成真正受益还有明确的一整条工程链：JNI 安全边界 → QNN 生命周期 → 专用 NPU worker → 持久数据通道 → buffer reuse → cross-chunk batching → Terrain integration。

这条链完成之前，不应把当前版本当最终性能版本。

本节审查时间：2026-10-03。
