
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
