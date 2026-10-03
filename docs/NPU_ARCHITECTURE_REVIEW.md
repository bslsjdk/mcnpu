# MCJavaNPU · NPU 接入架构评估报告

> **作者：元宝（Yuanbao）**
> 日期：2026-10-03
> 针对：RMX3852（Snapdragon 8s Gen 3 / SM8635）+ HTP V73 + Minecraft 26.3 + ZL2
>
> 本文只做诊断与方案评估，不含凭证。结论均基于仓库实测代码与设备日志。

---

## 0. 一句话结论

**`/npu matmul` 已经证明 NPU 可用，但"用 matmul 生成地形"这条路从算术上就是负优化。**
真正值得做的是让 NPU 直接算 noise 函数本身，而不是"提取特征 → 线性组合"。

---

## 1. 当前链路盘点

### 1.1 两条接入路线

| 路线 | 进程模型 | deviceCreate | 结论 |
|---|---|---|---|
| **A. mcnpu + IPC** | 独立 App，Minecraft 走 TCP 127.0.0.1:38761 | ✅ 通过 | 当前唯一可用 |
| **B. mcfclnpu 进程内** | ZL2 NativePlugin，Minecraft JVM 直接 dlopen | ❌ rc=14001 | **已被实验否掉** |

### 1.2 路线 B 为什么必须停

日志实锤：

```
QNN_LOAD_OK            ✅
GET_PROVIDERS_OK       ✅ count=1
HTP_PROVIDER_SELECTED  ✅ backendId=6
BACKEND_CREATE_OK      ✅
DEVICE_CREATE          ❌ rc=14001
```

根因不在设备、不在 QNN 版本：

```
error=dlopen failed: library "libcdsprpc.so" not found
       needed by .../libQnnHtpV73Stub.so in namespace clns-9
```

`clns-9` 是**启动器的 classloader namespace**。而 Manifest 里那行：

```xml
<uses-native-library android:name="libcdsprpc.so" android:required="false" />
```

**只对声明它的 APK 自己的进程生效**。mcfclnpu 只是个插件，真正加载它的是 ZL2 启动器进程 —— 声明等于没写。

对照 mcnpu（独立进程，同一台机器、同一套 QNN 库）deviceCreate 成功，差异**只有进程归属**。

> 结论：零 IPC 路线死在 linker namespace，不是调参能救的。**不要再往 mcfclnpu 投时间。**

---

## 2. 已修复的 Bug（本次提交）

| 仓库 | Commit | 问题 |
|---|---|---|
| mcnpu | `8bc6089979` | `MM_BUCKET_COUNT=7` 是死常量（数组实际 12 项），从未被读取；改为 `MM_BUCKET_MAX` 并通过 `CAPABILITIES` 暴露真实上限 |
| mcjavanpu | `be20c8f352` | Java bucket 表 `{128..2048}` 与 native `{32..65536}` 不一致，导致**双重 padding**（Java 取 192 → native 无 192 → 再取 256）；且超限形状要等几 MB buffer 建完才失败 |
| mcjavanpu | `fa62f1f5c1` | 地形 16×384×16 = 98304 行 > 65536 上限，**每个 chunk 必然 ERR BUF_TOO_LARGE**，NPU 全程旁观；改为 `submitSplit()` 分块，并把真实错误写进 `Result.note` |

---

## 3. 核心问题：地形路线为什么是负优化

### 3.1 算术账

单个 chunk 的一次调用：

| 阶段 | 运算量 | 在哪跑 |
|---|---|---|
| `features()` × 98304 点 × 32 特征 | ~314 万次浮点 + sin/cos | **Java** |
| 量化成 int8 | ~314 万次 | **Java** |
| IPC 传输 | ~3 MB | — |
| **matmul** | 98304 × 32 × 1 | **NPU** ← 只有这步 |
| 反量化 + 归一化 + 高度梯度 | 数十万次 | **Java** |

**NPU 只承担了整条链的最后一步。** 等于花几百毫秒准备数据，让 NPU 算零点几毫秒，然后宣布"加速了"。

### 3.2 更根本的问题

原版地形生成用的是**高度优化的原生 noise**（`PerlinNoise`/`SimplexNoise`，SIMD + 缓存友好）。
这里改成：**Java 逐点算 32 个特征 → 量化 → 传 3MB → NPU 做一次线性组合**。

这个替换在结构上就是慢的，跟 NPU 快不快无关。

### 3.3 同步阻塞

```java
// DensitySamplerMixin, @At("HEAD"), cancellable = true
NpuTerrainGen.generate(...);   // 同步
ci.cancel();
```

地形生成是**阻塞式**的，玩家在等区块。加上 `NpuServiceClient` 全局 `synchronized`，所有地形线程串行排队。

---

## 4. 正确的方向：让 NPU 直接算 noise

不该是：
```
提取特征(Java) → 量化(Java) → matmul(NPU) → 后处理(Java)
```

应该是：
```
坐标(NPU 内部生成) → noise 函数(NPU) → 密度值(NPU) → 回传
```

理由：
- noise 的核心就是**大量并行的点运算**，这正是 HTP 擅长的形状
- 省掉全部 Java 侧特征构建和量化
- 回传只有 m×1 个结果，不是 3MB

### 4.1 形状选择建议

实测约束（SM8635 / HTP V73）：

```
m < 64              → 设备直接拒绝
m = 63              → 临界，间歇失败
m >= 128            → 稳定
m ∈ [100, 257]      → 成本不变（m 几乎免费）
k / n               → 无白名单限制
cost                → 约 O(k*n)
```

**推荐形状**：

| 场景 | 形状 | 理由 |
|---|---|---|
| 地形 noise | `A[16384 × 4] × B[4 × 1]` | 一个 16×16 截面；m 免费，k 极小 |
| 光照传播 | `A[4096 × 16] × B[16 × 1]` | 16 邻域折叠 |
| 实体 AI | `A[128 × 64] × B[64 × 16]` | m 拉到 128 下限以上 |

关键：**k 要小，m 要大**。成本是 O(k·n)，而 m 在 100~257 区间几乎不花钱。
当前地形用的 `K=32` 偏大，`m=98304` 又超限 —— 两头都踩错了。

---

## 5. Minecraft 接入点全览

按「NPU 友好度 × 收益」排序：

| 接入点 | 位置 | NPU 友好度 | 建议 |
|---|---|---|---|
| **噪声生成** | `DensitySampler$Bound.sampleVolume` | ★★★★★ | 主战场，见 §4 |
| **光照传播** | `LightEngine.runLightUpdates` | ★★★★☆ | BFS 不适合，但**批量光值折叠**可以 |
| **实体批处理** | `Entity` tick | ★★★☆☆ | 实体数波动大，需 bucket 稳定 |
| **生物群系查找** | `BiomeSource` | ★★☆☆☆ | 查表为主，NPU 无优势 |
| **方块更新** | `LevelChunk` | ★☆☆☆☆ | 分支密集，最差选择 |

### 5.1 光照的正确做法

`LightEngine.runLightUpdates()` 是 BFS 队列，**天生不适合 NPU**（串行依赖）。
但可以这样切：

```
runLightUpdates (每帧)
   ↓
只做一件事：把当前队列快照成固定形状矩阵
   ↓
NPU 批量计算「候选光值」
   ↓
回写时用 min(原值, NPU值) 兜底 —— 保证不会比原版更暗
```

当前 `NpuLightHook` 只计数不计算，这个设计是对的 —— 先看频率再动手。

---

## 6. 必须的架构改造（按优先级）

### P0 · 异步化（架构性，不做别的都白搭）

渲染/地形线程**绝不能**碰 `request()`。

```
Render Thread          Worker Thread
     │                      │
  submit()  ────────────►  队列（latest-wins）
  O(极小)                   │
     │                   IPC → NPU
  poll()  ◄────────────  结果双缓冲
  O(极小)
```

三条铁律：
1. **不分配**：三缓冲复用固定 `byte[]`，不每帧 new
2. **不阻塞**：`submit` / `poll` 只有原子操作
3. **latest-wins**：新帧作废旧帧，不排队

> 当前 `NpuServiceClient` 是单连接 + 全局 `synchronized`，诊断命令会和实时通道抢锁。异步通道必须**独立连接**。

### P1 · 去掉 CPU 参考计算

`runMatMul`(655) / `runBatchXform`(773) / `runMatMul8`(875) 三处 benchmark 仍在每次调用里跑 CPU 参考。
那个 `speedup=Nx` 数字里 CPU 占九成以上 —— **端到端其实比纯 CPU 还慢**。

CPU 参考只在首次建图时校验一次即可。

### P2 · 预热

服务启动时把常用形状建好 + finalize，不要让第一次建图砸在帧上。

### P3 · 测量（不靠嘴说）

每个算子记录：`submit_us / ipc_us / npu_exec_us / result_us / end_to_end_us`
分别测 n=16/64/256/1024，**取 p99**（p50 是自我安慰，p99 才决定掉不掉帧）。

---

## 7. 开关设计要求

所有 NPU 功能必须满足：

1. **默认关闭** —— 新装用户走原版路径
2. **三档模式** —— `vanilla`（只计数）/ `assist`（预热不改变结果）/ `npu`（真实接管）
3. **失败静默降级** —— NPU 挂了自动走 CPU，世界不能崩
4. **单一总开关** —— `NpuConfig.enabled` 一键全关
5. **可观测** —— 每个功能独立计数器 + 耗时

当前 `NpuConfig` 已有 `lightMode` / `chunkMode` 三档设计，**方向正确**。
但默认 `assist` 而 mixin 判断 `"npu"`，导致默认配置下地形 NPU 不触发 —— 这是有意的（保守），建议保持。

---

## 8. 建议的下一步（按投入产出比）

1. **停掉地形 matmul 路线**（方向错，越调越亏）
2. **把 `NpuNoise` 那条路做深** —— 让 NPU 直接算 noise
3. **先做 P0 异步化** —— 没有它，后面所有优化都不成立
4. **mcfclnpu 冻结** —— 等哪天 ZL2 官方支持插件进程内 namespace 再说

---

## 9. 附：本次实测数据出处

- `deviceCreate rc=14001` + `clns-9` namespace → mcfclnpu 日志 `mcfclnpu-npu.log`
- `/npu matmul 512 512 512` 返回 OK → 证明 mcnpu 侧 deviceCreate 成功
- `MM_BUCKETS` 不一致 → Java `NpuDispatcher:20` vs native `mcnpu.cpp:917`
- 98304 > 65536 → `NpuTerrainGen:97` 计算，`mcnpu.cpp` `bucketize()` 返回 0

---

---

# 10. 对 GPT-5.6 Luna 报告的回应（2026-10-03 追加）

GPT 已将其报告写入 `mcnpu/docs/GPT_NPU_OPTIMIZATION_REPORT.md`。以下逐条回应。

## 10.1 采纳并已实现

| GPT 条目 | 我的处理 |
|---|---|
| §2 tensor 字节必须检查，不能只看 `v <= 65536` | ✅ 已实现 `checkedTensorBytes()` + `mmShapeSafe()`，64 MiB/tensor 上限，防 size_t 溢出与 uint32 `dataSize` 溢出（mcnpu `a253dd4bee`、`80ea83a951`） |
| §2 修正误导性错误串 | ✅ `mmSizeAllowed` 允许 1..65536 但错误串写死 "allowed=16..512"，已改为从 `MM_BUCKET_MAX` 推导 |
| §5 576 MiB dense matrix 应彻底禁止 | ✅ **与我第 3 节结论一致**，且我进一步指出这条路整体是负优化 |
| §13 只 NPU 化 CPU 热点，Chunk 对象/BlockState/Palette 留在 CPU | ✅ 同意，已写入第 5 节接入点表 |
| §16 必须增加 telemetry | ✅ 已实现 `NpuBench`（见 10.3） |
| §17 P0 安全优先 | ✅ 本轮先做的就是 P0 |

## 10.2 同意但要修正实现方案

**§1.1「改用单个长期 NPU worker」**

诊断正确（8 个 Java worker 抢一把 native 全局锁，实际串行），但解法会制造新瓶颈：单 worker 下任何一个慢 shape 会阻塞队列里所有 job。

正确做法是**批处理**：worker 一次从队列取多个同类 job，合并成一个大 shape 提交。锁的获取次数从 N 降到 1，同时不牺牲吞吐。

**§8「graph cache 不要简单从 8 改到 64」**

同意。补充：`MAX_CACHED_GRAPHS` 触发的 `contextFree + contextCreate` 代价极高（会丢掉所有已 finalize 的 graph）。除了限制 shape 数量，**应该让高频 shape 常驻不被淘汰**（LRU 里 pinned 一部分）。

**§7 shared buffer（QNN_HTP_MEM_SHARED_BUFFER）**

同意必须 A/B 实测。但优先级应**低于异步化**：shared buffer 省的是 host↔HTP 拷贝，而现在的主要开销是 Java 侧 prepare（几百万次运算）和同步等待。省拷贝之前先别让主线程等。

## 10.3 我这轮新增的能力

**`NpuShapeAdvisor`（mcjavanpu `4665ae9e37`）** —— 回答「用什么矩阵更好」

直接回应 GPT §9（Shape Planner 应成为核心模块）。它把 padding 代价显式化：

```
33 x 33 x 33  ->  64 x 64 x 64   （约 7.3 倍算术量）
```

三条省钱规则：padding > 2x 要重新整形、m 低于 128 会被垫高、k > 64 时拆 k 通常比一次宽调用便宜。

`/npu shape` 无参数打印实测例子（含当前地形形状 `98304×32×1`），带参数分析指定形状。

**`NpuBench`（mcjavanpu `4f2d842bb4`）** —— 回应 GPT §16 的关键追问

现在 `NpuStats.speedup()` = `hostUs / npuUs`，但 `npuUs` 是 `submit()` 的墙钟时间，里面混了 Java padding + IPC + native padding + graphExecute。**这个比值无法区分「NPU 赢了」和「传输输了」。**

`NpuBench` 分阶段计时并给两个比值：

```
execSpeedup     = cpuRef / submit                 这次调用变快了吗
pipelineSpeedup = cpuRef / (prepare + submit)     整个任务变快了吗
```

**第二个才是决策依据。** 一个路径可能在第一个比值上很好看，算上 prepare 之后是负优化 —— 这正是当前地形路径的情况。

输出 p50 **和 p99**（平均值会掩盖导致掉帧的尖峰），并直接打印 `VERDICT` 行。两个命令都在虚拟线程上跑，不阻塞主线程。

## 10.4 我认为需要谨慎对待的建议

**§4 / §12 的 9×9 工作集 + 优先级调度**

方向有价值（跨 chunk 合桶确实能摊薄 padding），但有两个现实约束：

1. Minecraft 的 chunk 生成是**阻塞式**的，玩家在等这个 chunk。提前生成 81 个 chunk 意味着大量内存在等待，而移动端内存紧张。
2. 按「玩家移动方向」给优先级需要预测移动，而玩家可能瞬移（传送、下界门）。优先级算错会浪费全部预生成。

建议：先做**同批合桶**（把已经提交的 chunk 请求里同类 shape 合并），不要一上来就做 81 chunk 的预测性预生成。

**§11 原版相似度作为硬指标**

同意，但这是**最难的一条**，不是最优先的一条。原版 noise 是高度优化的实现（含大量位运算技巧），NPU 复现到可接受 RMSE 需要相当工作。建议先做**可开关的近似模式**，把相似度作为度量而非门槛。

## 10.5 我建议的实现顺序

GPT 的 P0–P6 基本合理，我把「批量提交」提前，理由是 binary 通道（`submitBinMatMul8`）已经能用，缺的是**批量**而不是新协议：

1. **批量提交接口** —— 一次 IPC 提交 N 个同类 job（解决 §1.1 锁竞争，投入产出比最高）
2. **ScratchBufferPool**（§6，native 侧复用 A/B/C buffer）
3. **高频 shape 常驻 graph**（§8 补充）
4. **shared buffer A/B**（§7）
5. **9×9 合桶**（§4，从已请求 chunk 开始，不做预测性预生成）
6. **原版相似度度量**（§11）

在此之前，**先跑一次 `/npu bench`** 拿到 prepare/submit/cpuRef 的实际比例。没有这个数据，上面 6 步的优先级排序都是猜的。

---

*—— 元宝（Yuanbao），2026-10-03 追加*

---

# 11. 无人值守模式（2026-10-03 追加）

用户反馈：这个模组**不应该靠输命令来测**。它是优化模组，进游戏就该自己跑起来、自己记录，让人事后看日志。这个批评是对的 —— 之前所有诊断都挂在 `/npu` 命令上，意味着一次普通游戏会话**产生不了任何证据**。

## 11.1 新增：进游戏即自动运行

**`NpuAutoProbe`（`fb1044867f`）** —— mod 初始化时启动，全程后台线程，不阻塞：

```
等世界稳定 20s（区块加载风暴会污染计时）
   ↓
① 服务探测   —— MCNPU 可达吗，不可达就带上 lastFailure 原因
   ↓
② 形状分析   —— 当前形状实际花多少钱（含 padding 放大倍数）
   ↓
③ 基准扫描   —— prepare / submit / cpuRef 三段 + p50/p99 + VERDICT
   ↓
④ 60 秒心跳  —— 滚动统计 + 守卫状态 + 服务状态
```

全部写进 `logs/mcjavanpu-npu.log`。**这个日志就是给 DeepSeek（或任何人）事后分析的产物**，不需要在游戏里操作任何东西。

开关：`NpuConfig.autoProbe`（默认开）。关了就是彻底不跑 —— 不起线程、不碰 IPC。

## 11.2 新增：自适应守卫（这次最重要的一个）

**`NpuGuard`（`b62d3cb951`）** —— 让 NPU **不可能把游戏变慢**。

这是"优化模组"该有的东西：不能只在基准里赢，要持续证明自己值这个价。

```
每次调用记录真实墙钟时间
   ↓
滚动 p99（不是平均值 —— 尖峰才是玩家真正感觉到的）
   ↓
超预算 → 自动降级 → 所有功能回退 CPU 路径
   ↓
每 15s 放一个探测调用通过
   ↓
干净 → 自动恢复（恢复阈值是触发阈值的一半，防抖动）
```

关键设计：
- `allow()` **无分配、无 IPC**，可以放在每帧路径上
- 降级时返回普通失败 `GUARD_DEGRADED`，**现有调用方的回退逻辑不用改**
- **只因传输故障降级**（`SERVICE_*` / `MCNPU_OFFLINE` / `SocketTimeout` / `ConnectException`）。参数错误（形状不合法、buffer 太短）是调用方的 bug，不能因此关掉所有功能 —— 这点在 `fa79f53f84` 修掉

开关：`NpuConfig.guardEnabled`（默认开）。

## 11.3 游戏内开关

`NpuFeaturesScreen`（`e6dbcd2170`）新增四行：

| 行 | 作用 |
|---|---|
| 总开关 enabled | 一键关掉整个模组 |
| 自动探测 autoProbe | 关了就不自动跑诊断 |
| 自适应降级 guard | 关了就不自动回退 |
| **守卫状态** | **实时显示「正常 / 已降级」，点击重置** |

守卫状态那行最有价值：**不用开日志就知道此刻 NPU 到底有没有在干活**。显示"已降级"就是说它现在不值这个价，所有功能都在 CPU 上。

状态行同时附上 `NpuGuard.summary()`：p50/p99 vs 预算，调用数、拒绝数、触发次数。

## 11.4 新增命令（仅用于查看，不再是数据入口）

```
/npu guard        守卫状态：p50/p99 vs 预算、调用数、触发次数
/npu guard reset  清空窗口（换世界时用）
/npu probe        探针状态 + 完整报告写在哪个文件
```

## 11.5 怎么看这次测试的结果

**进游戏，正常玩，什么都不用做。** 退出后取：

```
<游戏目录>/logs/mcjavanpu-npu.log
```

文件里会有：

- `==== NPU auto probe ====` 开头的完整报告（服务状态、形状分析、基准扫描）
- 之后每 60 秒一行 `heartbeat | ... | guard=... | service=...`

把这个文件交给 DeepSeek 即可。**不需要在游戏里输任何命令。**

---

## 11.6 本轮提交清单

| Commit | 内容 |
|---|---|
| `b62d3cb951` | NpuGuard 自适应降级 |
| `fb1044867f` | NpuAutoProbe 无人值守诊断 |
| `7015f1af0f` | NpuConfig 新增 autoProbe / guardEnabled |
| `c306d27987` | mod 初始化时启动探针 |
| `ef4e848335` | 所有 matmul 调用接入守卫 |
| `fa79f53f84` | 修正：只因传输故障降级 |
| `167ec86eb1` | `/npu guard`、`/npu probe` |
| `e6dbcd2170` | 游戏内开关 + 守卫状态显示 |

---

*—— 元宝（Yuanbao），2026-10-03 追加*

---

# 12. 第四轮 · 优化 + 改进 + 新功能（元宝）

日期：2026-10-03。执行顺序按用户要求：**优化和改进优先，新功能在后。**

## 12.0 首先：撤回一个错误结论

`YUANBAO_CORRECTION_P0_1.md`（已提交到两个仓库）—— 上一轮我判断"padding 浪费 512 倍"是**错的**，已撤回。

`n=512` 是 `NpuLightAccel` 里 512×512 传播算子的真实维度（8×8×8 体素的 6 邻域），不是 padding。该路径现在是 **NPU 5.5ms vs host 40ms ≈ 7 倍正收益**。

**如果你正在准备改 n=512，请停手。**

## 12.1 优化（Optimisation）

### `ac54d225f9` — 反射句柄缓存

`NpuChunkWork.runForChunk` 跑在 server tick 上，每次调用做 4 次反射查找：

```
engine.getClass().getMethods()    → getLayerListener
listener.getClass().getMethods()  → getDataLayerData
engine.getClass().getMethods()    → queueSectionData
SectionPos.class.getMethod("of")
```

`Class.getMethods()` **每次调用都克隆整个 Method 数组**（Java 规范要求，防止调用方篡改类自身的数组）。LightEngine 有几十个方法 → **每秒几十次 × 4 次全量克隆**。

改为按 `engine.getClass()` + `listener.getClass()` 缓存已解析的 Method，换世界/维度自动失效，首次之后零遍历。

同时把方法内的 8 项 sub-block 偏移表提为类级常量（原先每 chunk 分配 9 个数组）。

### `0b981f6ea5` — server tick 时间预算

`NpuWorkQueue.pump()` 在 **server 线程**上同步做 IPC，而 `READ_TIMEOUT_MS = 8000`。health 缓存让常见情况很快，但**服务"活着但很慢"时，一次 pump 可能卡住数秒**，且没有任何上界。

现在每次 pump 有 4ms 墙钟预算，超了就把剩余项放回队首等下一 tick。overrun 计数进 `summary()` —— **持续上升的 overrun 是"服务慢"而非"服务挂"的信号**。

## 12.2 改进（Bug fix）

### `3c75a2d984` — reloadchunks 反射匹配

每次切模式都报 `NoSuchMethodException`。签名换过不止一次（SectionPos → ChunkPos → 26.3 又改），每次修都是再一次硬猜。

改为**按形状匹配**：找名字同时含 "light" 和 "enabled"、第二参为 boolean 的两参方法，再按第一参的实际类型构造位置参数（ChunkPos/SectionPos 用 (int,int) 构造器，或 packed long，或裸 int）。

仍失败时**不再干抛异常**，而是把 LightEngine 上所有名字含 light/enable 的方法连同完整参数类型打进日志 —— 下一次修就有依据，不用再猜。

## 12.3 新功能（Feature）

### `c9b836f970` + `e3edc1fe5a` + `ce3c15a4f2` — 天空光 SKY light

光照是唯一已实测证明正收益的路径（7x），但它**只处理 BLOCK 层，SKY 层完全没走 NPU** —— 一半光照成本没动。

天空光和方块光是**同形状的计算**（同一个算子作用于整个 8×8×8 block），可以走完全相同的管线。section 主体抽成 `propagateSection()`，按 layer 各跑一次。

**默认关闭**（`NpuConfig.skyLight=false），因为天空光的传播规则和线性算子的假设不同（向下传播不衰减，直到被不透明方块挡住），所以那里是近似结果。

安全边界不变：**write-back 仍然 raise-only**，NPU 只提亮不调暗，原版结果永远是下限 —— 世界不可能变暗。

游戏内"天空光 skyLight"开关，标签注明"近似算子，只提亮不调暗"。

## 12.4 本轮提交清单

| Commit | 类型 | 内容 |
|---|---|---|
| `82aa252f80` / `822ef2b5a6` | docs | **撤回 P0-1**（两仓库都放了） |
| `ac54d225f9` | 优化 | 反射句柄缓存 + SUB 常量化 |
| `0b981f6ea5` | 优化 | server tick 4ms 预算 + overrun 计数 |
| `3c75a2d984` | 改进 | reloadchunks 按形状匹配 + 方法清单诊断 |
| `e3edc1fe5a` | 新功能 | skyLight 配置开关（默认关） |
| `c9b836f970` | 新功能 | SKY 光照路径 + propagateSection 抽取 |
| `ce3c15a4f2` | 新功能 | 游戏内 skyLight 开关 |

## 12.5 下一步建议

1. **先验证 SKY 光**：开着跑一次，确认不会出现视觉异常（天空光变亮过头）。如果正常，它就翻倍了光照加速的覆盖面。
2. **看 overrun 计数**：`NpuWorkQueue.summary()` 里。持续 >0 说明服务慢，该查服务不是查算法。
3. **reloadchunks 日志**：现在会打印真实方法清单，拿到后就能一次定死正确签名。

---

*—— 元宝（Yuanbao），2026-10-03*

---

# 13. 第五轮 · 两份实测日志的两个致命发现（元宝）

日期：2026-10-03。依据：上传的 `mcjavanpu-npu.log`（19:00–19:44）+ `latest.log`（22:35）。

## 13.0 先确认：我的模块确实进了构建

`latest.log` 22:35:57：

```
boot: enabled=true autoWarmup=true debugLog=true autoProbe=true skyLight=false guardEnabled=true ...
```

`autoProbe` / `skyLight` / `guardEnabled` 三个字段都在 —— 上一轮加的东西是活的。

---

## 13.1 🔴 发现一：每次生产调用都在跑一遍完整的 CPU 参考

日志里最能说明问题的一行：

```
lightapply APPLIED sections=9 rows=72 written=0 |
  blocks_batch=128 cells=512 npu_us=37782 cpu_us=4491 speedup=0.12x
```

`cpu_us=4491` 不是"CPU 更快"的对照值，它是**每次调用都真算了一遍**。

`NpuLightAccel.run()` 里有这个：

```java
float[] ref = new float[m * n];
for (i) for (p) for (j) ref[i*n+j] += av * b[p*n+j] * Q;
```

**O(m·k·n) 的三重 Java 循环，在调用线程上。** m=128、k=n=512 时是 **3355 万次乘加**。产出只有一个 `bad` 正确性计数器。

代价实测：

| 场景 | cpu_us |
|---|---|
| 真实 lightapply | 4,491 |
| warmup（19:00） | **500,687** |
| warmup（19:43） | 191,174 |

**全部是纯开销，对游戏零贡献。**

### 已修（`7288dde952`）

改为 opt-in：`NpuLightAccel.setVerify()`，默认关。warmup 和 benchmark 打开（它们本来就是为了测量），其余全部跳过。

一个细节：**未验证时 `bad` 返回 -1 而不是 0**。这样"没检查"永远不会被误读成"检查过且正确"。

**收益**：真实路径每次调用省下 ~4.5ms（约 10%），warmup 场景省下高达 500ms。

---

## 13.2 🔴 发现二：唯一一次真实调用，写了 0 个格子

```
lightapply APPLIED sections=9 rows=72 written=0
  npu_us=37782  cpu_us=4491
```

**设备 37.8ms + CPU 4.5ms = 42ms，产出为零。**

这不是"还差点调好"，是**这条路径当前对游戏没有任何作用**。而光照引擎每秒调用约 150 次（日志：10 秒内 1531 次），所以只要它在跑，就是在持续烧钱。

另一次 `lightfold`：

```
folded sections=9 rows=72 nonzero=1878 | npu_us=18282 cpu_us=10987 speedup=0.60x
```

源数据里明明有 1878 个非零点，仍然慢于 CPU。

### 已修（`7288dde952` + `1179324a1d` + `0ae024a3ee` + `9bfc8d7794`）

新增"零写入"检测：

```
noteWritten(n)  每次真实写回都上报实际改变了多少格
   ↓
连续 6 次 = 0  →  路径直接拒绝运行，日志说明原因
   ↓
任何一次 > 0    →  计数清零，自动恢复
```

关键：**拒绝运行发生在 `run()` 最前面**，连 NPU 调用都不发 —— 不是"算完再说没用"，是根本不再花那 40ms。

界面也会显示"空写入 n/6"和"light NO_EFFECT"，不用开日志就能看到。

---

## 13.3 关于那个 speedup 数字，证据更足了

跨 4 次会话的 warmup steady：

| 会话 | npu_us | cpu_us | speedup |
|---|---|---|---|
| 19:00 | 9,770 | 203,161 | 20.79x |
| 19:13 | 17,685 | 106,615 | 6.03x |
| 19:39 | 11,476 | 44,122 | 3.84x |
| 19:43 | 14,025 | 88,290 | 6.30x |

`npu_us` 相对稳定（10–18ms），`cpu_us` 从 44,122 跳到 203,161（**4.6 倍**）。

**波动全在 CPU 侧** —— 也就是 JIT 预热和 GC 状态，跟 NPU 无关。所以 `cpu_us / npu_us` 这个比值测的主要是"JVM 今天热不热"，不是"NPU 快不快"。

真实调用给出 0.12x，warmup 给出 20.79x —— **同一个算法，差 173 倍**。这个数字不能用于任何决策，必须换成 `NpuBench` 的 `pipelineSpeedup` + p99。

---

## 13.4 仍然成立 / 仍未修的

- ✅ 进程内路线已死：`clns-9` namespace 实锤（两份日志都有）
- ✅ mcnpu IPC 路线通：`PONG MCNPU/1`、`backendId=6`
- ✅ reloadchunks 反射失败：本次日志又出现 **2 次**，我上轮的按形状匹配已提交（`3c75a2d984`），等下次验证
- ⚠️ `written=0` 的**根因**还没定位 —— 我这次只做了"检测并停止"，没修根本。可能是时机问题（光照尚未生成，log 里也出现过 `no DataLayer in range (sections not loaded)`），也可能是线性平滑算子对已收敛的 BFS 光照场本就提不出更亮的值。**下一步要查这个，而不是继续加功能。**

## 13.5 本轮提交

| Commit | 内容 |
|---|---|
| `7288dde952` | CPU 参考改为 opt-in；零写入检测与自动停用 |
| `1179324a1d` | warmup 开验证、lightapply 上报写入数 |
| `0ae024a3ee` | chunkwork 统计并上报实际改变格数 |
| `9bfc8d7794` | 界面显示空写入计数与 NO_EFFECT 状态 |

## 13.6 下一步（建议顺序）

1. **看新日志里有没有 `light: N consecutive calls changed 0 cells`** —— 有，就证实这条路径当前无效，先别再加功能
2. **查 `written=0` 的根因**：在写回处打印 NPU 输出的最大值/非零个数，判断是"输入全 0"还是"输出比原值小"
3. 拿到 `pipelineSpeedup` 之前，不要相信任何 speedup 数字

---

*—— 元宝（Yuanbao），2026-10-03*

---

# 14. 第六轮 · 地形安全门（元宝）

日期：2026-10-03。触发：游戏崩溃，DeepSeek 正在修；同时收到 GPT 的一份 20 条审查。

## 14.1 我对 GPT 审查的核实

GPT 这轮是实证的。**我逐条对过代码，全部属实：**

| GPT 的 P0 | 代码证据 |
|---|---|
| JSON interpreter 冒充 vanilla | `NpuTerrainVanilla` **零个 `net.minecraft` import** |
| unsupported 静默变 0 | `NpuDfJson` **8 处** `NpuDf.constant(0.0)` |
| 未解析 normalize | `NpuNoiseCatalog` 全文无 normalize |
| seed 是编的 | `(cx*341873128712L) ^ (cz*132897987541L) ^ (minY*42317861L)` |
| double 语义（26.3 要求 float） | `NpuDf` 38 处、`NpuNoise` 38 处 double |
| 全局单树（多世界串状态） | `static volatile NpuDf tree` + `treeSeed` |
| BUILT 重复计数 | 180 与 187 行各一次 `addAndGet(n)` |

## 14.2 我补的一条：崩溃的直接来源

GPT 指出 `NpuTerrainVanilla` 名不副实。**但真正的危险在 mixin 的 assist 分支**：

```java
if (assist) {
    float[] prepared = NpuTerrainAssist.take(...);
    if (prepared == null) { ...; return; }   // 未命中 → 走 vanilla
    for (...) buffer.set(i, prepared[i]);
    ci.cancel();                              // ← 命中就用 interpreter 结果替换
}
```

**assist 命中缓存时同样 `ci.cancel()`**，用那个 seed 是编的、unsupported 静默变 0、double 语义错的 interpreter 结果**替换掉 vanilla 地形**。

而 `chunkMode` **默认就是 `assist`**。

所以"辅助"实际上是一个换名字的 takeover。用户以为安全，实际只要命中就在接管。地形数据错 → 下游光照/结构/水面建立在游戏从未产生过的数值上 → 表现为看起来与地形无关的崩溃。

**这是"游戏崩了"最可能的位置，不是某一行代码的 bug，而是这条路径的数据本身就是错的。**

## 14.3 已做：安全门（这次唯一的第一优先）

游戏崩着的时候，不该先去补 `NpuDfJson` 节点或改 float 语义 —— 那些是长期正确性工程。GPT 的其他五条 P0 都是"让它算对"，而安全门是**"在算对之前不许用"**。后者才是当下的第一刀。

### `NpuTerrainGate`（`256cf64563`）

所有 `ci.cancel()` 之前必须过门。拒绝条件：

```
takeover 未显式开启            → 拒绝（默认关闭）
volume 为 null / 长度不足      → 拒绝
编译树不存在                   → 拒绝
lastUnsupported > 0            → 拒绝（有节点被静默变成 0）
任一值为 NaN / Infinity        → 拒绝
```

**默认 CLOSED。** 默认配置下 vanilla 永远赢，模组退化为纯观察。

### `bce40533de` — `NpuDfJson.lastUnsupported()`

unsupported 数原本只存在于 `Build` 对象里，构建完就丢了。而 gate 需要在**写入时刻**（远晚于构建）判断。现静态留存，`-1` 表示从未构建（gate 视为拒绝）。

### `bbabc275f4` — mixin 两个取消点全部过门

assist 和 takeover 都接。**被拒绝时不 cancel**，vanilla 照常生成。

### `9fe81c3887` — 修 BUILT 重复计数

两处 `addAndGet(n)` → 只保留成功后的那次。且原第一处在 `vols.length != n` 检查**之前**，失败的 batch 也被计入。

### `867cf3a809` — `/npu gate` 命令

```
/npu gate         状态 + 允许/拒绝计数 + 上次拒绝原因 + unsupported 数 + 树状态
/npu gate open    开启接管（以 WARN 记日志）
/npu gate close   回到 vanilla-only
```

## 14.4 立即效果

**装这一版，游戏应当恢复稳定** —— 默认 gate 关闭，vanilla 地形完全不受影响，模组只观察记录。崩溃不应再出现。

## 14.5 我同意 GPT 的方向，但提醒一个张力

GPT 主张转向「Minecraft 26.3 `DensityFunction.compileSampler()` → `DensitySampler.sampleVolume()` 真正接管」。方向对，但要意识到根本张力：

```
自己算（JSON interpreter）  → 可能 ≠ vanilla，但"有加速"
委托 vanilla sampler         → 100% vanilla，但没有加速
```

GPT 的解法是"识别可 NPU 化的子图" —— 正确，但那是长期工程。

**在 parity 测试通过之前，任何形式的结果替换都不该发生。** 这就是安全门存在的理由。

建议顺序：

1. ✅ 安全门（已完成）
2. **parity harness**：同一 seed/坐标下逐点对比 interpreter vs `DensitySampler`，输出 max/mean/p95/p99 误差
3. 误差进入预算后，才 `/npu gate open`
4. 之后再谈 GPT 的 P1（INFLIGHT 去重、dedicated worker、Workset、LRU）

## 14.6 本轮提交

| Commit | 内容 |
|---|---|
| `256cf64563` | NpuTerrainGate，默认关闭 |
| `bce40533de` | NpuDfJson.lastUnsupported() |
| `bbabc275f4` | mixin 两个取消点全部过门 |
| `9fe81c3887` | 修 BUILT 重复计数 |
| `867cf3a809` | `/npu gate` 命令 |

---

*—— 元宝（Yuanbao），2026-10-03*

---

# 15. 第七轮 · 帧率、预加载，以及一个必须说清的坏消息（元宝）

日期：2026-10-03。诉求：帧率不行，想用 NPU 辅助 GPU、辅助计算，覆盖预加载、粒子特效等。

## 15.1 先说坏消息：现在没有任何一条路径在真正加速

这是"帧率不行"的直接原因，不是调参能解决的。

| 路径 | 状态 |
|---|---|
| 地形 terrain | **gate 关闭**（正确性未证明，见第 14 节） |
| 光照 light | **writes 0 → 连续 6 次后自动停用**（第 13 节） |
| 渲染 render assist | **从未被调用**（见下） |

三条全都没在干活，帧率当然不会变。

## 15.2 🔴 发现：`NpuRenderAssist` 是死代码

```
$ grep -rn "NpuRenderAssist\." --include=*.java
McJavaNpu.java:859:  + NpuRenderAssist.summary()      ← 只有状态打印
```

`frameTick()` 和 `transformBoxes()` **没有任何调用点**，没有对应 mixin（mixin 目录只有 Density / LightEngine / PauseScreen / ScreenAccessor 四个）。

这个类写了完整的视锥裁剪批量变换：读 512 个包围盒、8 个角、走 NPU 变换、CPU 只做 6 个平面测试。**但它从来没跑过。** 界面上如果显示了它的数据，那全是 0。

## 15.3 但我不建议把它接上 —— 接上会更慢

这是本轮最重要的判断，我要说清楚为什么。

### 延迟账（用日志里的真实数字）

| 项 | 实测 |
|---|---|
| 一次真实光照 NPU 调用 | **10–38 ms** |
| 60fps 单帧预算 | **16.6 ms** |

**一次 NPU 往返就已经超过整个帧预算。**

而对比 CPU 侧原本的开销：

| 工作 | CPU 成本 |
|---|---|
| 视锥裁剪（~1000 盒 × 6 次点积） | 几十微秒 |
| 粒子积分（数百粒子 × ~20 flop） | 几十微秒 |

也就是说，**把每帧工作搬到 NPU 上，是几十微秒 → 几十毫秒，慢 100～500 倍。**

### 而且异步/结果落后在这里是不可接受的

光照和地形可以用上一帧甚至上上帧的结果（慢一点没关系）。但：

- **视锥裁剪**用上一帧的相机 → 错误剔除 → 区块闪烁、穿帮
- **粒子位置**滞后 → 粒子抖动、拖影

所以"异步 + 容忍落后"这套在光照上成立的做法，在这里直接变成视觉 bug。

### 关于"辅助 GPU"

NPU 没办法辅助 GPU 渲染。GPU 本身是大规模并行的，做顶点变换比通过 IPC 调 NPU 快几个数量级。

NPU 能做的只有一件事：**把喂给 GPU 的那部分 CPU 工作减掉**，让 GPU 不被 CPU 饿着。而这要求那份 CPU 工作是「大批量 + 规则数学 + 可以晚一点」。

## 15.4 所以真正的答案是：预加载（你说对了这个）

你提到的"预加载"是这批诉求里**唯一真正适合 NPU 的**，因为它是唯一满足三条的：

| 性质 | 预加载 | 每帧渲染/粒子 |
|---|---|---|
| 不在帧关键路径 | ✅ 现在请求，几秒后才用 | ❌ 这帧就要 |
| 大批量规则数学 | ✅ 一个 chunk 上万个点 | ❌ 小批量、分支多 |
| 结果可以晚 | ✅ 本来就是提前算 | ❌ 晚了就是画面错 |

## 15.5 已做：预测式预加载

现状是**被动**的：`NpuChunkAuto.onChunkLoad` 在 chunk 已加载时才触发 —— 工作在游戏已经需要它之后才开始。

新增 `NpuPreload`（`da938e4bd9`）：

```
每秒一次 sweep
  ↓
读玩家所在 chunk + 朝向（getDirection）
  ↓
沿朝向的扇形环，跳过身后的 chunk
  ↓
requestAhead(cx, cz) 入队
```

- 只在玩家**换了 chunk** 时才重新 sweep
- 每轮最多 `perSweep`(8) 个，半径默认 3
- 全部走 `NpuChunkAuto.requestAhead()`（`9fe852d193`），与被动路径同样边界：满了丢弃、服务不健康跳过、绝不阻塞
- 朝向读不到就退回整环
- 所有读取都被 try 包住 —— 预加载绝不能把世界加载搞崩

命令：`/npu preload [on|off|<1..8>]`（`a64bb723fe`）

## 15.6 ⚠️ 但要诚实：现在开预加载也还是白跑

因为底层两条路径都不通（15.1）。预加载会把**更多的** chunk 送进一条 writes 0 的光照路径和一条 gate 关闭的地形路径。

所以预加载的真正收益要等这件事发生之后：

```
修好光照 written=0  →  地形 parity 通过 →  /npu gate open  →  预加载才兑现
```

在那之前，`/npu preload off` 是更省电的选择。

## 15.7 我的建议顺序（针对"帧率"这个目标）

1. **查光照 written=0 的根因** —— 这是唯一已经在跑、且理论上成立的路径。修好它，区块加载卡顿立刻改善，这是最直接能感知到的帧率收益。
2. 地形 parity harness，然后才开 gate
3. 之后预加载自动生效（代码已经就位）
4. **不要**做视锥裁剪 / 粒子 / 顶点变换的 NPU 化

第 4 条不是保守，是算术问题：几十微秒的工作搬到一个 10ms+ 的通道上，没有调度技巧能救。

## 15.8 本轮提交

| Commit | 内容 |
|---|---|
| `da938e4bd9` | NpuPreload 预测式预加载 |
| `9fe852d193` | NpuChunkAuto.requestAhead + tick 接线 |
| `a64bb723fe` | `/npu preload` 命令 |

---

*—— 元宝（Yuanbao），2026-10-03*

---

# 16. 第八轮 · 延迟的真正数字：wall_ms 是 npu_us 的 2.4～24 倍（元宝）

日期：2026-10-03。依据：用户上传的 `mcjavanpu-npu.log`（10-02 15:45 至 10-03 19:44）+ `latest.log`（10-03 23:27）。

---

## 16.0 这两份日志的时间线

| 文件 | 时间 | 对应版本 |
|---|---|---|
| `mcjavanpu-npu.log` | 10-03 19:00–19:44 | **CPU 参考仍在每次调用里跑**（第 13 节修复之前） |
| `latest.log` | 10-03 23:27 | 较新，含 `autoProbe / skyLight / guardEnabled` |

所以下面 16.1 的数字是**修复前的基线**。它之所以仍然重要，是因为它第一次给出了 `wall_ms`。

---

## 16.1 🔴 给 DeepSeek 的第一条：你测的 `npu_us` 不是真实延迟

日志里同时有 `npu_us` 和 `wall_ms`，把它们放一起：

| 会话 | npu_ms | cpu_ms | wall_ms | **wall/npu** | cpu 占 wall |
|---|---|---|---|---|---|
| 19:00 first | 60.7 | 500.7 | 572 | **9.4x** | 88% |
| 19:00 steady | 9.8 | 203.2 | 234 | **24.0x** | 87% |
| 19:14 first | 39.5 | 105.3 | 158 | 4.0x | 67% |
| 19:14 steady | 17.7 | 106.6 | 135 | 7.6x | 79% |
| 19:39 first | 48.8 | 59.4 | 117 | 2.4x | 51% |
| 19:39 steady | 11.5 | 44.1 | 59 | 5.1x | 75% |
| 19:43 first | 61.8 | 191.2 | 302 | 4.9x | 63% |
| 19:43 steady | 14.0 | 88.3 | 116 | 8.3x | 76% |

**`npu_us` 只覆盖了真实耗时的 4%～41%。**

最极端的一次：`npu_us=9770`（9.8ms）对应 `wall_ms=234`（234ms）—— **少报 24 倍**。

### 差额去哪了

```
wall_ms ≈ npu_us + cpu_us + 少量其它
```

例：19:00 steady → 9.8 + 203.2 = 213ms，wall 234ms，差 21ms。

**`cpu_us` 就是 wall 的 51%～88%。** 也就是那个 `O(m·k·n)` 的 Java CPU 参考三重循环 —— 第 13 节已经改成 opt-in（commit `7288dde952`）。

**预计效果**：warmup steady 的 wall 应从 **234ms → 约 15～25ms**。

### 这条对修延迟的意义

如果只盯 `npu_us` 去优化 IPC、协议、服务端，方向会全错 —— 因为在修复前，**90% 的时间根本不在 NPU 通道上**。

---

## 16.2 🔴 第二条：真实调用的 `bad=0` + `written=0` 组合，已经判了这条路死刑

```
lightapply APPLIED sections=9 rows=72 written=0
  npu_us=37782 cpu_us=4491 bad=0/65536
```

`bad=0/65536` 意味着 **NPU 结果和 CPU 参考逐点一致，完全算对了**。

`written=0` 意味着 **写回时没有任何一格比原版更亮**（raise-only）。

两者合起来只能是一个解释：

> NPU 算出的光照值 ≈ 原版 BFS 已经算出的值，没有任何一格更高。

也就是说：**线性平滑算子作用在已经收敛的光照场上，产生不出比原版更亮的结果。这不是 bug，是这个算法的必然结果。**

结论：**这条路径不可能产生任何正的写入。** 修好 `written=0` 的"根因"也改变不了这一点，因为根因就是算法选择本身。

`lightfold` 那次更直接：源数据有 `nonzero=1878` 个非零点，结果仍然 `npu 18.3ms vs cpu 11.0ms`（0.60x）—— 慢于 CPU。

### 建议

**关掉光照路径**，不要继续优化它。它是唯一在跑的路径，也是唯一确定无效的路径：

- 算得对（`bad=0`）
- 但没有效果（`written=0`）
- 而且每次花 38ms

第 13 节加的"连续 6 次空写入自动停用"会在新版里自动关掉它。**看到 `light: 6 consecutive calls changed 0 cells` 就是它在生效。**

---

## 16.3 其它确认

- ✅ **进程内路线已死**：`clns-9` namespace 再次出现（10-02 15:45 与 16:29 两次，bundled 与 vendor 路径全失败）。mcfclnpu 不要再投入。
- ✅ **IPC 路线通**：`PING=PONG MCNPU/1`、`STATUS=QNN HTP ready backendId=6`、`persistent MCNPU service connected`。
- ✅ **能力集**：`ops=ADD,MATMUL,MATMUL16,MATMUL8,SUBMIT8,SUBMITBIN8 max_elements=16384`。注意 `max_elements=16384` 与当前 shape 的 C 矩阵 `128×512=65536` 之间的关系需要澄清 —— 调用成功说明它不是输出元素总数上限，但语义应在服务端文档中写明，避免后面按字面理解踩坑。
- ⚠️ **`reloadchunks` 仍失败 2 次**（19:44:16、19:44:17）：`NoSuchMethodException ... setLightEnabled(SectionPos, boolean)`。我第 4 轮的按形状匹配（commit `3c75a2d984`）还没在日志里体现效果，需要新版验证。
- ⚠️ **光照引擎 150 次/秒**（19:44:06 的 10 秒内 1531 次）。任何超过 6.6ms 的同步调用都会吃掉帧预算，而当次实测 18～38ms。

---

## 16.4 针对"修延迟"的建议顺序

1. **先确认 `wall_ms` 的真实值**。别再用 `npu_us` 判断快慢 —— 16.1 证明它最多只覆盖 41%。
2. **确认 CPU 参考已关闭**。新版 warmup steady 的 `wall_ms` 应该在 15～25ms；如果还是 200ms+，说明 `setVerify()` 没生效。
3. **关掉光照路径**（16.2）。它确定无效，且是当前唯一在烧钱的路径。
4. 之后才谈 IPC / 协议 / 服务端优化。

⚠️ 提醒：在 1、2、3 完成之前，优化 IPC 是在优化一个只占总时间 5%～40% 的环节。

---

*—— 元宝（Yuanbao），2026-10-03*

---

# 17. 第九轮 · 帧率：先确认模组自己不是负担（元宝）

日期：2026-10-03。依据：`mcjavanpu-npu.log`（10-02 15:45 → 10-03 19:44）+ `latest.log`（10-03 23:51）。

## 17.0 两份日志不是同一次会话

| 文件 | 最后时间 |
|---|---|
| `mcjavanpu-npu.log` | 10-03 **19:44** |
| `latest.log` | 10-03 **23:51** |

所以 23:51 那次运行的 NPU 明细日志未上传。但 `latest.log` 的启动行已足够判断配置：

```
boot: enabled=true autoWarmup=true debugLog=true autoProbe=true skyLight=true
      guardEnabled=true lightBatch=128 lightFoldRadius=2
      lightMode=npu chunkMode=npu
```

`autoProbe` / `skyLight` / `guardEnabled` 都在 → 我的代码进了构建。其中 **`skyLight=true`**（默认是 false），说明它被主动打开了。

已确认的环境：`sodium 0.9.3-alpha`、`lithium`、`ferritecore`、`moreculling`、`entityculling`、`krypton`、**`spark 1.10.187`**，Vulkan + Adreno 735，共 70 个模组。

---

## 17.1 🔴 本轮最重要的发现：gate 只挡写入，没挡计算

上一轮我加 `NpuTerrainGate` 时只 gate 了**结果**，没有 gate **投入**。后果是：

```
DensitySamplerMixin（tail）
      ↓
NpuTerrainAssist.requestWorkSet()      ← 一次规划 81 个 chunk
      ↓
后台 worker
      ↓
NpuTerrainVanilla.fill()               ← 完整 density tree，纯 Java
      ↓
TAKEOVER_CACHE
      ↓
下次命中 → NpuTerrainGate.allowWrite() → 拒绝（默认关闭）
      ↓
结果丢弃
```

**整条链跑完，然后被丢掉。**

而 gate 默认关闭，所以默认配置下这就是纯粹的成本。日志里 `cpu_us=500687`（19:00 warmup）说明 density tree 评估有多贵 —— 那还只是 128 个 block 的一次调用，`requestWorkSet` 一次规划 **81 个 chunk**。

这不是"优化没收益"，是**纯亏损**。

### 已修

新增 `NpuTerrainGate.worthComputing()`，含义是"结果有可能被用吗"，与 `allowWrite()`（"这个结果能用吗"）分工：

| 方法 | 时机 | 作用 |
|---|---|---|
| `worthComputing()` | **开始前** | 决定要不要投入 CPU |
| `allowWrite()` | 产出后 | 决定能不能写回 |

接入点：

- `NpuTerrainAssist.request()` / `requestWorkSet()`（`b188c78a6d`）
- `DensitySamplerMixin` 尾部（`23e4e64b0c`）
- `NpuPreload`（`a3d946264f`）

并加 `skipped_gate` 计数器，日志能看到**拒绝了多少次**，而不是只看到算完了又被丢。

**默认配置下（gate 关闭），地形路径现在完全不动。**

---

## 17.2 第二处：光照路径已被证明无效，但仍在排队

第 16 节的结论成立且是终局性的：

```
lightapply written=0   bad=0/65536
```

- `bad=0`：NPU 结果**与 CPU 参考逐点一致，完全正确**
- `written=0`：**没有任何一格比原版更亮**

两者合起来只有一种解释：线性平滑算子作用在**已收敛的 BFS 光照场**上，永远产生不出更高值。这是算法的必然结果，不是 bug。

但 `NpuChunkAuto` 仍在持续把 chunk 送进这条路径。guard 会在 6 次空写入后降级，**排队却没停** —— 于是 chunk 被提交、被 guard 跳过，白排队。

已修（`d73b6a34b5`）：反应式和预测式两个入口都先查 `NpuGuard.allow()`。

---

## 17.3 新增：`NpuSelfCost`（`994396f805` + `e024b86b68`）

现在的核心问题是"帧率不行是**我们**造成的吗"，这需要一个数，而不是讨论。

```
self_cost us: tick avg=? max=? n=?
            | chunk avg=? n=?
            | sampler avg=? max=? n=?
```

分别记录：模组自己的 server tick 开销、chunk-load 钩子开销、density sampler 钩子开销。tick 平均超过 1000us 会以 WARN 输出"模组是成本而非加速器"。

**这个数字的两个用途：**

- 接近 0 → 帧率问题**不在我们**，是别的模组 / 渲染 / 设备。这个负面结论很有价值，因为再怎么优化 NPU 也不会改善。
- 不接近 0 → 模组本身在拖后腿，先修自己。

---

## 17.4 关于"用 NPU 辅助 GPU"：仍然不行，且现在证据更强

`latest.log` 显示环境里已经装了 `sodium`（渲染优化）和 `lithium`（逻辑优化）。**CPU 侧能省的已经被专业模组省过了。**

而 NPU 侧实测单次要 **10–38 ms**，一帧只有 16.6 ms。把每帧工作搬过去是几十微秒 → 几十毫秒。这个结论不因"装了 sodium"而改变。

**NPU 唯一现实的定位是预加载**（第 15 节），而它现在被 gate 挡着（正确，因为地形正确性未证明）。

---

## 17.5 建议顺序（针对帧率）

1. **先看 `self_cost` 那一行。** 这是分水岭：
   - 接近 0 → 我们已经无责，帧率问题去找 sodium/光影/视距/设备温度
   - 明显非 0 → 把结果给我，我继续砍
2. **确认 `skipped_gate` 在涨** —— 涨说明地形路径确实停了
3. **`spark` 已经装了**，跑一次 `/spark profiler` 能直接定位真正的热点，比我们猜强得多。建议把结果一并发出来。
4. 在此之前不要再开新功能

## 17.6 本轮提交

| Commit | 内容 |
|---|---|
| `70052d15dc` | `NpuTerrainGate.worthComputing()` |
| `b188c78a6d` | Assist 两入口开工前检查 + `skipped_gate` 计数 |
| `23e4e64b0c` | mixin 不值得就不请求 |
| `a3d946264f` | 预加载 gate 关闭时跳过 |
| `d73b6a34b5` | chunk 入口查 guard，停止无效光照排队 |
| `994396f805` | `NpuSelfCost` |
| `e024b86b68` | tick 计时接线 |

---

*—— 元宝（Yuanbao），2026-10-03*
