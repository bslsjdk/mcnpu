# Minecraft 26.3 worldgen → NPU 完整接管链路设计

*元宝 · 2026-10-04*

---

## 0. 先解决读不到代码的问题（已完成）

`is_code_search_indexed: false` 是 GitHub 对私有仓库的常见状态 —— 索引没建，
**Code Search 必然返回 0**，与权限无关。

**绕开方式：Git Tree API**

```
GET /repos/bslsjdk/mcjavanpu/git/trees/main?recursive=1
```

返回 **71 个条目，`truncated: false`**，即完整文件树，无需搜索。
之后用 `contents` API 读单个文件，或 `tarball` 一次性下载。

**不要因为搜索为空就判定"仓库没代码"。**

---

## 1. 当前代码实况（读完之后）

### 接管点已经是对的

`mixin/DensitySamplerMixin.java` 已经挂在：

```
net.minecraft.world.level.levelgen.densityfunction.DensitySampler$Bound
  └─ sampleVolume(DensityBuffer, DensityVolume)   @HEAD cancellable
```

这是 **26.3 真正的 density 入口**。一个 `Bound` sampler 一次填一个 chunk 体积
（`chunkVolume()` 构造 16 × height × 16，overworld 约 98304 点）。
`mcjavanpu.mixins.json` 里 `required: true`、`defaultRequire: 1`，注入失败会直接报错而不是静默失效。

**所以"接入原版 worldgen 链"这一步，现状已经达成。**

### 但"NPU 算的那份"是假的

真正的问题在 `NpuTerrainVanilla`：

```
NpuTerrainVanilla.fill()
   → NpuVanillaJson.densityFunction("final_density")   读 jar 里的 JSON
   → NpuDfJson.buildTree()                             编译成树
   → NpuDf / NpuDfProgram                              求值
   → NpuNoise                                          自己实现 Perlin
```

**全程 Java，零 NPU 调用，零 IPC。**

`grep -c "net.minecraft" NpuTerrainVanilla.java` = **0**。
它是一个"根据 Minecraft 数据文件重写的 density 解释器"，不是 Minecraft 的 density 计算。

`NpuTerrainGate` 默认 CLOSED 是正确的 —— 它挡住的正是不该写进世界的这份结果。

---

## 2. 🔴 核心结论：matmul 后端做不了噪声

这是本次最重要的一条，它决定了整条链路该怎么设计。

### 2.1 mcnpu 现在只有 matmul

服务端命令集：`EXEC_MATMUL` / `EXEC_MATMUL8` / `SUBMITBIN_MATMUL8` / `EXEC_ADD` / `XFORM`。
**没有 noise。**

### 2.2 Perlin 噪声在代数上不是 matmul

Minecraft 的 `NormalNoise.getValue(x,y,z)`：

```
value = Σ_o  amp_o · perlin_o(x·2^o, y·2^o, z·2^o) / norm
```

单个 `perlin` 在某点：

```
v = Σ_{k=0..7}  w_k(f) · ( g_k · (p − c_k) )
        ↑            ↑        ↑
   非线性 fade   每点不同    线性，可成 matmul
   混合权重
```

逐项拆解：

| 步骤 | 能否 matmul |
|---|---|
| `D[m,k] = g_k · p_m` → `P[M,3] × G[3,8]` | ✅ **可以** |
| `E_k = g_k · c_k`（每 cell 常量） | ✅ 可预计算 |
| `w_k(f)` = fade 的乘积组合 | ❌ **逐点非线性** |
| `Σ_k w_k · (D_k − E_k)` | ❌ 逐元素乘 + 归约 |

关键障碍：`w_k` 依赖**每点的分数坐标**，即使同一 cell 内各点也不同；
而 `g_k` 又**逐 cell 不同**，跨 cell 无法共享权重矩阵。

**结论：Perlin 噪声无法用 matmul 表达。**

### 2.3 而噪声恰恰是 worldgen 的主要成本

`final_density` 的叶子几乎全是噪声通道（continentalness / erosion / temperature /
humidity / weirdness / jaggedness…），每个 5–9 个 octave。
剩下的 add / mul / spline / range_choice 是廉价或分支密集的。

**即：最该加速的部分，是当前后端唯一不能加速的形状。**

### 2.4 那插值呢？（唯一 matmul 形状的）

lattice → 全体积的三线性插值确实是稀疏 matmul（每行 8 个非零）。

但按实测：98304 个输出，按 cell 复用后**每输出 1 次乘加** ≈ 0.1 ms。
而一次 NPU submit **≥ 10 ms**。

**搬过去慢 100 倍。**

---

## 3. 真正的解法：在 DSP 上写噪声内核

上面推到死角，但**这条路恰好绕回了我们一直在打的那场仗**。

### 3.1 `libcdsprpc.so` 就是干这个的

`libcdsprpc.so` = **FastRPC** —— 把自定义代码放到 Hexagon DSP 上跑的通道。

我们之前为它打的 `clns-9` namespace 之战、ZL2 的 `uses-native-library`，
目标一直是"让 QNN 能连上 HTP"。但 FastRPC 的本职是：
**跑我们自己写的 DSP 内核**（.so skel），不止是服务 QNN。

### 3.2 于是链路成立

```
Minecraft Java
   ↓  mixin DensitySampler$Bound.sampleVolume @HEAD
mcjavanpu
   ↓  从 volume 取 lattice（stepX/stepY/stepZ，游戏自己的分辨率）
   ↓  批量噪声请求：M 个 lattice 点 × C 个通道
   ↓  IPC（p50 187 µs，很便宜）
mcnpu
   ↓  FastRPC / QNN custom op
Hexagon DSP / HTP
   ↓  Perlin 内核（真正的 SIMD，8 路）
   ↓  返回 M×C 个 float
mcjavanpu
   ↓  在 CPU 上完成 add/mul/spline/range_choice/cache
   ↓  三线性插值到 98304
   ↓  写回 DensityBuffer
Minecraft
   ↓  biome / surface rules / carvers / structures / block placement
完成真实区块生成
```

**分工明确：NPU 只做规则数值（噪声）；分支与状态逻辑留在 CPU。**
这正是用户描述的"Minecraft CPU 继续处理 biome、surface rules、carvers、structures"。

### 3.3 预算

一次 chunk：lattice 5×49×5 = 1225 点 × ~6 通道 = 7350 次噪声求值。
若 DSP 内核单次 submit 能在 **< 5 ms** 完成（对比 CPU 侧数毫秒 + IPC 0.2 ms），
才有净收益。**这就是需要实测的那个数。**

---

## 4. 已实现：噪声占比探针

在设计内核之前，先要知道噪声到底占多少。不知道这个数，
写内核就是赌博。

`NpuNoise` 新增统计（`722f9006b4`、`91e0a913c8`）：

- 1/64 采样计时（避免每次 `nanoTime` 的开销污染测量）
- **per-thread 累加**，4096 次 flush 一次 —— 共享 AtomicLong 每次自增会
  在 worker 线程间反复争用同一条 cache line，正好扭曲被测对象
- `NpuTerrainVanilla.summary()` 末尾输出：

```
noise_calls=? per_call_ns=? noise_us=?
```

与已有的 `eval_ms=` 并列，**两个数的比值就是答案**。

- 噪声占 **< 30%** → 内核收益有限，地形 NPU 路线应关闭
- 占 **> 60%** → 值得让 GPT 写 DSP 噪声内核

---

## 5. 分层职责（最终结构）

```
MC 26.3
   │
DensitySampler$Bound.sampleVolume          ← mixin 已挂
   │
NpuTerrainGate.worthComputing()            ← 开工前问：值不值得算
   │
TerrainWorkset（跨 chunk 分桶）
   │
GraphPlanner
   ├── NPU-friendly：noise octave 批量求值        ← DSP 内核（待 GPT）
   └── CPU-only：spline / range_choice / cache / beardifier
   │
Quantizer（int8 + scale）
   │
NpuQueue（持久化 socket，后台 worker）
   │
MCNPU → FastRPC → Hexagon
```

**必须留在 CPU 的**（不要试图搬）：
spline、range_choice（分支重）、cache 语义、beardifier（结构相关）、
aquifer、任何状态依赖或单点小运算。

---

## 6. 前置门槛（gate 打开前必须全绿）

`NpuTerrainGate` 继续 **CLOSED**。开的条件：

1. `NpuParity` 连续 3 次 clean（`max_abs` 在预算内、`bad=0`、`unsupported=0`）
2. 真实 world seed（不得由 chunk 坐标伪造）
3. 多世界隔离（按 `ServerLevel` 隔离 tree / cache / noise cache）
4. float 语义（26.3 全程 float，当前解释器用 double）
5. `normalize` 三种语义已解析（当前 `NpuNoiseCatalog` 完全没读）
6. unsupported 节点**不得静默返回 0**

第 6 条最危险：不认识的节点变成 0 → 上层 add/mul/spline 继续算 →
世界静默变形，日志只有一句 `unsupported=1`。

---

## 7. 分工

| 方 | 负责 |
|---|---|
| **元宝** | mcjavanpu：mixin / workset / gate / parity / 计量（本轮噪声探针已提交） |
| **GPT** | mcnpu：FastRPC 噪声内核 / QNN custom op / `NOISE_BATCH` 协议 |

**建议协议**（供 GPT 实现）：

```
NOISE_BATCH <seed> <channelCount> <pointCount> <x0> <y0> <z0> <sx> <sy> <sz> ...
→ OK NOISE <pointCount*channelCount floats>
```

Java 侧已有 CPU 参考实现（`NpuNoise`），内核落地后直接对比校验。
内核不存在前，Java 侧走 CPU，**不会变慢**。

---

## 8. 本轮提交

| SHA | 内容 |
|---|---|
| `722f9006b4` | `NpuNoise` 噪声耗时统计（1/64 采样，per-thread） |
| `91e0a913c8` | `NpuTerrainVanilla.summary()` 输出噪声占比，fill 后 flush |

---

*—— 元宝*
