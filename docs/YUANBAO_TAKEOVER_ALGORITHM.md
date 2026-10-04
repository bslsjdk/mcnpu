# NPU 接管世界生成 —— 完整算法与锁定模式

*元宝 · 2026-10-04*

---

# 第一部分：锁定模式（已实现）

## 问题

原来 `chunkMode="npu"` 的语义是：能拿到 NPU 结果就用，拿不到就**回退 vanilla**。

这让性能无法测量 —— 世界会变成"一半 NPU、一半原版"的混合体，
任何计时都无法归属到某条路径。用户说得对：**不锁定就无法知道性能如何。**

## 方案：每个世界锁死一种模式

`NpuTerrainLock`（`397e9ad6`）：

```
第一次 density 采样
    ↓
读世界 seed（NpuChunkWork.worldSeed()）
    ↓
查 worldLocks[seed]
    ├─ 有 → 用它（已冻结）
    └─ 无 → 取当前配置的 chunkMode，写入并持久化
```

- **按 seed 标识世界**：不新增 level-loading 的 mixin，seed 是唯一稳定的句柄
- **持久化**到 `mcjavanpu.properties`（`worldLock.<seed>=<mode>`），重启后仍然锁定
- **菜单改动只影响新世界**，不影响当前世界 —— 日志明确说明
- seed 未知时**推迟锁定**，避免在占位值上留下一个活太久的锁

## 锁定的接管模式：等待，不回退

`mixin/DensitySamplerMixin`（`5d922bd5`）：

| 情况 | 行为 |
|---|---|
| 拿到 NPU 结果 | 写入，vanilla 不跑 |
| 未拿到 | **请求 + 等待**（最多 8s） |
| 等待超时 | 写 **sentinel**，绝不写 vanilla |
| gate 拒绝 | 写 sentinel + 记录原因 |

**sentinel = `-2.0f` 常量**。选它的理由：

- 有限值 → 不会像 NaN 那样污染下游（carvers / surface rules 会炸在无关的地方）
- 明显负 → 区块变成空气，地上出现洞，**一眼看出失败**
- **绝不是 vanilla 的数字** → 失败不会伪装成成功

失败计数 + 日志（`747fd884`）：第 1 次完整打印，之后每 50 次一次。

> ⚠️ 此模式**会卡顿**（worldgen 线程阻塞等待）。它存在的目的是测量，不是游玩。
> 正常玩请用 `assist`。

---

# 第二部分：完整算法

## 0. 结论先行

**Minecraft worldgen 只能"NPU 辅助"，不能"NPU 接管"全部。**

但能做到：**density 计算链中，噪声部分完全由 NPU 承担**，
CPU 只保留分支/状态逻辑。这是可实现的最大接管范围。

---

## 1. 为什么噪声不能走 matmul

### 1.1 噪声是 worldgen 的主要成本

`final_density` 的叶子几乎全是噪声通道：
continentalness / erosion / temperature / humidity / weirdness / jaggedness …
每个 5–9 个 octave。

### 1.2 但 Perlin 代数上不是 matmul

```
v(p) = Σ_o amp_o · perlin_o(p·2^o) / norm
```

单个 perlin 在某点 p（所在 cell 为 c，分数坐标 f）：

```
v = Σ_{k=0..7} w_k(f) · ( g_k(c) · (p − c_k) )
```

| 子式 | 能否 matmul |
|---|---|
| `g_k · p` | ✅ 线性 |
| `w_k(f)` = fade 的乘积组合 | ❌ **逐点非线性** |
| `Σ_k w_k · (D_k − E_k)` | ❌ 逐元素乘 + 归约 |

`w_k` 依赖**每点的分数坐标**（同一 cell 内各点也不同）；
`g_k` 又**逐 cell 不同**，跨 cell 无法共享权重矩阵。

### 1.3 我试过的"按 cell 分组"方案，为什么失败

思路：把点按 cell 分组，组内共享 `g_k / c_k`，于是：

```
D = P[M,3] × Gᵀ[3,8]        ← 组内 G 共享，真 matmul ✅
term2 = W[M,8] × e[8,1]     ← e 组内共享，matvec ✅
v = reduce_sum(w ⊙ D) − term2
```

代数成立。**但分组大小会退化**：

- 低频 octave（continentalness base = -9）：整 chunk 的 lattice 点几乎都在**同一 cell** → 组很大 ✅
- 高频 octave（freq ≈ 0.5，cell 尺寸 2 格）：lattice 间距是 x/z=4、y=8，
  **每个点各自一个 cell** → 1225 组，每组 1 个点 ❌

组大小为 1 时，"matmul" 退化成 1225 次 `1×3 × 3×8`，
开销全部变成调度与启动成本，比 CPU 直接算慢得多。

**结论：matmul 后端做不了 Perlin。必须上自定义内核。**

---

## 2. 正确路径：FastRPC + HVX 自定义内核

### 2.1 libcdsprpc.so 就是为此存在的

`libcdsprpc.so` = **FastRPC** —— 把自定义代码放到 Hexagon DSP 上执行的通道。

我们之前为它打的 `clns-9` namespace 之战、ZL2 的 `uses-native-library`，
目标一直是"让 QNN 连上 HTP"。但 FastRPC 的本职是：
**跑我们自己的 DSP 内核（.so skel）**，不止服务 QNN。

### 2.2 HVX 正好有 Perlin 需要的指令

查证 Qualcomm HVX 指令集：

| HVX 指令 | Perlin 中的用途 |
|---|---|
| `vlut` / `vgather` | **梯度表查表**（表 ≤256 字节） |
| `vmpy` / `vadd` | 点积、fade 多项式 |
| `vfloat2int` | `cell = floor(p)` |
| `vshuff` / `vdeal` | 8 corner 数据重排 |
| 1024-bit 寄存器 | **32×f32 或 64×f16 一路** |

关键：**vanilla 梯度表是 16 个三维向量 = 16×3×4B = 192 字节 < 256 字节**
→ 整张表能放进 `vlut`，一次指令完成 gather。这正好是 Perlin 的瓶颈操作。

---

## 3. 内核算法（给 GPT 实现）

### 3.1 接口

```
输入： seed
      coords[M][3]      lattice 点世界坐标（f32）
      channels[C]       每通道：baseOctave, baseAmplitude, modifiers[], normalize
输出： values[M][C]     f32
```

一次调用覆盖**一个 chunk 的全部 lattice 点 × 全部通道**。

### 3.2 主循环（向量宽度 W = 32）

```
for each channel c:
  load baseOctave, amps[0..O-1], norm
  acc[W] = 0
  for o in 0..O-1:
     if amps[o] == 0: continue            // vanilla 允许 modifier=0
     freq = 2^(baseOctave + o)
     px = coords.x[W] * freq ; py.. ; pz..
     cx = floor(px) ; cy.. ; cz..          // vfloat2int
     fx = px - cx   ; fy.. ; fz..
     // 8 corners
     for k in 0..7:
         hx = cx + (k&1) ; hy = cy + ((k>>1)&1) ; hz = cz + ((k>>2)&1)
         idx = hash(seed, hx, hy, hz) & 15         // 向量整数 hash
         g = vlut(GRAD_TABLE, idx)                 // 192B 表，3 分量
         d[W] = g.x*px + g.y*py + g.z*pz           // 点积
         w[W] = weight_k(fx, fy, fz)               // fade 乘积
         acc += w * d
     ...
  values[][c] = acc * (amps[o] 加权后) / norm
```

`weight_k` 用 vanilla 的 fade：`t*t*t*(t*(t*6-15)+10)`，纯 `vmpy/vadd` 多项式。

### 3.3 关键实现要点

1. **hash 必须逐位复刻 Java 的 Xoroshiro128++/permutation**，
   否则梯度选错 → 地形完全不同。这是正确性第一风险点
2. **f32 语义**：26.3 全程 float。`double` 中间值会逐步偏parity
3. **梯度表常驻 DSP 紧耦合内存**，避免每次 gather 走 DDR
4. **按点并行（32 点一路）**，不是按 octave 并行 —— octave 之间有累加依赖
5. **多通道合并进同一次调用**：一次 FastRPC 往返摊掉固定开销

### 3.4 预算

一个 chunk：lattice 5×49×5 = **1225 点** × ~6 通道 × ~7 octave
= **51,450 次 octave 求值** × ~80 flop ≈ **4.1 MFLOP**

向量化（32 宽）后 ≈ **129k 条向量指令** → HVX 上理论 sub-ms，
实估 **1–3 ms**。

对比 CPU：每 octave 求值 ~100 ns → **~5 ms**（待噪声探针确认）。

> 📌 **这个估算必须被实测替代**。所以先落地了噪声占比探针
> （`722f9006`、`91e0a913`）：`noise_us=` 与 `eval_ms=` 的比值决定要不要动。
> 噪声占比 < 30% → 地形 NPU 路线关闭；> 60% → 值得写内核。

---

## 4. 完整链路

```
Minecraft Java
   ↓  mixin DensitySampler$Bound.sampleVolume @HEAD   ← 已挂上，26.3 真入口
   ↓
NpuTerrainLock.acquire(seed)          ← 锁定模式（已实现）
   ↓
lattice 提取（volume 的 stepX/stepY/stepZ，游戏自己的分辨率）
   ↓
┌──────── NPU 承担（规则、大批量）────────┐
│  NOISE_BATCH → IPC → mcnpu             │
│     → FastRPC → HVX Perlin 内核        │
│     → 返回 1225×C 个 f32               │
└────────────────────────────────────────┘
   ↓
┌──────── CPU 保留（分支/状态）──────────┐
│  add / mul                             │
│  spline（分支重）                      │
│  range_choice（分支）                  │
│  cache 语义（状态）                    │
│  三线性插值 → 98304                    │
└────────────────────────────────────────┘
   ↓
写回 DensityBuffer
   ↓
Minecraft 继续：biome / surface rules / carvers / structures / block placement
```

**为什么后一半必须留 CPU**：spline、range_choice、cache、beardifier、aquifer
都是分支或状态相关，NPU 不擅长，搬过去必然更慢。

**所以"NPU 接管"的诚实含义是：接管 density 数值计算链，不接管区块生成全流程。**

---

## 5. 校验（内核落地后必做）

Java 侧已有 CPU 参考实现 `NpuNoise`。内核落地后：

1. 同一 seed、同一批坐标，分别 CPU / DSP 求值
2. 逐点比对，输出 `max_abs / mean / p95 / p99 / mismatch / NaN`
3. 预算：`max_abs < 1e-4`（f32 舍入量级）
4. 通过后才允许 `NpuTerrainGate` OPEN

`NpuParity` 已有框架，接入即可。

---

## 6. 分工

| 方 | 负责 |
|---|---|
| **元宝** | mcjavanpu：锁定模式 ✅、mixin ✅、噪声探针 ✅、parity、workset |
| **GPT** | mcnpu：`NOISE_BATCH` 协议 + HVX 内核 + FastRPC skel |

**建议协议**：

```
NOISE_BATCH <seed> <C> <M> <x0,y0,z0> <stepX,stepY,stepZ> <chan0_spec> ...
→ OK NOISE <M*C floats>
```

内核不存在前，Java 侧走 CPU 参考，**不会变慢**。

---

## 7. 本轮提交（mcjavanpu）

| SHA | 内容 |
|---|---|
| `397e9ad6` | `NpuTerrainLock`：按 seed 冻结模式 |
| `6f98ca7f` | `NpuConfig`：`worldLock.<seed>` 持久化 |
| `5d922bd5` | mixin：锁定接管 = 等待 + sentinel，不回退 vanilla |
| `747fd884` | 接管失败日志可见 |

---

*—— 元宝*
