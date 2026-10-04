# Noise Kernel Spec — 给实现方

**作者：元宝** · 2026-10-04

这份文档定义 NPU 辅助路径的**唯一**数据传输接口。Java 侧已经按它实现
（`NpuNoiseBatch`），内核可以照此直接写。

---

## 0. 一句话

Minecraft 的 density 程序里，**只有噪声值得搬**。spline、range_choice、cache、
carvers 全是分支/状态相关，搬不了。所以内核只做一件事：

> 给一批点和一批噪声通道，返回每个通道在每个点的值。

其余全部留在 CPU。**这是辅助，不是接管。**

---

## 1. 为什么是这个形状

Bench 实测（`mcnpu/tools/worldgen-bench`）：

| 路线 | 结果 |
|---|---|
| matmul 地形 | **0.05x**，一个 chunk 要 60,907 次碎提交 |
| **批处理噪声内核** | **4.7x**，固定开销修好后 9.6x |

matmul 输不是因为数学错，是因为**每次提交先付 ~2485µs 固定开销**，六万次就补不回来了。
内核一次进一批点、回一批值，固定开销按块付一次。

---

## 2. 请求格式（v2）

小端。请求**不列坐标**——列坐标 16 chunks 要 235KB，而 lattice 是规则的，
描述它只要 224 字节。

```
u16 magic      = 0x4E42   ('NB')
u16 version    = 2
u16 flags      bit0: 结果 int8    bit1: 含 perlin 表
u16 channelCount
u16 chunkCount
u8  cellXZ
u8  cellY
u16 lx, ly, lz            每个 chunk 的 lattice 维度
u64 seed                  仅用于校验/日志，不参与计算

每个通道:
  u8  firstOctave
  u8  octaveCount
  f32 normalization
  f32 xzScale
  f32 yScale
  f32 amplitudes[octaveCount]        0 表示该 octave 跳过
  每个非零 octave:
    f64 xo, f64 yo, f64 zo           perlin 偏移
    u8  perm[256]                    Fisher-Yates 置换

每个 chunk:
  i32 chunkX, i32 chunkZ, i32 minY
```

lattice 点由内核自己生成：

```
x = chunkX*16 + ix*cellXZ
y = minY      + iy*cellY
z = chunkZ*16 + iz*cellXZ
```

噪声在缩放后的坐标上求值：

```
nx = x * xzScale
ny = y * yScale
nz = z * xzScale
```

---

## 3. 🔴 最重要的一条：传表，不传种子

请求里**没有** PRNG 状态。perlin 的偏移和置换表是 Java 侧算好后直接传过去的。

**不要试图在 DSP 上复刻 Java 的 Xoroshiro128++。**

理由：复刻错一点不会崩，也不会报错，只会生成一个**看起来合理但不同的世界**——
这正是 `NpuTerrainGate` 存在的唯一原因，也是最难查的一类故障。
传表把这个风险整个消掉：内核根本不需要 PRNG，出错就是表对不上，可以直接打印对比。

代价：每 octave 280 字节。19 个 octave ≈ 5.3KB，对 76KB 的结果回传来说不到 7%。

---

## 4. 结果格式

```
u16 magic   = 0x4E52   ('NR')
u16 status      0 = ok
u16 channelCount
u32 pointCount
每个通道: f32 scale
payload: channel-major，int8[channelCount * pointCount]
```

值 = `int8 * scale`。

用 int8 是因为 bench 实测量化误差 MAE **0.0004**——低于任何能改变地形的阈值，
而回传字节是 fp32 的四分之一。

16 chunks × 1225 点 × 4 通道 = 76KB。

---

## 5. 每个点的计算

```
value = 0
for o in 0..octaveCount:
    if amplitudes[o] == 0: continue
    freq = 2^(firstOctave + o)
    value += perlin(perm_o, xo_o, yo_o, zo_o,
                    nx*freq, ny*freq, nz*freq) * amplitudes[o]
return value / normalization
```

perlin 内部（Java 实现逐字对照）：

```
fx = nx*freq + xo ;  ix = floor(fx) ;  dx = fx - ix
X = ix & 255                      （iy/iz 同理）
u = fade(dx) = dx^3 * (dx*(dx*6-15) + 10)

a0 = p[X] + Y ;  a1 = p[X+1] + Y
b0 = p[a0] + Z ; b1 = p[a0+1] + Z ; b2 = p[a1] + Z ; b3 = p[a1+1] + Z

gradDot(hash, x, y, z):
    h = hash & 15
    u' = h < 8 ? x : y
    v  = h < 4 ? y : (h == 12 || h == 14 ? x : z)
    return ((h&1)==0 ? u' : -u') + ((h&2)==0 ? v : -v)

g000 = gradDot(p[b0],     dx,   dy,   dz  )
g100 = gradDot(p[b2],     dx-1, dy,   dz  )
g010 = gradDot(p[b0+1],   dx,   dy-1, dz  )
g110 = gradDot(p[b2+1],   dx-1, dy-1, dz  )
g001 = gradDot(p[b1],     dx,   dy,   dz-1)
g101 = gradDot(p[b3],     dx-1, dy,   dz-1)
g011 = gradDot(p[b1+1],   dx,   dy-1, dz-1)
g111 = gradDot(p[b3+1],   dx-1, dy-1, dz-1)

x00 = lerp(u, g000, g100) ; x10 = lerp(u, g010, g110)
x01 = lerp(u, g001, g101) ; x11 = lerp(u, g011, g111)
y0  = lerp(v, x00, x10)   ; y1  = lerp(v, x01, x11)
return lerp(w, y0, y1)
```

**注意两个坑**（我在 bench 里都踩过）：

1. **corner 位序是 `z, y, x`，不是 `x, y, z`**。写反的话误差 0.665 → 0.004。
2. 26.3 全程 **float** 语义，中间步骤也是。double 中间值 ≠ float 中间值。

---

## 6. 必须一起做的两件事

### 6.1 绝不能混 seed / 维度

一次调用只能属于**同一个世界的同一个维度**。

我第一版按到达顺序分组，不同世界的 chunk 进了同一批，
密度 MAE 从 0.0004 跳到 **0.04（100 倍）**，而吞吐数字**完全正常**。
它永远不会崩，只会表现为"某些地方地形微妙地不对"。

Java 侧已按 `(seed, minY)` 分组强制执行。内核侧可以假设成立，
但服务端校验一下更稳。

### 6.2 固定开销比 kernel 速度值钱

| 每次固定开销 | ms/chunk | 加速比 |
|---|---|---|
| 2485（当前） | 1.41 | **4.71x** |
| 800 | 0.89 | 7.51x |
| 187（纯 IPC） | 0.70 | **9.57x** |

**`service_queue_us=2298` 每次请求完全相同**——真实排队会有波动，
精确常数更像取了陈旧时间戳。如果它是测量 artefact，修掉它**加速比直接翻倍**，
这比优化 kernel 本身还值钱。

---

## 7. 验收

1. 单 chunk：内核 vs CPU 参考，**逐点对比**，max_abs_err 应 < 1e-3
2. 用 `mcnpu/tools/worldgen-bench` 的 differential compare 跑 100 seed × 100 chunk
3. 目标：density MAE < 0.005，sign agreement > 99.5%
4. 过了才轮到 `NpuTerrainGate` 打开

**gate 保持关闭直到第 3 条通过。** 未经验证就替换 vanilla 是最坏的一类 bug：
不立刻崩，而是污染光照、结构、水面，然后在完全无关的地方爆。

---

## 8. 现状

| 部件 | 状态 |
|---|---|
| 协议 + 编码/解码 | ✅ 已实现（`NpuNoiseBatch`） |
| 批处理器（seed/dim 隔离） | ✅ 已实现（`NpuNoiseBatcher`） |
| 密度程序批量噪声提取 | ✅ 已实现（`NpuNoiseAssist` + `NpuDfProgram`） |
| **内核** | ❌ **不存在** |
| 默认行为 | CPU 参考，结果与现在**逐位一致** |

`NpuNoiseBatch.available()` 为 false 直到服务端在 CAPABILITIES 里广告
`NOISE_BATCH`。在此之前整条路径是 no-op，不会变慢。

---

*—— 元宝*
