# NPU 辅助 vs NPU 接管 —— 术语与实现边界

*元宝 · 2026-10-04*

---

## 一句话

**我们做的是「NPU 辅助」，不是「NPU 接管」。**

---

## 定义

| | NPU 辅助（assist） | NPU 接管（takeover） |
|---|---|---|
| 含义 | NPU 只干它擅长的那部分，CPU 继续干剩下的 | **整块工作全部由 NPU 干** |
| 失败时 | 回退 vanilla，游戏照常 | 无回退 —— 崩或错 |
| 我们的地形方案 | ✅ **是** | ❌ 不是 |

---

## 为什么地形只能是辅助

Minecraft worldgen 不是一段规则数学，它是**混合负载**：

### NPU 适合（大批量、规则、数值）

- Noise octave 批量求值
- 大批量 add / mul
- 规则向量运算
- 插值

### NPU 不适合（分支、状态、结构相关）

- `spline`（分支重）
- `range_choice`（分支）
- cache 语义（状态）
- `beardifier`（结构相关）
- aquifer（状态依赖）
- carvers / surface rules / block placement
- 极小的单点运算

后一半**永远留在 CPU**。所以地形生成在任何情况下都不可能是"全部 NPU 干"。

**"NPU 接管地形"在 Minecraft 里没有对应的可实现形态。**

---

## 代码里的历史错误（已修）

`chunkMode="npu"` 原先被实现成**接管**：

```java
throw new IllegalStateException("NPU terrain result unavailable; vanilla fallback disabled in npu mode");
```

两个问题叠加：

1. **它根本做不到接管** —— `NpuTerrainVanilla.fill()` 里**零 NPU 调用**，
   是一个 CPU 解释器顶着"接管"的名
2. **失败模式是崩溃** —— 一个 CPU 解释器没准备好，就让 worldgen 线程死掉

`d226ba110d` 已把两处 `throw` 改为**记录错误 + 回退 vanilla**。
管道真坏了日志里照样看得见，但世界不会因此崩。

---

## 正确的分工（我们实际要建的）

```
Minecraft Java
   ↓  mixin: DensitySampler$Bound.sampleVolume @HEAD
mcjavanpu
   ↓  从 volume 取 lattice（游戏自己的 stepX/stepY/stepZ）
   ↓
   ├── NPU 辅助部分 ────────────────┐
   │   噪声通道批量求值              │  ← Hexagon DSP 内核
   │   （规则、大批量）              │
   └────────────────────────────────┘
   ↓
   ├── CPU 保留部分 ────────────────┐
   │   add / mul / spline           │
   │   range_choice / cache         │
   │   三线性插值                   │
   └────────────────────────────────┘
   ↓
写回 DensityBuffer
   ↓
Minecraft 继续：biome / surface rules / carvers / structures / block placement
```

**NPU 只贡献中间结果；最终区块仍由 Minecraft 自己完成。**
这就是"辅助"。

---

## 对配置的影响

`chunkMode` 三档：

| 值 | 实际语义 | 建议 |
|---|---|---|
| `vanilla` | 完全不介入 | 默认，安全 |
| `assist` | **NPU 辅助**（命中缓存才用，未命中回退） | ✅ 我们真正要做的 |
| `npu` | 历史命名的"接管" | 已去崩溃，但仍建议用 assist |

**推荐统一用 `assist`。**

---

## 命名建议（后续可清理）

代码里残留的 `takeover` 命名（`NpuTerrainGate.takeoverAllowed`、
`peekTakeover`、`countTakeoverServed`）实际表达的是
"允许 NPU 辅助结果写入"，不是"接管"。

暂不改 API 名（避免破坏调用方），但**语义以本文档为准**。

---

*—— 元宝*
