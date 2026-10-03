# ⚠️ 更正 · 元宝 2026-10-03

## 撤回：P0-1「padding 浪费 512 倍」是错误判断

**如果你（DeepSeek 或其他 AI）正在准备执行 `YUANBAO_TEST_PLAN_AND_LOG_ANALYSIS.md` 里的 P0-1，请立刻停下。那个任务是错的，执行它会把正确的实现改坏。**

### 我当时错在哪

我从日志这一行推断：

```
pad_to=128x512x512   cells=512   bad=0/65536
```

看到 `cells=512`，就认定"每个 block 只需要 1 个输出"，于是认为逻辑形状是 `A[128×512] × B[512×1]`，`n=512` 是 512 倍的 padding 浪费。

**这个推断是错的。** 看了 `NpuLightAccel` 的实现后才明白：

```java
L_out = A * L_in ,  A = I + w * (sum of the 6 face neighbours)
每个 8×8×8 block 展平 = 512 cells
一批 m 个 block → (m × 512) × (512 × 512) int8 matmul
```

A 是 **512×512 的真实传播算子矩阵**（8×8×8 体素的 6 邻域关系）。每个 block **本来就要输出 512 个光照值**，不是 1 个。

`bad=0/65536` 里的 65536 = 128 × 512 = m × n，正是 C 矩阵的完整大小。**完全吻合，没有浪费。**

### 而且这个路径现在是赚的

类注释里有 requantisation 缓存落地后的实测：

```
512^3 steady state:  NPU 5.5ms  vs  host ~40ms   →  约 7 倍加速
propagation operator numerically exact: bad=0/262144
```

我报告里引用的 `lightapply cpu=4491us npu=37782us → 慢 8.4 倍` 是**旧日志**，采集于 requantisation scale 缓存（`518521ed59`）**之前**。那个数字已经失效。

**结论：光照路径目前是正收益，不是负优化。不要动 n=512。**

### 保留的部分

同一份报告里这些结论**依然成立，请继续执行**：

- ✅ **1.1 进程内路线已死**（`clns-9` namespace 实锤）——别再投 mcfclnpu
- ✅ **1.3 speedup 数字不可信**（cpu_us 波动 100 倍）——换 `pipelineSpeedup` + p99
- ✅ **1.4 reloadchunks 反射失败**（`setLightEnabled` NoSuchMethodException）——需要修
- ✅ **1.5 光照引擎 150 次/秒**——仍是调度约束
- ✅ **P0-2 / P0-3 / P0-4**——照常执行

### 这一轮我实际做的优化（已提交）

`ac54d225f9` — `NpuChunkWork` 反射缓存：

```
每次 runForChunk 都在 server tick 上做 4 次 Class.getMethods() 全量遍历
（getLayerListener / getDataLayerData / queueSectionData / SectionPos.of）
而 getMethods() 每次都克隆整个 Method 数组
→ 每秒几十次 × 4 次全量克隆
```

改为按 engine/listener 的 Class 缓存已解析的 Method，换世界自动失效；另外把方法内的 8 项 sub-block 偏移表提为常量（原先每个 chunk 分配 9 个数组）。

---

## 教训

我从日志的一个字段反推语义，没有先读实现。**日志字段的含义必须由代码定义，不能由数字形状猜测。** 后面凡是"从日志推断根因"的结论，都要先回代码确认一遍再写进任务列表。

---

*—— 元宝（Yuanbao），2026-10-03*
