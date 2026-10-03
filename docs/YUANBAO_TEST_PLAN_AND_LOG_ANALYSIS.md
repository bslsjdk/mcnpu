# 元宝 · 日志分析与设备测试计划

> **作者：元宝（Yuanbao）**
> 日期：2026-10-03
> 依据：用户上传的 `mcjavanpu-npu.log` + `latest.log`
> 读者：DeepSeek（执行修复）、用户（执行测试）
>
> 本文分三部分：**已确认的结论** → **必须修的任务** → **下一步怎么测**。
> DeepSeek 请直接执行第二部分；用户请按第三部分操作，然后上传日志。

---

# 第一部分 · 已确认的结论

## 1.1 mcnpu（IPC 路线）是通的，别再折腾 mcfclnpu

```
boot: available=true device=QNN HTP ready backendId=6 providers=1
PING=PONG MCNPU/1
CAPABILITIES=OK MCNPU/1 backend=HTP_V73 ops=ADD,MATMUL,MATMUL16,MATMUL8,SUBMIT8,SUBMITBIN8 max_elements=16384
```

而 mcfclnpu（进程内）日志里是：

```
QNN_LOAD_FAIL ... libQnnHtp.so ... is not accessible for the namespace "clns-9"
```

**`clns-9` 是启动器的 classloader namespace。** 这是我第 1 节预判的那个问题，现在实锤了。

> **结论：进程内路线已死，不要再投入。所有精力放在 mcnpu + IPC。**

## 1.2 🔴 最严重的问题：padding 浪费了 512 倍

日志里这一行藏着全部答案：

```
pad_to=128x512x512   blocks_batch=128   cells=512   bad=0/65536
```

反推形状：

```
A[128 x 512]   B[512 x 512]   C[128 x 512]
```

`bad=0/65536` 里的 65536 = 128 × 512，正是 C 的元素数，吻合。

但 `cells=512` 说明**每个 block 只需要 1 个输出**。也就是逻辑形状其实是：

```
A[128 x 512] x B[512 x 1] -> C[128 x 1]
```

而实际执行的是 `n=512`。

| 项 | 逻辑需要 | 实际执行 | 浪费 |
|---|---|---|---|
| B 矩阵 | 512 | 262,144 | **512 倍** |
| cost ≈ O(k·n) | 512 | 262,144 | **512 倍** |

按我在 `NpuShapeAdvisor` 里写的成本模型（cost ≈ O(k·n)，m 几乎免费），**这一次调用把算力花在了 511/512 的空转上**。

### 这是负优化的直接原因

用日志里最干净的一次对比（`lightapply`，CPU 参考明显已 JIT 预热）：

```
cpu_us=4491    npu_us=37782    speedup=0.12x
```

**CPU 4.5ms，NPU 37.8ms —— 慢 8.4 倍。**

另一次 `lightfold`：

```
cpu_us=10987   npu_us=18282    speedup=0.60x   （慢 1.7 倍）
```

**当前光照路径是明确的负优化，不是"还没调好"。**

### 修掉之后能到多少

如果 `n` 从 512 降到最小桶 32（bucketize 的下限）：

```
cost:  262144  ->  16384     降低 16 倍
npu_us: 17685  ->  ~1100us   （估算）
```

~1.1ms 就进入了可接受区间。**这是当前投入产出比最高的一刀。**

## 1.3 🔴 那个 speedup 数字不可信，别拿它做决策

同一个 warmup 动作，跨 6 次会话：

```
npu_us:  60679  39453  48845  61804  9770  17685  11476  14025
cpu_us: 500687 105332  59372 191174 ...   203161 106615  44122  88290
```

`cpu_us` 波动超过 **100 倍**（4,491 → 500,687）。这说明它测的不是一个稳定的东西 —— 里面混了 JIT 预热、GC、线程调度。

于是 speedup 在 **0.12x 到 20.79x** 之间横跳。**这个数字没有任何决策价值。**

我在上一轮已经为此写了 `NpuBench`，它会分阶段测：

```
execSpeedup     = cpuRef / submit
pipelineSpeedup = cpuRef / (prepare + submit)      ← 决策用这个
```

并输出 p50 / p99。**请用它替换现有算法，不要继续用 `cpu_us / npu_us`。**

## 1.4 🔴 `reloadchunks` 反射失败（26.3 API 已变）

```
[NPU] reloadchunks FAIL NoSuchMethodException:
net.minecraft.server.level.ThreadedLevelLightEngine
  .setLightEnabled(SectionPos, boolean)
```

每次切换 lightMode / chunkMode 都会触发，日志里出现了至少 2 次。

**26.3 里这个方法名或签名已改。** 结果是模式切换后光照不会重建，用户看到的效果和实际配置不一致 —— 会让后面所有"A/B 对比"失真。

## 1.5 光照引擎调用频率（给调度用）

```
19:43:56  light engine calls=1
19:44:06  light engine calls=1531      （10 秒内 1530 次）
19:44:16  light engine calls=3329
```

**约 150 次/秒。** 这个频率下，任何超过 **6ms** 的同步 NPU 调用都会直接吃掉帧预算。

当前 `npu_us` 普遍在 11–38ms，**全部超标**。

---

# 第二部分 · DeepSeek 必须执行的任务

按优先级排。**请一次只做 P0，改完提交，让用户测一轮再动 P1。** 否则又会出现"改了 17 个地方不知道哪刀生效"的考古现场。

## P0-1 · 修 padding：n 不该是 512

**目标**：让 `pad_to` 从 `128x512x512` 降到 `128x512x32`。

**要求**：

1. 找到光照路径里 `n` 被设成 512 的位置（`NpuLightAccel` 或 `NpuLightHook` 的提交逻辑）
2. 逻辑 `n=1` 时，走 `bucketize(1) → 32`，**不要硬编码 512**
3. 提交前用 `NpuShapeAdvisor.advise(m, k, n)` 打印 `padding_ratio`，**要求 < 2.0**
4. 提交后日志里应该看到 `pad_to=128x512x32`

**验收**：`npu_us` 从中位 ~14000 降到 **< 3000**。

## P0-2 · 修 `reloadchunks` 反射

**要求**：

1. 不要猜方法名。在失败分支里**反射列出 `ThreadedLevelLightEngine` 的所有方法名和签名**，打印到日志
2. 下一次日志拿到方法清单后，改用正确的方法
3. 在此之前，失败时**优雅降级**（记 WARN 继续跑），不要抛异常打断模式切换

参考写法：

```java
for (Method m : ThreadedLevelLightEngine.class.getDeclaredMethods()) {
    NpuLog.log("LIGHT_ENGINE_METHOD " + m.getName() + " "
        + Arrays.toString(m.getParameterTypes()));
}
```

## P0-3 · 把 speedup 换成 NpuBench 的口径

**要求**：所有对外汇报 speedup 的地方，改用：

```
pipelineSpeedup = cpuRef / (prepare + submit)
```

并同时打印 p99。现有 `cpu_us / npu_us` 的单值算法**废弃**。

## P0-4 · 确认我的自动探针已生效

我上一轮提交了 `NpuAutoProbe` + `NpuGuard`，但本次日志里**没有它们的输出**（日志 session 停在 19:43，latest.log 是 21:56，且没有 autoprobe 报告）。

**要求**：确认最新构建里包含这两个模块。下次日志**必须**出现：

```
==== NPU auto probe ====
service: UP
...
==== end auto probe ====
```

以及每 60 秒一行：

```
heartbeat | ... | guard=ok | service=UP
```

如果没有，说明构建没带上，先解决构建问题再往下做。

---

# 第三部分 · 用户测试步骤（这段给人类看）

**你什么都不用配置，也什么都不用输。**

## 步骤

1. 确认 `mcjavanpu.properties` 里 `autoProbe=true`、`guardEnabled=true`
   （游戏内 NPU 功能界面也能切，见"自动探测 / 自适应降级"两行）
2. **进游戏，正常玩 5–10 分钟**
   - 正常跑图、加载区块、开关光照都行
   - **不要**刻意输 `/npu` 命令
3. **退出游戏**
4. 上传这两个文件：
   - `<游戏目录>/logs/mcjavanpu-npu.log` ← **主要看这个**
   - `<游戏目录>/logs/latest.log`（可选，只在崩溃时需要）

## 我要在日志里看到的

| 标记 | 含义 |
|---|---|
| `==== NPU auto probe ====` | 自动探针跑起来了 |
| `pad_to=...` | padding 后的真实形状（期望 `128x512x32`） |
| `npu_us=` | NPU 单次耗时（期望 **< 3000**） |
| `pipelineSpeedup=` | 分阶段加速比（**> 1.0 才算真加速**） |
| `guard=ok` / `guard=DEGRADED` | 守卫有没有触发 |
| `heartbeat \|` | 每 60 秒的滚动统计 |
| `LIGHT_ENGINE_METHOD` | 26.3 的真实方法清单（P0-2 产出） |

## 判定标准（这次就用这个，不再靠感觉）

```
pipelineSpeedup > 1.0   且   npu_us p99 < 6000     → 这条路径成立，继续做
pipelineSpeedup < 1.0   或   npu_us p99 > 16000    → 路径仍是负优化，先别接更多功能
guard=DEGRADED 频繁出现                             → 服务不稳，先查服务别查算法
```

---

# 第四部分 · 优先级提醒（给 DeepSeek）

GPT 报告里的 9×9 工作集、shared buffer、原版相似度度量 —— **这些现在都不要动**。

理由：当前最热的问题是一个 **512 倍的 padding 浪费**。在它修掉之前做那些优化，等于在漏水的桶上雕花。

顺序只能是：

```
P0-1 padding   →  P0-2 反射  →  P0-3 口径  →  P0-4 探针确认
        ↓
   测一轮，看 pipelineSpeedup
        ↓
   > 1.0 才继续做批处理 / 9x9 / shared buffer
```

**没有实测数据之前，不要新增功能。**

---

*—— 元宝（Yuanbao），2026-10-03*
