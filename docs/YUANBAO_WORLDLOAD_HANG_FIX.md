# 世界加载卡死：根因与修复（元宝）

## 结论先行

**服务没启动，不是代码坏了。** 但我的代码把一个正常状态（独立 APK 还没打开）变成了
"世界永远加载不完"，这个必须修。

时间线对得上：

```
16:47:32  mcjavanpu 启动   Connection refused      ← MCNPU 还没开
16:48:47  创建世界         锁定 seed=0 mode=npu
16:48:47  guard DEGRADED   MCNPU_OFFLINE
16:48:55  等满 8000ms      写 sentinel
16:49:03 / 11 / 19 ...     每 8 秒一个区块，一直到 16:50:16+
16:53:57  MCNPU 诊断       PING PONG / QNN ready   ← 服务这时才起来
```

## 三个 bug

### 1. 每个区块阻塞 8 秒（主因）

`DensitySamplerMixin` 的 takeover 分支：`peekTakeover` 拿不到就 `requestWorkSet` 然后
死等到 8 秒超时。spawn 区域几百个区块 × 8 秒 = 比任何人愿意等的时间都长。

**服务没开是一个正常状态，不是故障**，不该按故障那样重罚。

已改为三档：

| 情况 | 等待 |
|---|---|
| 服务在应答 | 2 秒（worker 每 drain 出一批，实际远小于此） |
| 首次 miss，服务未应答 | 等满 30 秒宽限 —— 中途打开 MCNPU 仍然得到 NPU 地形 |
| 宽限已过 | **不等待**，立即标记缺失 |

宽限只给一次，且第一次会打日志提示"现在去启动 MCNPU 还来得及"。

### 2. 未知 seed 是 0，而 0 是合法种子

`NpuChunkWork.lastWorldSeed` 默认 0，`catch` 里也写 0。而 `NpuTerrainLock` 用
`Long.MIN_VALUE` 判"未知" —— 永远不成立。

于是 lock 把"还没拿到 level"当成"这是 seed=0 的世界"，**并且持久化进配置文件**。
日志里 `world lock: seed=0 mode=npu` 就是它。

已改为 `SEED_UNKNOWN = Long.MIN_VALUE`（Minecraft 不可能产生这个种子），未知时推迟锁定。

### 3. 未知 seed 时返回的 mode 未归一化

`acquire()` 在未知分支直接返回 `cfg.chunkMode` 原样，调用方再 `equalsIgnoreCase`。
现在统一小写，只保留一种拼写可比。

## 提交（CI 全绿）

- `96bfd752daf` NpuChunkWork：未知 seed 改用 MIN_VALUE 哨兵
- `6865bcdf50a` NpuTerrainLock：共用哨兵 + mode 归一化
- `a616e685c19` DensitySamplerMixin：分档等待，不再每区块 8 秒

## 没有改的东西

- **没有回退 vanilla。** takeover 拿不到结果时写的仍是 `MISSING = -2.0f` 哨兵：
  有限值（不像 NaN 会污染下游）、明显为负（区块成空气，地上出现洞，一眼看出失败）、
  **绝不是 vanilla 的数字**（失败不会伪装成成功）。
- `NpuTerrainGate` 逻辑未动。
- 没有改默认 `chunkMode`（仍为 `assist`）。用户把菜单切到 `npu` 才是锁定接管。

## 一个反复出现的教训

这已经是第五次：**保护 / 测量机制本身造成停摆。**

| 机制 | 本意 | 实际 |
|---|---|---|
| guard 降级 | 防止 NPU 拖慢游戏 | NPU 全程归零且无法自愈 |
| pump 预算 | 防止卡 server tick | 队头饥饿，8231 提交 0 处理 |
| takeover 等待 | 保证测量完整性 | 每区块 8 秒，世界加载不完 |

共同点都只回答了"何时停"，没回答"停了谁负责恢复""会不会把唯一出路也堵死"。

## 给 GPT

1. `service_queue_us=2298` 在每个请求上完全相同 —— 真实排队会有波动，精确常数更像
   取了陈旧时间戳。若确认是测量问题，**修掉它加速比从 4.71x 到 9.57x**，比优化
   kernel 本身更值钱。这仍是最高优先级。
2. 已确认可用 shape：`m=1024 k=32 n=32`、`m=4096 k=32 n=32`、`m=128 k=512 n=512`。
   terrain lattice 现在提交 1024x32x32 与 256x32x32 两种，服务在时应能跑通。
