# P0: 给 MC 模组用的大批量接口

状态：已实现，两侧 CI 绿。

- `mcjavanpu` — `NpuBigAddClient`（模组侧）+ `/npu bigadd` 命令
- `mcnpu` — `NpuBigAdd.add(a, b)` 便捷入口 + 成本契约 Javadoc

## 一、为什么模组不能直接用 `NpuBigAdd`

`NpuBigAdd` 在 **MCNPU App 进程里**。之前两次通过、2.4 秒的 9×9，是 App 自己
导入 `.binadd` 后内部调用 —— **游戏进程从未真正推过数据**。

所以 P0 的真实含义是：给模组一侧做一个对称的客户端，走 BINADD 数据面，
把切分/并行/合并隐藏在接口后面。

```
MC 模组  →  NpuBigAddClient.add(a, b)  →  IPC BINADD  →  服务  →  HTP
        ←  合并后的 out[]              ←  二进制结果  ←
```

## 二、接口

```java
Result add(float[] a, float[] b)                     // 自动 way + 2 sockets
Result add(float[] a, float[] b, int wayElements, int parallelism)
int    maxWay()                                      // 实测，不是常量
String selfTest(total, wayElements, parallelism)     // 与 CPU 参考逐元素比对
```

`Result`：`out`（失败时 null）、`ways`、`wayElements`、`okWays`、`calls`、
`totalUs`、`status`，以及 `ok()`。

## 三、成本（实测，Snapdragon 8s Gen 3）

| 项 | 数值 |
|---|---|
| 端到端 | **~0.3 µs/元素** |
| 9×9 = 7,962,624 | **2.4 s，两次复现，mismatch=0** |
| 内存 | **12 字节/元素**（a + b + out 同时在场） |

**因此：按 chunk 调用（98,304 元素 ≈ 1.2 MB），不要按 9×9。** 9×9 一次性是
~95 MB 单一分配，在游戏进程里不现实。

## 四、三条纪律

**1. way 大小来自实测，不来自常量。**
`CAPABILITIES max_elements` 由设备探测得出。**已测到 65536，而
`NpuTerrainLattice.MAX_ELEMENTS` 仍硬编码 16384** —— 那条路径的路数因此是
必要值的 4 倍。本次未改（gate 关闭，改动需重新验 parity），**建议 GPT 确认
后处理**。

**2. 先跑一路，再跑全部。**
486 路若因同一原因失败，会报 486 条一模一样的超时、且原因丢失。预检一次
往返（约 1 秒）就带回服务原文。

**3. 失败是返回值，不是异常。**
这在 worldgen 下运行，抛异常不是降级，是让世界崩掉。
另外 `ok` 必须等于 case 数，不能只看"header 以 OK 开头"——部分服务的批次
仍带回完整长度的 body，直接拷贝就会把未写入的 way 当 0 合并进去。这正是
之前 `max_abs = max|a+b|`、99.5% 全错却没有任何报错的原因。

## 五、下一步：P1 单区块端到端

```
/npu bigadd            # 默认 98304 = 一个 chunk
/npu bigadd 98304
```

结果进 `mcjavanpu-npu.log`，形如：

```
BIGADD SELFTEST total=98304 ways=1 n=65536 ... max_abs=0.0 mismatch=0
```

**验收**：`mismatch=0`、`max_abs<1e-3`，并记录单区块端到端耗时。

命令异步执行（虚拟线程）—— 阻塞服务器线程几秒等于停世界，不是测量。
结果只写日志不回聊天：worker 活得比命令长，跨线程碰 `CommandSourceStack`
是在用游戏不拥有的线程改游戏状态。

## 六、必须说明的一点

**ADD 不是加速器。** 它没有累加链，只能证明链路通、带宽够、结果正确。
真正让世界生成变快的东西必须替换掉一项真实成本（Perlin 噪声），那是 P3，
需要 Hexagon 内核走 FastRPC。

把 P0 的 2.4 秒读成"世界生成快了"是误读 —— 那个数字是**搬运 800 万个数**
的耗时，不是省下的时间。
