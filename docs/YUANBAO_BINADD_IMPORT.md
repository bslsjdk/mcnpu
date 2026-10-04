# .binadd 二进制导入

## 1. INVALID 的真实原因（不是 IPC，不是 JSON，不是 OOM）

`MainActivity` 里有一行本地长度校验：

```java
|| aa.length() > 16384)      // 之前是 > 1024
```

- 1024 → `1024 > 1024` = false → **发出去了，pass 3/3**
- 4096 / 16384 → true → **本地直接拒，从未发送**

所以三个猜测（IPC body 上限 / JSON parser 上限 / OOM）全部不成立。
那些环节失败不会返回 `INVALID`，只有这一行会。

另有第二道上限在服务端：读行缓冲 256 KB，16384 的十进制 ADD 命令约 360 KB，
会撞墙。已改为 8 MB，读超时 3s → 15s（冷建图实测 150–230 ms）。

## 2. 9×9 的真相：486 路，不是 9 路也不是 27 路

```
MC 26.3 区块 = 16×16×384 = 98,304
9×9        = 81 × 98,304 = 7,962,624 元素
ADD 图最大 shape          = 16,384
路数                      = 7,962,624 / 16,384 = 486（整除，无尾部 padding）
```

**方案 A（每路 9 区块 = 884,736 元素）在物理上不存在这个 shape**，会被拒。
想减少路数，前提是 GPT 把 16384 这个天花板抬上去（native 侧，归他）。

## 3. 文件从不整体加载

63.7 MB 的文件若一次性读成 float[]，加输出数组和每路缓冲，远超单进程可用内存。
按 way 分 64 路一段（4 MB/输入数组）流式读取、计算、比对、丢弃，
驻留内存与 n 无关。**第 400 路失败也能报出前 399 路的结果。**

## 4. 每次 IPC 调用的体量由客户端决定

486 路一次提交 = 63.7 MB body，服务端要一次分配，必崩。
改为按 `TARGET_BODY_BYTES = 2 MB` 分批：n=16384 → 16 case/次 → **约 31 次调用**。
n 更小时 case 数自动上调（上限 64），避免被往返开销主导。

## 5. 用法

```bash
python3 tools/gen9x9_bin.py                 # 9×9，60.8 MB，约 1.6 s
python3 tools/gen9x9_bin.py --chunks 4      # 小样本先验证通路
```

App 点「导入二进制 .binadd（9×9）」，选文件。

## 6. 预期输出

```
=== IMPORTED BINADD ===
n=7962624  cases=1  ways=486  way_elements=16384  parallelism=4  file_mb=60.8
crc32 header=... actual=... OK verify_us=...
graph_cached: first=false, rest=true (shape 固定，图只建一次)
ok_ways=486/486  ipc_calls=31  npu_us=...  cpu_cmp_us=...
compared=7962624
max_abs=0.0
mismatch=0
elapsed_ms=...
SUMMARY PASS  failed_segments=0
```

判定写的是 `!(d <= TOL)` 而不是 `d > TOL`：NaN 会让后者为 false，
于是失败的元素会**同时从计数和最大值里消失**。

## 7. 下一步（给 GPT）

`addPadSize` 的 16384 天花板。抬到 65536 → 486 路降到 122 路，
建图次数与每次调用的固定开销一起除以 4。这是目前收益最大的单点。
