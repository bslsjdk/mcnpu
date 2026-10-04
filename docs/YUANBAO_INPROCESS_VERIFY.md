# 进程内路线验证方案（2026-10-04）

## 之前为什么没人验证成功

不是代码问题 —— **是没有触发者**。

```
mcfclnpu 插件 → ZL2 注入 -Dmcjavanpu.native=<path>
                                    ↓
                        ZL2 到此为止，不 load
                                    ↓
                        （没有任何代码 System.load）
                                    ↓
                        .so 从未被加载，日志从未产生
```

所以我上一轮说"装插件看日志"是错的 —— **没人会去 load，日志根本不会出现**。

## 已加的触发者（4 个提交，CI 全绿）

| 提交 | 内容 |
|---|---|
| `989a16d636` | `NpuInProcessProbe`：load + init + 报告 |
| `4e5ae66ec0` | `NpuRuntime` 声明 6 个 native 方法 |
| `6739702395` | 自动探针第 0 步（在服务探测之前） |
| `3716fa5a31` | `/npu inprocess` 命令 |

JNI 符号绑定 `bslsjdk.mcjavanpu.NpuRuntime`，所以声明必须在那个类里 ——
这是为何探针逻辑单独成类、声明留在原处。

## 怎么验证

装 mcfclnpu 插件，进游戏，**什么都不用做**。看 `logs/mcjavanpu-npu.log`：

```
inprocess: lib=/data/app/.../libmcfclnpu.so load=OK init=? us=? device=? verdict=...
```

| 结果 | 含义 |
|---|---|
| `SKIP no -Dmcjavanpu.native` | 插件未安装/未挂载 |
| `load=FAIL` + 异常 | dlopen 失败，异常原文即原因 |
| `init=OK` + `IN-PROCESS NPU READY` | 🎉 **零 IPC 通了** |
| `init=FAIL` + device=错误串 | namespace 通了但 QNN 拒绝，看 `logs/mcfclnpu-npu.log` 的 STAGE |

若 `init=FAIL`，同时看 native 日志的 STAGE 序列定位到具体阶段：
`RPC_PRELOAD` → `QNN_DEP` → `QNN_LOAD` → `BACKEND_CREATE` → `DEVICE_CREATE`

## 结果出来后

- **READY** → 跨进程 IPC 整套退役，改走 JNI。12ms → 微秒级
- **FAIL** → 日志会指明是 namespace 还是 QNN，据此决定下一步

---

*—— 元宝*
