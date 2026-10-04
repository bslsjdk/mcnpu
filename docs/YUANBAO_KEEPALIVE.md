# 保住 MCNPU 进程（世界加载期间不被杀）

## 现象

世界创建时服务被杀，重建耗时 2m40s：

```
16:46:12  服务 onCreate → QNN 就绪
16:46:25  EXEC SMOKE OK
16:46:55  日志中断            <- 进程被杀
16:47:32  MC 启动 → Connection refused
16:49:35  服务 onCreate（第二次）
```

两次都是 `reason=onCreate`，说明是进程被杀后重建，不是服务内部重启。
世界加载的内存压力是典型触发条件。

## 已做

| 手段 | 位置 |
|---|---|
| doze 白名单 `cmd deviceidle whitelist +` | NpuKeepAlive |
| AppOps RUN_IN_BACKGROUND / RUN_ANY_IN_BACKGROUND / WAKE_LOCK allow | NpuKeepAlive |
| standby bucket 置 ACTIVE | NpuKeepAlive |
| 每 10 分钟重新施加（ROM 会重置） | NpuKeepAlive |
| `android:stopWithTask="false"` | Manifest |
| 显式唤醒广播，绕过 START_STICKY 延迟 | NpuWakeReceiver |
| 每 0.5s 重探服务，回来即恢复 NPU 地形 | mcjavanpu DensitySamplerMixin |

## 为什么用反射调 Shizuku

`Shizuku.newProcess(...)` 的签名在不同版本间有差异，编译期依赖它会把
一个可选增强变成硬构建失败。改为反射，API 不存在时只记日志。
（第一次就是这么红的：直接调用版构建失败，反射版通过。）

## 验证

诊断输出新增一行：

```
KEEPALIVE: 已保活 · uid=0 doze_whitelist=YES RUN_IN_BACKGROUND=ok ...
```

`uid=0` 表示 Shizuku 以 root 运行，能力更强；非 0 是 adb shell，上面几条仍可用。

手动唤醒（免等待）：

```
adb shell am broadcast -a bslsjdk.mcnpu.action.WAKE -n bslsjdk.mcnpu/.NpuWakeReceiver
```

## 仍未解决

ColorOS 的自启/省电是自有策略，`deviceidle whitelist` 只覆盖 AOSP 那套。
若 KEEPALIVE 显示 `doze_whitelist=YES` 仍被杀，需要在系统设置里把 MC NPU
设为「允许后台运行 / 不优化」，并锁定最近任务卡片。
