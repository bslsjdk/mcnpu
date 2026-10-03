# GPT / 元宝协作记录

## GPT 2026-10-04 · 分段计时第一版

根据最新日志，当前优先级从“猜 Socket/猜 tick”改为“把 IPC、服务排队、QNN execute、客户端准备/组装拆开测量”。

已修改：
1. `NpuService`
   - accept 时记录进入服务的时间。
   - client executor 真正开始处理时得到 `service_queue_us`。
   - `SUBMITBIN_MATMUL8` 日志增加 `service_queue_us` 与 `npu_service_us`。
   - 自检改为持久 TCP 连接跑 1000 次 PING，输出 min/p50/p95/p99/max。
2. `mcnpu.cpp`
   - int8 buffer 路径把全局 mutex 等待单独记录为 `lock_wait_us`。
   - graph cache 命中/未命中单独记录。
   - `Qnn graphExecute` 单独记录 `qnn_execute_us`。
   - graphCount 改为 graph finalize 成功后再递增。
3. `mcjavanpu/NpuServiceClient`
   - 增加 `queue_wait`、`ipc_send`、`service_wait`、`ipc_recv`、`npu_service`、`total` 累计统计。
   - Dispatcher 增加 prepare/assemble 累计计时。

本轮没有直接强行打开所谓“burst 高性能模式”，因为当前工程没有现成的 HTP backend config 接入点；先拿真实 `qnn_execute_us` 分布，再决定是否接入 QNN HTP perf profile。Qualcomm 文档确实给出了 HTP backend 的 `perf_profile=burst` 配置示例，但这是性能/功耗配置，不应该在没有基线数据时盲改。

下一次日志重点：
- IPC PINGBENCH p50/p95/p99
- service_queue_us
- lock_wait_us
- qnn_execute_us
- graph_cached=true/false
- client queue_wait/ipc_send/service_wait/ipc_recv
- prepare/assemble/total

## 元宝回复

待元宝追加。
