# mcnpu

Persistent Android QNN/HTP V73 service for Minecraft Java NPU acceleration.

## IPC contract
The service listens on TCP loopback 127.0.0.1:38761.
Commands: PING, STATUS, CAPABILITIES, SMOKE, EXEC_ADD, QUIT.
Successful execution replies begin with OK HTP_GRAPH_EXECUTE.
STATUS returns QNN HTP ready... when ready, otherwise NPU_OFFLINE plus the last initialization error.

The Fabric mod in bslsjdk/mcjavanpu is the IPC client. The MCNPU service owns QNN/HTP and keeps the runtime initialized for repeated calls.

On Android 16, local-network protection is an opt-in compatibility feature during the current rollout. MCNPU declares INTERNET and NEARBY_WIFI_DEVICES; the app requests Nearby devices access before starting the IPC service. ACCESS_LOCAL_NETWORK is not required for targetSdk 35; Android 17/targetSdk 37+ is where the new local-network permission is enforced.

QNN assets are copied transactionally into app-private files/qnnlibs before the version stamp is committed. Native QNN lifecycle and execution are serialized.

The IPC listener is self-healing: bind/accept failures close the broken ServerSocket, wait briefly, and retry while the foreground service is running.

## Architecture review

- [GPT-5.6 Luna NPU optimization report](docs/GPT_NPU_OPTIMIZATION_REPORT.md) — current code review, memory/IPC/batching risks, 9×9 workset design, and recommended implementation order.
