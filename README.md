# mcnpu

Persistent Android QNN/HTP V73 service for Minecraft Java NPU acceleration.

## IPC contract
The service listens on Android abstract local socket mcnpu_ipc_v1.
Commands: PING, STATUS, CAPABILITIES, SMOKE, EXEC_ADD, QUIT.
Successful execution replies begin with OK HTP_GRAPH_EXECUTE.
STATUS returns QNN HTP ready... when ready, otherwise NPU_OFFLINE plus the last initialization error.

The Fabric mod in bslsjdk/mcjavanpu is the IPC client. This app alone owns QNN/HTP and keeps the runtime initialized for repeated calls.

QNN assets are copied transactionally into app-private files/qnnlibs before the version stamp is committed. Native QNN lifecycle and execution are serialized.

IPC peer credentials are checked. By default only the MCNPU app's own UID is trusted. Additional launcher packages can be allowlisted through the MCNPU app's SharedPreferences file under the key "ipc/trusted_packages", as a comma-separated package list. Rejected peers are logged with UID and resolved package names. This is intentionally deny-by-default so changing launchers does not silently broaden IPC access.