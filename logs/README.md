# Minecraft logs

Uploaded from the device so both AI collaborators work from the same evidence.

| path | what |
|---|---|
| `logs/latest.log` | the game log, overwritten on every upload |
| `logs/mcjavanpu-npu.log` | the mod's own log: NPU results, telemetry dumps, mode switches |
| `logs/archive/*.log.gz` | rotated/crash logs, timestamped, never overwritten |

Upload with:

```sh
GITHUB_TOKEN=xxx sh scripts/upload-logs.sh
```

Add `Telemetry` dumps by running `/npu perf dump` in game first - that writes the per-job CSV
into `mcjavanpu-npu.log`, which this script then ships.
