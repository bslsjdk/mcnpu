#!/system/bin/sh
# Upload Minecraft logs to the private mcnpu repo so both AI collaborators can read them.
#
# Usage:  GITHUB_TOKEN=xxx sh upload-logs.sh [instance-dir]
#
# Defaults to the Zalith/Fabric instance used on this device. Uploads the live logs plus any
# rotated .gz files, so a crash that has already rolled over is still captured.
#
# The repo is private, which matters: latest.log contains absolute paths, player coordinates and
# mod lists. Never point this at a public repo.
#
# Requires: curl, base64 (busybox versions are fine).

set -u

TOKEN="${GITHUB_TOKEN:-}"
REPO="${MC_LOG_REPO:-bslsjdk/mcnpu}"
DEST="${MC_LOG_DEST:-logs}"
DIR="${1:-/sdcard/26.3/versions/26.3 Fabric 0.19.5}"
STAMP="$(date +%Y-%m-%d_%H%M%S)"

if [ -z "$TOKEN" ]; then
  echo "GITHUB_TOKEN is not set" >&2
  exit 1
fi

API="https://api.github.com/repos/$REPO/contents"
LOGS="$DIR/logs"

put() {
  src="$1"
  dst="$2"
  [ -f "$src" ] || { echo "skip (missing): $src"; return 0; }

  # A file that already exists in the repo must be replaced with its current sha.
  sha=$(curl -s -H "Authorization: Bearer $TOKEN" \
             -H 'Accept: application/vnd.github+json' \
             "$API/$dst?ref=main" | sed -n 's/.*"sha": *"\([^"]*\)".*/\1/p' | head -1)

  b64=$(base64 -w0 "$src")
  if [ -n "$sha" ]; then
    printf '{"message":"logs: %s (%s)","content":"%s","branch":"main","sha":"%s"}' \
      "$(basename "$dst")" "$STAMP" "$b64" "$sha" > /tmp/mclog_body.json
  else
    printf '{"message":"logs: %s (%s)","content":"%s","branch":"main"}' \
      "$(basename "$dst")" "$STAMP" "$b64" > /tmp/mclog_body.json
  fi

  code=$(curl -s -o /tmp/mclog_resp.json -w '%{http_code}' -X PUT \
         -H "Authorization: Bearer $TOKEN" \
         -H 'Accept: application/vnd.github+json' \
         -d @/tmp/mclog_body.json "$API/$dst")
  echo "$dst -> HTTP $code"
}

# Live logs: always the same path, so the other side knows where to look.
put "$LOGS/latest.log" "$DEST/latest.log"
put "$LOGS/mcjavanpu-npu.log" "$DEST/mcjavanpu-npu.log"

# Rotated/crash logs are timestamped on the way in, because they never overwrite.
i=0
for f in "$LOGS"/*.log.gz; do
  [ -f "$f" ] || continue
  i=$((i + 1))
  put "$f" "$DEST/archive/$(basename "$f")"
done

echo "done: live logs + $i archived"
