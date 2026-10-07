package bslsjdk.mcnpu;

import android.app.AlertDialog;
import android.app.Activity;
import android.os.Bundle;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.net.Uri;
import org.json.JSONArray;
import org.json.JSONObject;
import android.widget.*;
import android.text.method.ScrollingMovementMethod;
import java.io.FileInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.atomic.AtomicBoolean;

public final class MainActivity extends Activity {
    /**
     * Read at use time, not frozen here. The native ladder's top rung is measured
     * on the device at startup; a hardcoded copy of it would keep rejecting sizes
     * the phone actually accepts, which is exactly how 4096 and 16384 spent
     * several rounds looking like an HTP limitation.
     */
    private static int maxCaseLen() { return NpuServiceClient.maxAddElements(); }
    private static final int REQ_LOCAL_NETWORK = 5101;
    private TextView npuState, npuDetail, shizukuState, log;
    /**
     * Everything appended to the on-screen log this session.
     *
     * The TextView is trimmed for rendering (trimVisibleLog keeps the last 20000 chars), so it
     * cannot be used as the copy source. This buffer is what copy/share actually reads.
     */
    private static final StringBuilder sessionLog = new StringBuilder();
    private static final int SESSION_LOG_CAP = 200_000;
    private ScrollView logScroll;
    private String lastServiceLog = "";
    private final ScheduledExecutorService statusExecutor = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread t = new Thread(r, "mcnpu-ui-status");
        t.setDaemon(true);
        return t;
    });
    private final AtomicBoolean statusInFlight = new AtomicBoolean();
    private ScheduledFuture<?> statusFuture;

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        ShizukuHelper.init();
        ShizukuHelper.setCallback(this::refreshStatus);
        setContentView(R.layout.activity_main);

        npuState = findViewById(R.id.npuState);
        npuDetail = findViewById(R.id.npuDetail);
        shizukuState = findViewById(R.id.shizukuState);
        log = findViewById(R.id.log);
        logScroll = findViewById(R.id.logScroll);
        log.setMovementMethod(ScrollingMovementMethod.getInstance());

        findViewById(R.id.start).setOnClickListener(v -> startNpuService());
        findViewById(R.id.test).setOnClickListener(v -> runSmoke());
        findViewById(R.id.importTest).setOnClickListener(v -> chooseTestFile());
        findViewById(R.id.importBin).setOnClickListener(v -> chooseBinFile());
        findViewById(R.id.ipcWhitelist).setOnClickListener(v -> editIpcWhitelist());
        findViewById(R.id.shizukuOpen).setOnClickListener(v -> openShizuku());
        findViewById(R.id.shizukuRequest).setOnClickListener(v -> requestShizuku());
        findViewById(R.id.copyLog).setOnClickListener(v -> copyLog());
        findViewById(R.id.shareLog).setOnClickListener(v -> shareLog());
        findViewById(R.id.refreshLog).setOnClickListener(v -> refreshServiceLogIncremental());
        refreshLogOnly();

        ensureLocalNetworkPermission();
    }

    private void ensureLocalNetworkPermission() {
        if (Build.VERSION.SDK_INT < 33) {
            startNpuService();
            return;
        }
        if (checkSelfPermission("android.permission.NEARBY_WIFI_DEVICES") == PackageManager.PERMISSION_GRANTED) {
            appendLog("本地网络权限：已授权");
            startNpuService();
            return;
        }
        appendLog("本地网络权限未授权，正在申请。Android 16 会阻止本地 socket 时返回 EPERM。");
        requestPermissions(new String[]{"android.permission.NEARBY_WIFI_DEVICES"}, REQ_LOCAL_NETWORK);
    }

    @Override public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQ_LOCAL_NETWORK) {
            boolean granted = grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED;
            appendLog("本地网络权限结果：" + (granted ? "已授权" : "拒绝"));
            if (granted) startNpuService();
            else appendLog("MC NPU IPC 未启动：需要本地网络权限才能监听 127.0.0.1:38761");
        }
    }

    @Override protected void onResume() {
        super.onResume();
        if (statusFuture == null || statusFuture.isCancelled()) statusFuture = statusExecutor.scheduleAtFixedRate(this::refreshStatus, 0, 1, TimeUnit.SECONDS);
    }

    @Override protected void onPause() {
        if (statusFuture != null) statusFuture.cancel(false);
        super.onPause();
    }

    @Override protected void onDestroy() {
        statusExecutor.shutdownNow();
        super.onDestroy();
    }

    private void editIpcWhitelist() {
        final EditText input = new EditText(this);
        input.setSingleLine(false);
        input.setHint("例如：com.movtery.zalithlauncher.v2,net.kdt.pojavlaunch");
        input.setText(getSharedPreferences("ipc", MODE_PRIVATE).getString("trusted_packages", ""));
        new AlertDialog.Builder(this)
                .setTitle("IPC 客户端白名单")
                .setMessage("TCP IPC 已限制为本机回环地址。此设置仅保留兼容入口。")
                .setView(input)
                .setNegativeButton("取消", null)
                .setPositiveButton("保存", (d, which) -> {
                    String value = input.getText().toString().trim();
                    getSharedPreferences("ipc", MODE_PRIVATE).edit()
                            .putString("trusted_packages", value).apply();
                    appendLog("IPC 白名单已保存: " + (value.isEmpty() ? "<未设置>" : value));
                }).show();
    }

    private void startNpuService() {
        try {
            Intent i = new Intent(this, NpuService.class);
            if (Build.VERSION.SDK_INT >= 26) startForegroundService(i);
            else startService(i);
            appendLog("请求启动独立 NPU 服务");
            statusExecutor.schedule(this::refreshStatus, 800, TimeUnit.MILLISECONDS);
        } catch (Throwable t) {
            appendLog("启动失败: " + t);
        }
    }

    private static final int PICK_TEST_FILE = 4101;
    private static final int PICK_BIN_FILE = 4102;

    private void chooseTestFile() {
        try {
            Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            i.addCategory(Intent.CATEGORY_OPENABLE);
            i.setType("application/json");
            startActivityForResult(i, PICK_TEST_FILE);
        } catch (Throwable t) {
            appendLog("打开文件选择器失败: " + t);
        }
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        if (requestCode == PICK_TEST_FILE) {
            new Thread(() -> runImportedTest(uri)).start();
        } else if (requestCode == PICK_BIN_FILE) {
            new Thread(() -> runImportedBin(uri)).start();
        }
    }

    /**
     * Binary import. The picker is opened with a wildcard type because .binadd is
     * not a registered MIME type; the header magic is what actually identifies a
     * valid file, so a wrong pick is reported as a magic mismatch rather than
     * being silently rejected by the intent filter.
     */
    private void chooseBinFile() {
        try {
            Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            i.addCategory(Intent.CATEGORY_OPENABLE);
            i.setType("*/*");
            i.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{"application/octet-stream", "*/*"});
            startActivityForResult(i, PICK_BIN_FILE);
        } catch (Throwable t) {
            appendLog("打开文件选择器失败: " + t);
        }
    }

    private void runImportedBin(Uri uri) {
        try {
            runOnUiThread(() -> appendLog("二进制导入：正在复制到缓存…"));
            File f = NpuBinFile.materialize(this, uri);
            String desc = NpuBinFile.describe(f);
            runOnUiThread(() -> appendLog("文件就绪 " + (f.length() / 1048576.0) + " MB\n" + desc));
            NpuBinFile.Progress sink = line -> runOnUiThread(() -> appendLog(line));
            String report = NpuBinFile.run(f, sink);
            runOnUiThread(() -> appendLog(report));
        } catch (Throwable t) {
            runOnUiThread(() -> appendLog("二进制导入失败: " + t));
        }
    }

    private void runImportedTest(Uri uri) {
        try {
            String json;
            long jsonBytes = 0;
            try (java.io.InputStream in = getContentResolver().openInputStream(uri);
                 java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream()) {
                if (in == null) throw new java.io.IOException("无法打开文件");
                byte[] buf = new byte[8192]; int n;
                while ((n = in.read(buf)) != -1) out.write(buf, 0, n);
                json = out.toString("UTF-8");
            }
            JSONObject root = new JSONObject(json);
            String task = root.optString("task", "add");
            JSONArray cases = root.getJSONArray("inputs");
            // Ask the service what its probe measured before deciding what is too
            // long. Without this the cap stays at the pre-probe default and sizes
            // the device actually accepts keep being rejected locally.
            NpuServiceClient.refreshMaxAddElements();
            StringBuilder result = new StringBuilder();
            result.append("=== IMPORTED NPU TEST ===\n")
                  .append("task=").append(task).append("\n")
                  .append("cases=").append(cases.length()).append("\n");
            java.util.ArrayList<float[]> pendingA = new java.util.ArrayList<>();
            java.util.ArrayList<float[]> pendingB = new java.util.ArrayList<>();
            int pass = 0;
            for (int k = 0; k < cases.length(); k++) {
                JSONObject item = cases.getJSONObject(k);
                JSONArray aa = item.getJSONArray("a");
                JSONArray bb = item.getJSONArray("b");
                // Say which rule rejected the case. A bare INVALID gave no way to tell
                // "wrong shape" from "too long", and 4096/16384 looked like an HTP
                // failure when they were rejected here without ever being sent.
                String bad = null;
                if (!"add".equalsIgnoreCase(task)) bad = "task not add";
                else if (aa.length() != bb.length()) bad = "len mismatch a=" + aa.length() + " b=" + bb.length();
                else if (aa.length() == 0) bad = "empty";
                else if (aa.length() > maxCaseLen()) bad = "len " + aa.length() + " exceeds cap " + maxCaseLen();
                if (bad != null) {
                    result.append("CASE ").append(k).append(": INVALID ").append(bad).append("\n");
                    continue;
                }
                float[] fa = new float[aa.length()], fb = new float[aa.length()];
                for (int i=0;i<aa.length();i++) {
                    double av = aa.getDouble(i), bv = bb.getDouble(i);
                    if (!Double.isFinite(av) || !Double.isFinite(bv) || av > Float.MAX_VALUE || av < -Float.MAX_VALUE || bv > Float.MAX_VALUE || bv < -Float.MAX_VALUE) throw new IllegalArgumentException("非有限或超出 float 范围");
                    fa[i] = (float) av; fb[i] = (float) bv;
                }
                pendingA.add(fa); pendingB.add(fb);
            }
            // One binary call for every accepted case. Beyond removing the decimal
            // text, this is what the multi-chunk path needs: one fixed-shape call
            // per batch instead of one call per chunk.
            if (!pendingA.isEmpty()) {
                int n = pendingA.get(0).length;
                float[][] a2 = pendingA.toArray(new float[0][]);
                float[][] b2 = pendingB.toArray(new float[0][]);
                long binT0 = System.nanoTime();
                NpuServiceClient.BinAddResult binReply = NpuServiceClient.binAdd(a2, b2, n);
                long binUs = (System.nanoTime() - binT0) / 1000L;
                int okCount = binReply.okCount;
                result.append("BIN n=").append(n).append(" cases=").append(pendingA.size())
                      .append(" roundtrip_us=").append(binUs)
                      .append(" json_bytes~").append(jsonBytes)
                      .append(" ").append(binReply.status).append("\n");
            }
            result.append("SUMMARY pass=").append(pass).append("/")
                  .append(cases.length()).append(" NPU=HTP V73\n")
                  .append("NOTE=每个 case 记录 JSON 实际长度和 IPC 往返时间；SIZE_UNSUPPORTED 会包含 native requested_n。\n");
            String output = result.toString();
            runOnUiThread(() -> appendLog(output));
        } catch (Throwable t) {
            runOnUiThread(() -> appendLog("导入测试失败: " + t));
        }
    }

    private void runSmoke() {
        new Thread(() -> {
            String s = NpuServiceClient.request("SMOKE");
            runOnUiThread(() -> appendLog("HTP Graph Execute: " + s));
        }).start();
    }

    private void requestShizuku() {
        try {
            if (!ShizukuHelper.available()) {
                appendLog("Shizuku 未运行。请先在 Shizuku 中启动服务，再重新点击申请。");
                openShizuku();
                return;
            }
            if (ShizukuHelper.granted()) {
                appendLog("MC NPU 已获得 Shizuku 授权。");
                return;
            }
            ShizukuHelper.requestPermission();
            appendLog("已请求授权。Shizuku 应显示 MC NPU 的授权确认界面。");
        } catch (Throwable t) {
            appendLog("Shizuku 请求异常: " + t);
        }
    }

    private void openShizuku() {
        try {
            Intent launch = getPackageManager().getLaunchIntentForPackage("moe.shizuku.privileged.api");
            if (launch == null) {
                appendLog("未安装 Shizuku。");
                return;
            }
            startActivity(launch);
        } catch (Throwable t) {
            appendLog("打开 Shizuku 失败: " + t);
        }
    }

    private void refreshStatus() {
        if (!statusInFlight.compareAndSet(false, true)) return;
        // The keepalive loop may still be waiting for the grant; wake it now that
        // the result is known instead of leaving it to the 10s poll.
        NpuKeepAlive.kick(this);
        new Thread(() -> {
            try {
                String ping = NpuServiceClient.request("PING");
                String npu = NpuServiceClient.request("STATUS");
                String sz = ShizukuHelper.status();
                runOnUiThread(() -> {
                    boolean online = ping.startsWith("PONG");
                    npuState.setText(online ? "● NPU 服务：在线" : "● NPU 服务：离线");
                    npuDetail.setText(online ? npu : "服务未在线，下面显示本地持久诊断日志");
                    shizukuState.setText(sz + "    |    授权结果：" + ShizukuHelper.result());
                });
            } finally {
                statusInFlight.set(false);
            }
        }, "mcnpu-status").start();
    }

    private void refreshLogOnly() {
        if (log == null) return;
        String s = readLocalLog();
        if (s.isEmpty()) s = "暂无持久日志";
        lastServiceLog = s;
        log.setText(s);
        scrollLogToBottom();
    }

    private void refreshServiceLogIncremental() {
        if (log == null) return;
        new Thread(() -> {
            String current = readLocalLog();
            if (current.isEmpty() || current.equals(lastServiceLog)) return;
            final String delta;
            if (!lastServiceLog.isEmpty() && current.startsWith(lastServiceLog)) delta = current.substring(lastServiceLog.length());
            else delta = "\n[日志重新同步]\n" + current;
            lastServiceLog = current;
            runOnUiThread(() -> { log.append(delta); trimVisibleLog(); scrollLogToBottom(); });
        }).start();
    }

    private String readLocalLog() {
        try (FileInputStream in = openFileInput("mcnpu.log");
             ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buf = new byte[4096];
            int n;
            while ((n = in.read(buf)) != -1) out.write(buf, 0, n);
            String s = out.toString(StandardCharsets.UTF_8);
            if (s.length() > 20000) s = s.substring(s.length() - 20000);
            return s;
        } catch (Throwable ignored) {
            return "";
        }
    }

    private String fullDiagnostic() {
        String now = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(new Date());
        return "MC NPU DIAGNOSTIC " + now + "\n"
                + "PING: " + NpuServiceClient.request("PING") + "\n"
                + "STATUS: " + NpuServiceClient.request("STATUS") + "\n"
                + "SHIZUKU: " + ShizukuHelper.status() + "\n"
                + "ADD_MAX: " + NpuServiceClient.request("CAPABILITIES") + "\n"
                + "PROBE: " + lastProbeLine() + "\n"
                + "KEEPALIVE: " + NpuKeepAlive.lastReport() + "\n"
                + "AUTH_RESULT: " + ShizukuHelper.result() + "\n\n"
                + "--- PERSISTENT SERVICE LOG ---\n"
                + readLocalLog()
                + "\n--- UI SESSION LOG ---\n"
                + sessionLogSnapshot();
    }

    /**
     * The last ADD_PROBE line in the service log, or why there isn't one.
     *
     * The probe is the only thing that knows the real ADD ceiling, and it is the
     * number the whole way-splitting depends on. It used to be reachable only by
     * reading thousands of log lines, so it was effectively never read.
     */
    private String lastProbeLine() {
        try {
            String log = readLocalLog();
            String found = "";
            for (String line : log.split("\n")) {
                if (line.contains("ADD_PROBE") || line.startsWith(" fp32 ") || line.startsWith(" fp16 ")) {
                    found = line;
                }
            }
            return found.isEmpty() ? "none - probe has not completed" : found;
        } catch (Throwable t) {
            return "unavailable: " + t.getClass().getSimpleName();
        }
    }

    private void copyLog() {
        new Thread(() -> {
            try {
                String diagnostic = fullDiagnostic();
                runOnUiThread(() -> {
                    ClipboardManager cm = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
                    cm.setPrimaryClip(ClipData.newPlainText("MC NPU diagnostics", diagnostic));
                    Toast.makeText(this, "诊断日志已复制，可以直接粘贴给 GPT", Toast.LENGTH_SHORT).show();
                });
            } catch (Throwable t) {
                runOnUiThread(() -> Toast.makeText(this, "复制日志失败: " + t.getClass().getSimpleName(), Toast.LENGTH_SHORT).show());
            }
        }).start();
    }

    private void shareLog() {
        new Thread(() -> {
            try {
                String diagnostic = fullDiagnostic();
                runOnUiThread(() -> {
                    Intent i = new Intent(Intent.ACTION_SEND);
                    i.setType("text/plain");
                    i.putExtra(Intent.EXTRA_SUBJECT, "MC NPU 诊断日志");
                    i.putExtra(Intent.EXTRA_TEXT, diagnostic);
                    startActivity(Intent.createChooser(i, "发送 MC NPU 诊断日志"));
                });
            } catch (Throwable t) {
                runOnUiThread(() -> Toast.makeText(this, "分享日志失败: " + t.getClass().getSimpleName(), Toast.LENGTH_SHORT).show());
            }
        }).start();
    }

    private void appendRawLogDelta(String delta) {
        if (log == null || delta == null || delta.isEmpty()) return;
        log.append(delta);
        trimVisibleLog();
        scrollLogToBottom();
    }

    private void appendLog(String s) {
        if (log == null) return;
        String line = "\n[" + new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date()) + "] " + s;
        // Keep everything the screen ever showed, not just what still fits in the TextView.
        // The view is trimmed to the last 20000 chars for rendering, but copy/share must not
        // lose the beginning of the session - imported test results showed on screen and were
        // then missing from the clipboard, because only the persistent file was being copied.
        synchronized (sessionLog) {
            sessionLog.append(line);
            if (sessionLog.length() > SESSION_LOG_CAP) {
                sessionLog.delete(0, sessionLog.length() - SESSION_LOG_CAP);
            }
        }
        appendRawLogDelta(line);
    }

    private static String sessionLogSnapshot() {
        synchronized (sessionLog) {
            return sessionLog.length() == 0 ? "(本会话无本地事件)" : sessionLog.toString();
        }
    }

    private void trimVisibleLog() {
        if (log == null) return;
        CharSequence cs = log.getText();
        if (cs.length() <= 20000) return;
        log.setText(cs.subSequence(cs.length() - 20000, cs.length()));
    }

    private void scrollLogToBottom() {
        if (logScroll == null) return;
        logScroll.post(() -> logScroll.fullScroll(ScrollView.FOCUS_DOWN));
    }
}