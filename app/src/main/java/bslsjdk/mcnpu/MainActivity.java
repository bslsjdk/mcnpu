package bslsjdk.mcnpu;

import android.app.Activity;
import android.os.Bundle;
import android.content.Intent;
import android.os.Build;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.net.Uri;
import android.provider.OpenableColumns;
import org.json.JSONArray;
import org.json.JSONObject;
import android.widget.*;
import android.text.method.ScrollingMovementMethod;
import java.io.FileInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public final class MainActivity extends Activity {
    private TextView npuState, npuDetail, shizukuState, log;
    private ScrollView logScroll;
    private final Object logLock = new Object();
    private String lastServiceLog = "";
    private final ScheduledExecutorService statusExecutor = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread t = new Thread(r, "mcnpu-ui-status");
        t.setDaemon(true);
        return t;
    });
    private final AtomicBoolean statusInFlight = new AtomicBoolean();


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
        findViewById(R.id.shizukuOpen).setOnClickListener(v -> openShizuku());
        findViewById(R.id.shizukuRequest).setOnClickListener(v -> requestShizuku());
        findViewById(R.id.copyLog).setOnClickListener(v -> copyLog());
        findViewById(R.id.shareLog).setOnClickListener(v -> shareLog());
        findViewById(R.id.refreshLog).setOnClickListener(v -> refreshServiceLogIncremental());
        refreshLogOnly();

        startNpuService();
    }

    @Override protected void onResume() {
        super.onResume();
        statusExecutor.scheduleAtFixedRate(this::refreshStatus, 0, 1, TimeUnit.SECONDS);
    }

    @Override protected void onPause() {
        statusExecutor.shutdownNow();
        super.onPause();
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
        if (requestCode != PICK_TEST_FILE || resultCode != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        new Thread(() -> runImportedTest(uri)).start();
    }

    private void runImportedTest(Uri uri) {
        try {
            String json;
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
            StringBuilder result = new StringBuilder();
            result.append("=== IMPORTED NPU TEST ===\n")
                  .append("task=").append(task).append("\n")
                  .append("cases=").append(cases.length()).append("\n");
            int pass = 0;
            for (int k = 0; k < cases.length(); k++) {
                JSONObject item = cases.getJSONObject(k);
                JSONArray aa = item.getJSONArray("a");
                JSONArray bb = item.getJSONArray("b");
                if (!"add".equalsIgnoreCase(task) || aa.length() != bb.length() || aa.length() == 0 || aa.length() > 1024) {
                    result.append("CASE ").append(k).append(": INVALID\n");
                    continue;
                }
                StringBuilder a = new StringBuilder(), b = new StringBuilder();
                for (int i=0;i<aa.length();i++) {
                    if(i>0){a.append(',');b.append(',');}
                    double av = aa.getDouble(i), bv = bb.getDouble(i);\n                    if (!Double.isFinite(av) || !Double.isFinite(bv) || av > Float.MAX_VALUE || av < -Float.MAX_VALUE || bv > Float.MAX_VALUE || bv < -Float.MAX_VALUE) throw new IllegalArgumentException("非有限或超出 float 范围");\n                    a.append(Float.toString((float) av)); b.append(Float.toString((float) bv));
                }
                String reply = NpuServiceClient.request("ADD " + a + "|" + b);
                boolean ok = reply.startsWith("OK HTP_GRAPH_EXECUTE");
                if(ok) pass++;
                result.append("CASE ").append(k).append(": ").append(reply).append("\n");
            }
            result.append("SUMMARY pass=").append(pass).append("/")
                  .append(cases.length()).append(" NPU=HTP V73\n");
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
            if (!lastServiceLog.isEmpty() && current.startsWith(lastServiceLog)) {
                delta = current.substring(lastServiceLog.length());
            } else {
                // 服务日志被截断/轮转时，从当前尾部重新同步，而不是静默清空。
                delta = "\n[日志重新同步]\n" + current;
            }
            lastServiceLog = current;
            runOnUiThread(() -> {
                log.append(delta);
                trimVisibleLog();
                scrollLogToBottom();
            });
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
                + "AUTH_RESULT: " + ShizukuHelper.result() + "\n\n"
                + "--- PERSISTENT SERVICE LOG ---\n"
                + readLocalLog();
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
        String line = "\n[" +
                new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date()) + "] " + s;
        appendRawLogDelta(line);
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
    }}
