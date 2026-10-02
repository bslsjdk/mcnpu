package bslsjdk.mcnpu;

import android.app.Activity;
import android.os.Bundle;
import android.content.Intent;
import android.os.Build;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.widget.*;
import java.io.FileInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public final class MainActivity extends Activity {
    private TextView npuState, npuDetail, shizukuState, log;
    private final android.os.Handler handler = new android.os.Handler(android.os.Looper.getMainLooper());
    private final Runnable refresher = new Runnable() {
        @Override public void run() {
            refreshStatus();
            handler.postDelayed(this, 2000);
        }
    };

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        ShizukuHelper.init();
        ShizukuHelper.setCallback(this::refreshStatus);
        setContentView(R.layout.activity_main);

        npuState = findViewById(R.id.npuState);
        npuDetail = findViewById(R.id.npuDetail);
        shizukuState = findViewById(R.id.shizukuState);
        log = findViewById(R.id.log);

        findViewById(R.id.start).setOnClickListener(v -> startNpuService());
        findViewById(R.id.test).setOnClickListener(v -> runSmoke());
        findViewById(R.id.shizukuOpen).setOnClickListener(v -> openShizuku());
        findViewById(R.id.shizukuRequest).setOnClickListener(v -> requestShizuku());
        findViewById(R.id.copyLog).setOnClickListener(v -> copyLog());
        findViewById(R.id.shareLog).setOnClickListener(v -> shareLog());

        startNpuService();
    }

    @Override protected void onResume() {
        super.onResume();
        handler.removeCallbacks(refresher);
        handler.post(refresher);
    }

    @Override protected void onPause() {
        handler.removeCallbacks(refresher);
        super.onPause();
    }

    private void startNpuService() {
        try {
            Intent i = new Intent(this, NpuService.class);
            if (Build.VERSION.SDK_INT >= 26) startForegroundService(i);
            else startService(i);
            appendLog("请求启动独立 NPU 服务");
            handler.postDelayed(this::refreshStatus, 800);
        } catch (Throwable t) {
            appendLog("启动失败: " + t);
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
        new Thread(() -> {
            String ping = NpuServiceClient.request("PING");
            String npu = NpuServiceClient.request("STATUS");
            String sz = ShizukuHelper.status();
            String localLog = readLocalLog();
            runOnUiThread(() -> {
                boolean online = ping.startsWith("PONG");
                npuState.setText(online ? "● NPU 服务：在线" : "● NPU 服务：离线");
                npuDetail.setText(online ? npu : "服务未在线，下面显示本地持久诊断日志");
                shizukuState.setText(sz + "    |    授权结果：" + ShizukuHelper.result());

                String now = new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date());
                log.setText("[" + now + "] PING    " + ping
                        + "\nSTATUS  " + npu
                        + "\nSHIZUKU " + sz
                        + "\n\n--- MC NPU 持久日志 ---\n"
                        + (localLog.isEmpty() ? "暂无日志" : localLog));
                log.setSelection(log.length());
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
        try {
            ClipboardManager cm = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            cm.setPrimaryClip(ClipData.newPlainText("MC NPU diagnostics", fullDiagnostic()));
            Toast.makeText(this, "诊断日志已复制，可以直接粘贴给 GPT", Toast.LENGTH_SHORT).show();
        } catch (Throwable t) {
            Toast.makeText(this, "复制日志失败: " + t.getClass().getSimpleName(), Toast.LENGTH_SHORT).show();
        }
    }

    private void shareLog() {
        try {
            Intent i = new Intent(Intent.ACTION_SEND);
            i.setType("text/plain");
            i.putExtra(Intent.EXTRA_SUBJECT, "MC NPU 诊断日志");
            i.putExtra(Intent.EXTRA_TEXT, fullDiagnostic());
            startActivity(Intent.createChooser(i, "发送 MC NPU 诊断日志"));
        } catch (Throwable t) {
            Toast.makeText(this, "分享日志失败: " + t.getClass().getSimpleName(), Toast.LENGTH_SHORT).show();
        }
    }

    private void appendLog(String s) {
        if (log == null) return;
        String old = log.getText().toString();
        log.setText(old + "\n[" +
                new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date()) + "] " + s);
        log.setSelection(log.length());
    }
}
