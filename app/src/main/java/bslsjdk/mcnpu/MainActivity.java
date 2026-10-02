package bslsjdk.mcnpu;

import android.app.Activity;
import android.os.Bundle;
import android.content.Intent;
import android.os.Build;
import android.widget.*;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public final class MainActivity extends Activity {
    private TextView npuState, shizukuState, log;
    private final android.os.Handler handler = new android.os.Handler(android.os.Looper.getMainLooper());
    private final Runnable refresher = new Runnable() {
        @Override public void run() { refreshStatus(); handler.postDelayed(this, 2000); }
    };

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        ShizukuHelper.init();
        ShizukuHelper.setCallback(this::refreshStatus);
        setContentView(R.layout.activity_main);

        npuState = findViewById(R.id.npuState);
        shizukuState = findViewById(R.id.shizukuState);
        log = findViewById(R.id.log);

        findViewById(R.id.start).setOnClickListener(v -> startNpuService());
        findViewById(R.id.test).setOnClickListener(v -> runSmoke());
        findViewById(R.id.shizukuOpen).setOnClickListener(v -> openShizuku());
        findViewById(R.id.shizukuRequest).setOnClickListener(v -> requestShizuku());

        startNpuService();
        refreshStatus();
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
            appendLog("已请求启动独立 NPU 服务");
        } catch (Throwable t) {
            appendLog("启动失败: " + t);
        }
    }

    private void runSmoke() {
        new Thread(() -> {
            String s = NpuServiceClient.request("SMOKE");
            runOnUiThread(() -> appendLog("HTP 图执行: " + s));
        }).start();
    }

    private void requestShizuku() {
        try {
            if (!ShizukuHelper.available()) {
                appendLog("Shizuku 未运行，正在打开 Shizuku");
                openShizuku();
                return;
            }
            if (ShizukuHelper.granted()) {
                appendLog("MC NPU 已获得 Shizuku 授权");
                return;
            }
            ShizukuHelper.requestPermission();
            appendLog("已发起授权请求，请在 Shizuku 弹出的“允许 MC NPU 使用 Shizuku”窗口中允许");
        } catch (Throwable t) {
            appendLog("Shizuku 请求异常: " + t);
        }
    }

    private void openShizuku() {
        try {
            Intent launch = getPackageManager().getLaunchIntentForPackage("moe.shizuku.privileged.api");
            if (launch == null) {
                appendLog("未安装 Shizuku");
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
            runOnUiThread(() -> {
                npuState.setText(ping.startsWith("PONG") ? "● NPU 服务：在线" : "● NPU 服务：离线");
                shizukuState.setText(sz);
                log.setText("时间 " + new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date())
                        + "\nPING    " + ping
                        + "\nSTATUS  " + npu
                        + "\nShizuku " + sz
                        + "\n授权结果 " + ShizukuHelper.result());
            });
        }).start();
    }

    private void appendLog(String s) {
        if (log == null) return;
        log.setText(log.getText() + "\n[" +
                new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date()) + "] " + s);
    }
}