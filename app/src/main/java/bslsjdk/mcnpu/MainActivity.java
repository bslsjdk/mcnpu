package bslsjdk.mcnpu;

import android.app.Activity;
import android.os.Bundle;
import android.content.Intent;
import android.os.Build;
import android.widget.*;
import android.view.Gravity;

public final class MainActivity extends Activity {
    private TextView status;

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);

        ShizukuHelper.init();

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(32, 32, 32, 32);

        TextView title = new TextView(this);
        title.setText("MC NPU\nQNN / HTP V73");
        title.setTextSize(24);
        root.addView(title);

        Button start = new Button(this);
        start.setText("启动 NPU 服务");
        start.setOnClickListener(v -> startNpuService());
        root.addView(start);

        Button shizukuOpen = new Button(this);
        shizukuOpen.setText("打开 Shizuku");
        shizukuOpen.setOnClickListener(v -> openShizuku());
        root.addView(shizukuOpen);

        Button shizuku = new Button(this);
        shizuku.setText("向 Shizuku 请求 MC NPU 授权");
        shizuku.setOnClickListener(v -> requestShizuku());
        root.addView(shizuku);

        Button test = new Button(this);
        test.setText("检测 HTP / 服务");
        test.setOnClickListener(v -> new Thread(() -> {
            String s = NpuServiceClient.request("STATUS");
            runOnUiThread(() -> setStatus(s + "\n" + ShizukuHelper.status()));
        }).start());
        root.addView(test);

        status = new TextView(this);
        status.setGravity(Gravity.TOP);
        status.setText("正在启动服务...");
        status.setPadding(0, 24, 0, 0);
        root.addView(status);

        setContentView(root);
        startNpuService();
        refreshStatus();
    }

    @Override protected void onResume() {
        super.onResume();
        if (status != null) refreshStatus();
    }

    private void startNpuService() {
        try {
            Intent i = new Intent(this, NpuService.class);
            if (Build.VERSION.SDK_INT >= 26) startForegroundService(i);
            else startService(i);
            setStatus("正在启动 MC NPU...\n" + ShizukuHelper.status());
        } catch (Throwable t) {
            setStatus("启动服务失败\n" + t.getClass().getSimpleName() + ": " + t.getMessage());
        }
    }

    private void requestShizuku() {
        try {
            if (!ShizukuHelper.available()) {
                setStatus("Shizuku 未运行。先点“打开 Shizuku”启动它，然后回来点授权。");
                return;
            }
            if (ShizukuHelper.granted()) {
                setStatus("MC NPU 已获得 Shizuku 授权。");
                return;
            }
            ShizukuHelper.requestPermission();
            setStatus("已向 Shizuku 发起 MC NPU 授权请求，请在 Shizuku 的授权窗口/应用列表中允许。");
        } catch (Throwable t) {
            setStatus("Shizuku 请求失败\n" + t.getClass().getSimpleName() + ": " + t.getMessage());
        }
    }

    private void openShizuku() {
        try {
            Intent launch = getPackageManager().getLaunchIntentForPackage("moe.shizuku.privileged.api");
            if (launch == null) {
                setStatus("未找到 Shizuku，请先安装 Shizuku。");
                return;
            }
            launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(launch);
        } catch (Throwable t) {
            setStatus("打开 Shizuku 失败\n" + t.getClass().getSimpleName() + ": " + t.getMessage());
        }
    }

    private void refreshStatus() {
        new Thread(() -> {
            String s = NpuServiceClient.request("STATUS");
            runOnUiThread(() -> setStatus(s + "\n" + ShizukuHelper.status()));
        }).start();
    }

    private void setStatus(String s) {
        if (status != null) status.setText(s);
    }
}
