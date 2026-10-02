package bslsjdk.mcnpu;

import android.app.Activity;
import android.os.Bundle;
import android.content.Intent;
import android.os.Build;
import android.widget.*;

public final class MainActivity extends Activity {
    private TextView status;

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(32, 32, 32, 32);

        TextView title = new TextView(this);
        title.setText("MC NPU\nQNN / HTP V73");
        title.setTextSize(24);
        root.addView(title);

        status = new TextView(this);
        status.setText("Service starting...");
        root.addView(status);

        Button start = new Button(this);
        start.setText("启动 NPU 服务");
        start.setOnClickListener(v -> startNpuService());
        root.addView(start);

        Button shizuku = new Button(this);
        shizuku.setText("授权 Shizuku");
        shizuku.setOnClickListener(v -> {
            if (!ShizukuHelper.available()) {
                status.setText("SHIZUKU_OFFLINE\n请先启动 Shizuku");
            } else if (ShizukuHelper.granted()) {
                status.setText("SHIZUKU_GRANTED");
            } else {
                ShizukuHelper.requestPermission();
                status.setText("正在请求 Shizuku 权限...");
            }
        });
        root.addView(shizuku);

        Button test = new Button(this);
        test.setText("检测 HTP");
        test.setOnClickListener(v -> new Thread(() -> {
            String s = NpuRuntime.status();
            runOnUiThread(() -> status.setText(s + "\n" + ShizukuHelper.status()));
        }).start());
        root.addView(test);

        setContentView(root);
        startNpuService();
    }

    @Override protected void onResume() {
        super.onResume();
        if (status != null) {
            status.setText("MC NPU: " + (NpuRuntime.isReady() ? "READY" : "starting")
                    + "\n" + ShizukuHelper.status());
        }
    }

    private void startNpuService() {
        Intent i = new Intent(this, NpuService.class);
        if (Build.VERSION.SDK_INT >= 26) startForegroundService(i);
        else startService(i);
        status.setText("MC NPU service starting...\n" + ShizukuHelper.status());
    }
}
