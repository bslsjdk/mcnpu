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
        LinearLayout root=new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(32,32,32,32);
        TextView title=new TextView(this);
        title.setText("MC NPU\nQNN / HTP V73");
        title.setTextSize(24);
        root.addView(title);
        status=new TextView(this);
        status.setText("Service starting...");
        root.addView(status);
        Button start=new Button(this);
        start.setText("启动 NPU 服务");
        start.setOnClickListener(v -> startNpuService());
        root.addView(start);
        Button test=new Button(this);
        test.setText("检测 HTP");
        test.setOnClickListener(v -> new Thread(() -> {
            String s=NpuRuntime.status();
            runOnUiThread(() -> status.setText(s));
        }).start());
        root.addView(test);
        setContentView(root);
        startNpuService();
    }
    private void startNpuService() {
        Intent i=new Intent(this,NpuService.class);
        if(Build.VERSION.SDK_INT>=26) startForegroundService(i); else startService(i);
        status.setText("MC NPU service starting...");
    }
}
