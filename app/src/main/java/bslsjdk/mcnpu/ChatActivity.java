package bslsjdk.mcnpu;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.content.Context;
import android.graphics.Color;
import android.graphics.Typeface;
import android.view.Gravity;
import android.view.View;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import org.json.JSONArray;
import org.json.JSONObject;

public final class ChatActivity extends Activity {
    private static final String PREFS = "chat";
    private static final String HISTORY = "history";
    private LinearLayout messages;
    private ScrollView scroll;
    private EditText input;
    private TextView runtimeState;
    private TextView statusLine;
    private final Handler main = new Handler(Looper.getMainLooper());

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        setContentView(R.layout.activity_chat);

        messages = findViewById(R.id.messages);
        scroll = findViewById(R.id.messagesScroll);
        input = findViewById(R.id.input);
        runtimeState = findViewById(R.id.runtimeState);
        statusLine = findViewById(R.id.statusLine);

        findViewById(R.id.send).setOnClickListener(v -> sendMessage());
        input.setOnEditorActionListener((v, actionId, event) -> {
            if (event != null && event.getKeyCode() == android.view.KeyEvent.KEYCODE_ENTER
                    && event.isShiftPressed()) return false;
            if (event != null && event.getKeyCode() == android.view.KeyEvent.KEYCODE_ENTER) {
                sendMessage();
                return true;
            }
            return false;
        });

        loadHistory();
        updateRuntimeState();
        initLocalRuntime();
    }

    private void initLocalRuntime() {
        new Thread(() -> {
            boolean ok = NpuRuntime.init(getApplicationContext());
            main.post(() -> {
                updateRuntimeState();
                if (ok) statusLine.setText("本地 NPU 在线 · Bonsai 2 PQ2_0 内核待接入");
            });
        }, "mcnpu-init").start();
    }

    @Override protected void onResume() {
        super.onResume();
        updateRuntimeState();
    }

    private void updateRuntimeState() {
        boolean npu = NpuRuntime.isReady();
        runtimeState.setText(npu ? "NPU 在线" : "本地");
        runtimeState.setTextColor(npu ? Color.rgb(22, 120, 75) : Color.rgb(100, 116, 139));
        statusLine.setText(npu
                ? "Bonsai 2 · PQ2_0 · MCNPU HTP V73"
                : "Bonsai 2 · PQ2_0 · 等待本地推理内核");
    }

    private void sendMessage() {
        String text = input.getText().toString().trim();
        if (text.isEmpty()) return;

        addBubble("user", text);
        input.setText("");
        saveHistory();

        /*
         * Deliberately do not fake an AI answer here. The chat shell is ready, but
         * Bonsai 2 inference is not yet wired into this Activity. The next runtime
         * layer will replace this status path with streaming local tokens.
         */
        if (!NpuRuntime.isReady()) {
            addBubble("system",
                    "本地推理内核尚未启动。聊天界面已经就位，Bonsai 2 PQ2_0 推理接入后将在这里流式输出。");
            saveHistory();
            return;
        }

        addBubble("system",
                "MCNPU 已在线，但 Bonsai 2 推理引擎尚未接入聊天通道。当前不会伪造模型输出。");
        saveHistory();
    }

    private void addBubble(String role, String text) {
        TextView bubble = new TextView(this);
        bubble.setText(text);
        bubble.setTextSize(16);
        bubble.setTextColor(Color.rgb(17, 24, 39));
        bubble.setPadding(16, 12, 16, 12);
        bubble.setTypeface(Typeface.DEFAULT, Typeface.NORMAL);
        bubble.setTag(role);

        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.setMargins(8, 6, 8, 6);

        if ("user".equals(role)) {
            bubble.setBackgroundColor(Color.rgb(225, 235, 255));
            lp.gravity = Gravity.END;
        } else if ("system".equals(role)) {
            bubble.setTextSize(13);
            bubble.setTextColor(Color.rgb(71, 85, 105));
            bubble.setBackgroundColor(Color.rgb(238, 241, 245));
            lp.gravity = Gravity.CENTER_HORIZONTAL;
        } else {
            bubble.setBackgroundColor(Color.WHITE);
            lp.gravity = Gravity.START;
        }

        messages.addView(bubble, lp);
        scroll.post(() -> scroll.fullScroll(View.FOCUS_DOWN));
    }

    private void loadHistory() {
        messages.removeAllViews();
        String raw = getSharedPreferences(PREFS, MODE_PRIVATE).getString(HISTORY, "[]");
        try {
            JSONArray arr = new JSONArray(raw);
            for (int i = 0; i < arr.length(); i++) {
                JSONObject m = arr.getJSONObject(i);
                addBubble(m.optString("role", "assistant"), m.optString("text", ""));
            }
        } catch (Throwable ignored) {
            getSharedPreferences(PREFS, MODE_PRIVATE).edit().remove(HISTORY).apply();
        }
    }

    private void saveHistory() {
        JSONArray arr = new JSONArray();
        for (int i = 0; i < messages.getChildCount(); i++) {
            View v = messages.getChildAt(i);
            if (!(v instanceof TextView)) continue;
            TextView t = (TextView) v;
            String role = t.getTag() instanceof String ? (String) t.getTag() : "assistant";
            try {
                JSONObject m = new JSONObject();
                m.put("role", role);
                m.put("text", t.getText().toString());
                arr.put(m);
            } catch (Throwable ignored) {}
        }
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(HISTORY, arr.toString()).apply();
    }
}
