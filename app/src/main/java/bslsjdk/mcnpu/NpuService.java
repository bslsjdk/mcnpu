package bslsjdk.mcnpu;

import android.app.*;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.IBinder;
import android.net.LocalServerSocket;
import android.net.LocalSocket;
import android.net.Credentials;
import android.os.Build;
import java.io.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class NpuService extends Service {
    private static final String SOCKET_NAME = "mcnpu_ipc_v1";
    private static final int NOTIFICATION_ID = 1001;
    private static final String CHANNEL = "mcnpu";
    private static final String PREFS = "ipc";
    private static final String PREF_TRUSTED_PACKAGES = "trusted_packages";
    private final ExecutorService clients = Executors.newFixedThreadPool(8);
    private volatile boolean running;
    private LocalServerSocket server;

    @Override public void onCreate() {
        super.onCreate();
        try {
            createChannel();
            Notification n = notification("正在初始化 QNN / HTP V73");
            if (Build.VERSION.SDK_INT >= 34)
                startForeground(NOTIFICATION_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
            else startForeground(NOTIFICATION_ID, n);
            running = true;
            clients.execute(this::serverLoop);
        } catch (Throwable t) {
            log("服务启动失败: " + t);
            stopSelf();
        }
    }

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        return START_STICKY;
    }

    private void serverLoop() {
        log("服务线程启动");
        boolean ok = NpuRuntime.init(getApplicationContext());
        updateNotification(ok ? "HTP V73 已就绪" : "HTP 初始化失败");
        log(ok ? "QNN/HTP 初始化成功" : "QNN/HTP 初始化失败: " + NpuRuntime.getLastError());
        try {
            server = new LocalServerSocket(SOCKET_NAME);
            log("IPC 监听 LOCAL_ABSTRACT " + SOCKET_NAME);
            updateNotification(ok ? "MC NPU 在线 · HTP V73" : "MC NPU 在线 · HTP 初始化失败");
            while (running) {
                LocalSocket socket = server.accept();
                if (!isTrustedPeer(socket)) {
                    log("IPC rejected untrusted peer");
                    try { socket.close(); } catch (Throwable ignored) {}
                    continue;
                }
                clients.execute(() -> handle(socket));
            }
        } catch (Throwable t) {
            log("IPC 停止: " + t);
            if (running) updateNotification("MC NPU: IPC stopped");
        }
    }

    private boolean isTrustedPeer(LocalSocket socket) {
        int uid = -1;
        String[] packages = null;
        try {
            Credentials peer = socket.getPeerCredentials();
            uid = peer.getUid();
            if (uid == android.os.Process.myUid()) return true;

            packages = getPackageManager().getPackagesForUid(uid);
            String configured = getSharedPreferences(PREFS, MODE_PRIVATE)
                    .getString(PREF_TRUSTED_PACKAGES, "");
            if (packages != null && !configured.trim().isEmpty()) {
                for (String p : packages) {
                    for (String allowed : configured.split(",")) {
                        if (p.equals(allowed.trim()) && !allowed.trim().isEmpty()) {
                            log("IPC accepted configured peer uid=" + uid + " package=" + p);
                            return true;
                        }
                    }
                }
            }

            log("IPC rejected peer uid=" + uid + " packages=" +
                    (packages == null ? "<none>" : String.join(",", packages)));
        } catch (Throwable t) {
            log("IPC peer credential check failed uid=" + uid + " packages=" +
                    (packages == null ? "<none>" : String.join(",", packages)) + ": " + t);
        }
        return false;
    }

    private void handle(LocalSocket socket) {
        try (LocalSocket s = socket;
             BufferedReader in = new BufferedReader(new InputStreamReader(s.getInputStream()));
             BufferedWriter out = new BufferedWriter(new OutputStreamWriter(s.getOutputStream()))) {
            String line;
            while ((line = in.readLine()) != null) {
                String cmd = line.trim();
                log("IPC <- " + cmd);
                String reply;
                if (cmd.equals("PING")) reply = "PONG MCNPU/1";
                else if (cmd.equals("STATUS")) reply = NpuRuntime.status();
                else if (cmd.equals("SMOKE")) {
                    long t = System.nanoTime();
                    reply = NpuRuntime.smoke() ? "OK HTP_GRAPH_EXECUTE" : "ERR HTP_GRAPH_EXECUTE";
                    log("EXEC SMOKE result=" + reply + " elapsed_ms=" + ((System.nanoTime() - t) / 1_000_000.0));
                } else if (cmd.equals("CAPABILITIES")) {
                    reply = "OK MCNPU/1 backend=HTP_V73 ops=ADD max_elements=1024";
                } else if (cmd.startsWith("EXEC_ADD ")) {
                    reply = handleAdd(cmd.substring(9));
                } else if (cmd.startsWith("ADD ")) {
                    reply = handleAdd(cmd.substring(4));
                } else if (cmd.equals("QUIT")) {
                    reply(out, "BYE");
                    break;
                } else {
                    reply = "ERR UNKNOWN_COMMAND";
                }
                reply(out, reply);
                log("IPC -> " + reply);
            }
        } catch (Throwable t) {
            log("IPC client closed: " + t);
        }
    }

    private String handleAdd(String payload) {
        long t0 = System.nanoTime();
        try {
            String[] parts = payload.split("\\|", -1);
            if (parts.length != 2) return "ERR ADD_FORMAT";
            String[] as = parts[0].split(",", -1);
            String[] bs = parts[1].split(",", -1);
            if (as.length == 0 || as.length != bs.length || as.length > 1024) return "ERR ADD_SIZE";
            float[] a = new float[as.length], b = new float[bs.length];
            for (int i = 0; i < as.length; i++) {
                a[i] = Float.parseFloat(as[i]);
                b[i] = Float.parseFloat(bs[i]);
                if (!Float.isFinite(a[i]) || !Float.isFinite(b[i])) return "ERR ADD_NONFINITE";
            }
            String result = NpuRuntime.add(a, b);
            log("EXEC ADD n=" + a.length + " result=" + result +
                    " elapsed_ms=" + ((System.nanoTime() - t0) / 1_000_000.0));
            return result;
        } catch (Throwable t) {
            String result = "ERR ADD_EXCEPTION " + t.getClass().getSimpleName();
            log("EXEC ADD exception=" + t);
            return result;
        }
    }

    private static void reply(BufferedWriter out, String s) throws IOException {
        out.write(s);
        out.write("\n");
        out.flush();
    }

    private void createChannel() {
        if (Build.VERSION.SDK_INT >= 26) {
            getSystemService(NotificationManager.class).createNotificationChannel(
                    new NotificationChannel(CHANNEL, "MC NPU", NotificationManager.IMPORTANCE_LOW));
        }
    }

    private Notification notification(String text) {
        Notification.Builder b = Build.VERSION.SDK_INT >= 26
                ? new Notification.Builder(this, CHANNEL)
                : new Notification.Builder(this);
        return b.setContentTitle("MC NPU")
                .setContentText(text)
                .setSmallIcon(android.R.drawable.stat_sys_download_done)
                .setOngoing(true)
                .build();
    }

    private void updateNotification(String text) {
        try {
            getSystemService(NotificationManager.class).notify(NOTIFICATION_ID, notification(text));
        } catch (Throwable t) {
            android.util.Log.e("MCNPU", "notification failed", t);
        }
    }

    private synchronized void log(String s) {
        String line = System.currentTimeMillis() + " " + s;
        android.util.Log.i("MCNPU", s);
        try (FileOutputStream out = openFileOutput("mcnpu.log", MODE_APPEND)) {
            out.write((line + "\n").getBytes(java.nio.charset.StandardCharsets.UTF_8));
        } catch (Throwable ignored) {}
    }

    @Override public void onDestroy() {
        running = false;
        try { if (server != null) server.close(); } catch (Throwable ignored) {}
        clients.shutdownNow();
        NpuRuntime.shutdown();
        log("服务停止");
        super.onDestroy();
    }

    @Override public IBinder onBind(Intent intent) {
        return null;
    }
}
