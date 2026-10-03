package bslsjdk.mcnpu;

import android.app.*;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ServiceInfo;
import android.os.IBinder;
import android.os.Build;
import java.io.*;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class NpuService extends Service {
    private static final int IPC_PORT = 38761;
    private static final int NOTIFICATION_ID = 1001;
    private static final String CHANNEL = "mcnpu";
    private final ExecutorService clients = Executors.newFixedThreadPool(8);
    private volatile boolean running;
    private ServerSocket server;
    private final java.util.concurrent.atomic.AtomicBoolean serverLoopStarted = new java.util.concurrent.atomic.AtomicBoolean();
    private final java.util.concurrent.atomic.AtomicInteger workerEpoch = new java.util.concurrent.atomic.AtomicInteger();
    private volatile long workerBeatMs;
    private volatile String workerPhase = "STOPPED";
    private volatile boolean listenerUp;
    /** accept 循环是否真的在跑（端口 bound != 服务可用）。 */
    private volatile boolean acceptAlive;
    private static final long WORKER_STALL_MS = 15_000L;

    @Override public void onCreate() {
        super.onCreate();
        try {
            createChannel();
            Notification n = notification("正在初始化 QNN / HTP V73");
            if (Build.VERSION.SDK_INT >= 34)
                startForeground(NOTIFICATION_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
            else startForeground(NOTIFICATION_ID, n);

            if (Build.VERSION.SDK_INT >= 33) {
                boolean nearby = checkSelfPermission("android.permission.NEARBY_WIFI_DEVICES")
                        == PackageManager.PERMISSION_GRANTED;
                log("NEARBY_WIFI_DEVICES=" + (nearby ? "GRANTED" : "NOT_GRANTED"));
            }

            running = true;
            ensureServerLoop("onCreate");
        } catch (Throwable t) {
            log("服务启动失败: " + t);
            stopSelf();
        }
    }

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        if (!running) running = true;
        long stall = workerBeatMs == 0 ? Long.MAX_VALUE
                : System.currentTimeMillis() - workerBeatMs;
        boolean listenerHealthy = acceptAlive && server != null && !server.isClosed();
        log("服务 startCommand startId=" + startId + " worker=" + serverLoopStarted.get()
                + " phase=" + workerPhase + " stallMs=" + stall
                + " listenerHealthy=" + listenerHealthy
                + " server=" + (server != null && !server.isClosed()));
        if (serverLoopStarted.get() && !listenerHealthy && stall > WORKER_STALL_MS) {
            log("IPC worker STALL stallMs=" + stall + " phase=" + workerPhase
                    + " -> advance epoch and restart");
            workerEpoch.incrementAndGet();
            running = false;
            try { if (server != null) server.close(); } catch (Throwable ignored) {}
            serverLoopStarted.set(false);
            workerPhase = "STALL_RESET";
            running = true;
        }
        ensureServerLoop("onStartCommand");
        return START_STICKY;
    }

    private void ensureServerLoop(String reason) {
        if (!running) return;
        if (!serverLoopStarted.compareAndSet(false, true)) return;
        final int myEpoch = workerEpoch.get();
        workerBeatMs = System.currentTimeMillis();
        workerPhase = "STARTING";
        log("IPC worker START reason=" + reason);
        clients.execute(() -> {
            try {
                workerPhase = "RUNNING";
                serverLoop(myEpoch);
            } catch (Throwable t) {
                log("IPC worker CRASH: " + t.getClass().getName() + ": " + t.getMessage());
            } finally {
                serverLoopStarted.set(false);
                workerPhase = "EXIT";
                workerBeatMs = System.currentTimeMillis();
                log("IPC worker EXIT running=" + running);
                if (running) {
                    new Thread(() -> {
                        try { Thread.sleep(250); }
                        catch (InterruptedException e) { Thread.currentThread().interrupt(); return; }
                        ensureServerLoop("worker-restart");
                    }, "mcnpu-ipc-supervisor").start();
                }
            }
        });
    }

    private void serverLoop(int myEpoch) {
        log("服务线程启动 epoch=" + myEpoch);

        // 先建立控制面监听，再初始化 QNN。即使 native init 卡住，客户端也能连上获取状态。
        ServerSocket prebound = null;
        try {
            prebound = new ServerSocket();
            prebound.setReuseAddress(true);
            InetAddress loopback = InetAddress.getByName("127.0.0.1");
            prebound.bind(new InetSocketAddress(loopback, IPC_PORT), 16);
            server = prebound;
            listenerUp = true;
            workerPhase = "LISTENING_INIT";
            workerBeatMs = System.currentTimeMillis();
            log("IPC 监听 READY_FOR_INIT " + loopback.getHostAddress() + ":" + IPC_PORT);
            earlyConnectProbe(loopback);
        } catch (Throwable t) {
            workerPhase = "BIND_FAILED";
            workerBeatMs = System.currentTimeMillis();
            log("IPC early bind 失败: " + t);
            try { if (prebound != null) prebound.close(); } catch (Throwable ignored) {}
            listenerUp = false;
            server = null;
        }

        log("QNN/HTP init BEGIN");
        workerBeatMs = System.currentTimeMillis();
        long initStart = System.nanoTime();
        boolean ok = NpuRuntime.init(getApplicationContext());
        workerBeatMs = System.currentTimeMillis();
        workerPhase = ok ? "INIT_READY" : "INIT_FAILED";
        log("QNN/HTP init END ok=" + ok + " elapsed_ms=" + ((System.nanoTime() - initStart) / 1_000_000.0));
        updateNotification(ok ? "HTP V73 已就绪" : "HTP 初始化失败");
        log(ok ? "QNN/HTP 初始化成功" : "QNN/HTP 初始化失败: " + NpuRuntime.getLastError());
        while (running && workerEpoch.get() == myEpoch) {
            workerBeatMs = System.currentTimeMillis();
            ServerSocket ss = null;
            try {
                InetAddress loopback = InetAddress.getByName("127.0.0.1");
                if (server != null && !server.isClosed()) {
                    ss = server;
                } else {
                    ss = new ServerSocket();
                    ss.setReuseAddress(true);
                    ss.bind(new InetSocketAddress(loopback, IPC_PORT), 16);
                    server = ss;
                    listenerUp = true;
                }

                if ("INIT_READY".equals(workerPhase) || "INIT_FAILED".equals(workerPhase)) {
                    // 保留初始化结果，不让正常 accept 循环把 phase 倒退。
                } else {
                    workerPhase = "LISTENING";
                }
                workerBeatMs = System.currentTimeMillis();
                log("IPC 监听 LOOPBACK " + ss.getInetAddress().getHostAddress() + ":" + ss.getLocalPort());
                updateNotification(ok ? "MC NPU 在线 · HTP V73" : "MC NPU 在线 · HTP 初始化失败");
                acceptAlive = true;
                boolean selfTestDone = false;
                while (running && workerEpoch.get() == myEpoch && server == ss && !ss.isClosed()) {
                    if (!selfTestDone) {
                        // 只有 accept 循环开跑之后 PONG 才可能被处理；放在 init 之前测只会超时。
                        selfTestDone = true;
                        selfTestLoopback();
                    }
                    try {
                        Socket socket = ss.accept();
                        workerBeatMs = System.currentTimeMillis();
                        log("IPC ACCEPT " + socket.getRemoteSocketAddress());
                        try {
                            clients.execute(() -> handle(socket));
                        } catch (Throwable t) {
                            try { socket.close(); } catch (Throwable ignored) {}
                            log("IPC client dispatch failed: " + t);
                        }
                    } catch (Throwable t) {
                        if (!running || ss.isClosed()) break;
                        log("IPC accept exception; keeping listener alive: " + t);
                        try {
                            Thread.sleep(50);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            break;
                        }
                    }
                }
            } catch (Throwable t) {
                if (running) {
                    workerPhase = "BIND_OR_ACCEPT_FAILED";
                    workerBeatMs = System.currentTimeMillis();
                    log("IPC accept/bind 失败: " + t);
                    updateNotification("MC NPU: IPC retrying");
                }
            } finally {
                acceptAlive = false;
                if (server == ss) {
                    listenerUp = false;
                    server = null;
                }
                try { if (ss != null) ss.close(); } catch (Throwable ignored) {}
            }

            if (running) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }

        if (workerEpoch.get() != myEpoch) {
            acceptAlive = false;
            try { if (prebound != null && prebound != server) prebound.close(); } catch (Throwable ignored) {}
            log("IPC worker EPOCH_EXIT myEpoch=" + myEpoch + " currentEpoch=" + workerEpoch.get());
        }
        log("IPC 循环结束");
    }

    /**
     * 初始化前的连通性探针：只验证 loopback 上 TCP connect 能成功（即端口已 bind/listen）。
     * 不要求服务端回包 —— accept 循环要等 QNN 初始化结束后才启动，
     * 此时发 PING 只会得到 SocketTimeoutException，不能据此判断 loopback 被系统拦截。
     */
    private void earlyConnectProbe(InetAddress loopback) {
        new Thread(() -> {
            long t0 = System.nanoTime();
            try (Socket t = new Socket()) {
                t.connect(new InetSocketAddress(loopback, IPC_PORT), 2000);
                log("IPC EARLY_PROBE CONNECT_OK " + loopback.getHostAddress() + ":" + IPC_PORT
                        + " elapsed_us=" + ((System.nanoTime() - t0) / 1000)
                        + " (accept loop starts after QNN init; no reply expected)");
            } catch (Throwable e) {
                log("IPC EARLY_PROBE CONNECT_FAIL " + loopback.getHostAddress() + ":" + IPC_PORT
                        + " error=" + e.getClass().getName() + " msg=" + e.getMessage());
            }
        }, "mcnpu-ipc-early-probe").start();
    }

    private void selfTestLoopback() {
        new Thread(() -> {
            long t0 = System.nanoTime();
            try (Socket t = new Socket()) {
                InetAddress lo = InetAddress.getByName("127.0.0.1");
                t.connect(new InetSocketAddress(lo, IPC_PORT), 2000);
                t.setSoTimeout(2000);
                t.getOutputStream().write("PING\n".getBytes(java.nio.charset.StandardCharsets.UTF_8));
                t.getOutputStream().flush();
                BufferedReader br = new BufferedReader(
                        new InputStreamReader(t.getInputStream(), java.nio.charset.StandardCharsets.UTF_8));
                String reply = br.readLine();
                log("IPC SELFTEST PASS addr=" + lo.getHostAddress()
                        + " localhost=" + InetAddress.getLoopbackAddress().getHostAddress()
                        + " all=" + java.util.Arrays.toString(InetAddress.getAllByName("localhost"))
                        + " reply=" + reply
                        + " elapsed_us=" + ((System.nanoTime() - t0) / 1000));
            } catch (Throwable e) {
                log("IPC SELFTEST FAIL addr=127.0.0.1"
                        + " localhost=" + safeLoopbackAddress()
                        + " all=" + safeLocalhostAddresses()
                        + " error=" + e.getClass().getName()
                        + " msg=" + e.getMessage());
            }
        }, "mcnpu-ipc-selftest").start();
    }

    private static String safeLoopbackAddress() {
        try {
            return InetAddress.getLoopbackAddress().getHostAddress();
        } catch (Throwable t) {
            return "<error:" + t.getClass().getSimpleName() + ">";
        }
    }

    private static String safeLocalhostAddresses() {
        try {
            return java.util.Arrays.toString(InetAddress.getAllByName("localhost"));
        } catch (Throwable t) {
            return "<error:" + t.getClass().getSimpleName() + ">";
        }
    }

    private void handle(Socket socket) {
        try (Socket s = socket;
             BufferedReader in = new BufferedReader(new InputStreamReader(s.getInputStream()));
             BufferedWriter out = new BufferedWriter(new OutputStreamWriter(s.getOutputStream()))) {
            // 小包请求不要撞上 Nagle + delayed-ACK（实测 p99 往返 ~50ms，p50 仅 ~1.4ms）
            try { s.setTcpNoDelay(true); } catch (Throwable ignored) {}
            String line;
            while ((line = in.readLine()) != null) {
                String cmd = line.trim();
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
            if (as.length == 0 || as.length != bs.length || as.length > 16384) return "ERR ADD_SIZE";
            float[] a = new float[as.length], b = new float[bs.length];
            for (int i = 0; i < as.length; i++) {
                a[i] = Float.parseFloat(as[i]);
                b[i] = Float.parseFloat(bs[i]);
                if (!Float.isFinite(a[i]) || !Float.isFinite(b[i])) return "ERR ADD_NONFINITE";
            }
            String result = NpuRuntime.add(a, b);
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
        log("服务停止: running=false");
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