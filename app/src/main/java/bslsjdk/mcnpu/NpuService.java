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
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public final class NpuService extends Service {
    /**
     * A n=16384 ADD command is ~360 KB of decimal text. The old 256 KB ceiling
     * rejected it before it ever reached the HTP.
     */
    private static final int MAX_LINE_BYTES = 8 * 1024 * 1024;
    /**
     * How long a connected client may stay silent before its handler thread is
     * given back. Pooled connections that are opened but never written to used to
     * hold a slot until the client's own timeout fired - 15s in the field - and a
     * handful of those is a large share of a small pool.
     */
    private static final int IDLE_TIMEOUT_MS = 30_000;
    /** Cap on the raw BINADD body: 16384 elements x 2 arrays x 4 bytes x 4096 cases would be 512 MB. */
    private static final long MAX_BIN_BODY = 64L * 1024 * 1024;
    /** Read granularity for control lines; also the pushback buffer size. */
    private static final int LINE_BLOCK = 8192;
    private static final int IPC_PORT = 38761;
    private static final int NOTIFICATION_ID = 1001;
    private static final String CHANNEL = "mcnpu";
    // Bound IPC work so connection floods cannot exhaust Android threads/file descriptors.
    // 8 active + 8 queued was the whole ceiling, and the 9x9 import opens one
    // socket per way plus the poller and the diagnostic, so it hit REJECT and the
    // closed socket then cost the client a full 15 s read timeout. The device
    // serialises execute behind one lock, so extra connections add memory and
    // transfer overlap, never device throughput - 2 MB bodies x 16 is 32 MB.
    private final ThreadPoolExecutor clients = new ThreadPoolExecutor(
            12, 16, 0L, TimeUnit.MILLISECONDS,
            new java.util.concurrent.LinkedBlockingQueue<>(64),
            new ThreadPoolExecutor.AbortPolicy());
    private final java.util.concurrent.atomic.AtomicInteger activeClients = new java.util.concurrent.atomic.AtomicInteger();
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

            // Ask Shizuku to keep us out of doze / background-freeze before the world
            // load starts. Runs on its own thread; never blocks or fails startup.
            NpuKeepAlive.apply(this);

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
        // Ask the backend once at startup which ops it registered. This is a pure
        // query over an in-memory list - it builds no graph and queues no device
        // work, so it is safe here, unlike addProbe which must wait for the
        // listener. Logging it automatically means the answer is in the service
        // log without anyone having to send OPPROBE by hand.
        if (ok) {
            try {
                long t0 = System.nanoTime();
                String ops = NpuRuntime.opProbe();
                log("OPPROBE_AUTO elapsed_ms=" + ((System.nanoTime() - t0) / 1_000_000.0) + " " + ops);
            } catch (Throwable t) {
                log("OPPROBE_AUTO FAILED " + t);
            }
            // Same reasoning as above: a pure query, no graph, no device work.
            // This is the one line that decides whether the noise kernel is
            // written at all, so it belongs in the startup log by default.
            try {
                long t0 = System.nanoTime();
                String cap = NpuRuntime.perlinCap();
                log("PERLIN_CAP_AUTO elapsed_ms=" + ((System.nanoTime() - t0) / 1_000_000.0) + " " + cap);
            } catch (Throwable t) {
                log("PERLIN_CAP_AUTO FAILED " + t);
            }
        }
        // Spawned once, after the listener is up. Probing before that would delay
        // the socket MC is waiting on, and the probe builds graphs - it must never
        // sit between init and the first accept.
        boolean addProbeStarted = false;
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
                if (ok && !addProbeStarted) {
                    addProbeStarted = true;
                    startAddProbe();
                }
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
                        long acceptedNs = System.nanoTime();
                        workerBeatMs = System.currentTimeMillis();
                        log("IPC ACCEPT " + socket.getRemoteSocketAddress()
                                + " active=" + activeClients.get()
                                + " queued=" + clients.getQueue().size());
                        try {
                            clients.execute(() -> {
                                activeClients.incrementAndGet();
                                try {
                                    handle(socket, acceptedNs);
                                } finally {
                                    activeClients.decrementAndGet();
                                }
                            });
                        } catch (RejectedExecutionException rejected) {
                            // Say so before closing. A bare close leaves the client
                            // blocked in read until its timeout, which is how two
                            // segments of the 9x9 import each burned 15 s and looked
                            // like an NPU that had stopped working.
                            try {
                                java.io.OutputStream os = socket.getOutputStream();
                                os.write(("ERR OVERLOAD active=" + activeClients.get()
                                        + " queued=" + clients.getQueue().size() + "\n")
                                        .getBytes(java.nio.charset.StandardCharsets.UTF_8));
                                os.flush();
                            } catch (Throwable ignored) {}
                            try { socket.close(); } catch (Throwable ignored) {}
                            log("IPC REJECT overload active=" + activeClients.get()
                                    + " queued=" + clients.getQueue().size()
                                    + " remote=" + socket.getRemoteSocketAddress());
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
                long[] samples = new long[1000];
                for (int i = 0; i < samples.length; i++) {
                    long p0 = System.nanoTime();
                    t.getOutputStream().write("PING\n".getBytes(java.nio.charset.StandardCharsets.UTF_8));
                    t.getOutputStream().flush();
                    String pr = br.readLine();
                    if (!"PONG MCNPU/1".equals(pr)) throw new IOException("bad ping reply: " + pr);
                    samples[i] = (System.nanoTime() - p0) / 1000L;
                }
                java.util.Arrays.sort(samples);
                log("IPC PINGBENCH n=1000 min_us=" + samples[0]
                        + " p50_us=" + samples[499] + " p95_us=" + samples[949]
                        + " p99_us=" + samples[989] + " max_us=" + samples[999]);
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

    /**
     * True for commands that actually reach the device.
     *
     * PING, STATUS and CAPABILITIES are answered from memory. They are also the
     * commands clients poll with, and polling opens a connection roughly once a
     * second - so treating them as "real work" killed the startup probe before it
     * could measure its first candidate size. ADDPROBE is deliberately excluded
     * too: it is the probe, and aborting it from inside itself makes no sense.
     */
    private static boolean holdsDeviceLock(String cmd) {
        if (cmd.startsWith("BINADD ")) return true;
        if (cmd.startsWith("SUBMITBIN_MATMUL8 ")) return true;
        if (cmd.startsWith("PREWARM8 ")) return true;
        if (cmd.equals("SMOKE")) return true;
        if (cmd.startsWith("EXEC_")) return true;
        if (cmd.startsWith("ADD ")) return true;
        if (cmd.startsWith("MATMUL")) return true;
        return false;
    }

    private void handle(Socket socket, long acceptedNs) {
        try (Socket s = socket) {
            // 小包请求不要撞上 Nagle + delayed-ACK（实测 p99 往返 ~50ms，p50 仅 ~1.4ms）
            try { s.setTcpNoDelay(true); } catch (Throwable ignored) {}
            try { s.setSoTimeout(IDLE_TIMEOUT_MS); } catch (Throwable ignored) {}
            // BufferedInputStream is safe here: unlike BufferedReader it does not decode or
            // pre-consume the binary tensor payload. It also removes thousands of tiny read()
            // calls from the control header path. The client already uses the same buffer size.
            java.io.PushbackInputStream in = new java.io.PushbackInputStream(
                    new BufferedInputStream(s.getInputStream(), 256 * 1024), LINE_BLOCK);
            OutputStream out = new BufferedOutputStream(s.getOutputStream(), 256 * 1024);
            // Queue time is genuine server-side queueing: accept -> the moment this
            // connection starts being served. It is measured once per connection
            // because that is the only place queueing happens in this design - once a
            // handler owns the socket it is dedicated to it.
            //
            // It used to be computed once and then stamped onto every command on the
            // connection, so a session with 30 requests reported the same
            // service_queue_us 30 times. A per-request field that cannot vary is not a
            // measurement, and it made the figure look like a fixed hardware cost.
            final long connQueueUs = Math.max(0L, (System.nanoTime() - acceptedNs) / 1000L);
            boolean helloDone = false;
            while (true) {
                // readUs is how long this handler sat blocked waiting for the request's
                // bytes. It is the only part of the request's life that varies per
                // command, and when it is large the client is the one being slow.
                long readStartNs = System.nanoTime();
                String line;
                try {
                    line = readLineUtf8(in, MAX_LINE_BYTES);
                } catch (java.net.SocketTimeoutException idle) {
                    log("IPC idle close remote=" + s.getRemoteSocketAddress()
                            + " idle_ms=" + IDLE_TIMEOUT_MS);
                    break;
                }
                long readUs = (System.nanoTime() - readStartNs) / 1000L;
                if (line == null) break;
                String cmd = line.trim();
                // Real work outranks a diagnostic, but only real work does. This used
                // to run once per connection, and status polling opens a connection
                // about once a second - so the startup probe was killed by the first
                // PING and never measured a single candidate size.
                if (addProbeRunning && holdsDeviceLock(cmd)) NpuRuntime.abortProbe();
                if (!helloDone) {
                    if (cmd.equals("HELLO MCJAVA_NPU/1")) {
                        writeLineUtf8(out, "OK HELLO MCNPU/1");
                        helloDone = true;
                        log("IPC SESSION HELLO remote=" + s.getRemoteSocketAddress());
                        continue;
                    }
                    // Keep backwards compatibility with older clients, but only the
                    // first command is allowed to be a handshake. Legacy clients can
                    // continue directly because their protocol predates HELLO.
                    helloDone = true;
                }
                if (cmd.startsWith("BINADD ")) {
                    try {
                        handleBinAdd(in, out, cmd.substring(7), connQueueUs, readUs);
                    } catch (Throwable t) {
                        log("BIN ADD exception=" + t);
                        writeLineUtf8(out, "ERR BINADD_EXCEPTION " + t.getClass().getSimpleName());
                    }
                    continue;
                }
                if (cmd.startsWith("SUBMITBIN_MATMUL8 ")) {
                    try {
                        handleSubmitBinMatMul8(in, out, cmd.substring(18), connQueueUs, readUs);
                    } catch (Throwable t) {
                        log("BIN SUBMIT exception=" + t);
                        writeLineUtf8(out, "ERR BIN_SUBMIT_EXCEPTION " + t.getClass().getSimpleName());
                    }
                    continue;
                }
                String reply;
                if (cmd.equals("PING")) reply = "PONG MCNPU/1";
                else if (cmd.equals("STATUS")) reply = NpuRuntime.status();
                else if (cmd.startsWith("PREWARM8 ")) {
                    try {
                        String[] pp = cmd.substring(9).trim().split("[x*, ]+");
                        if (pp.length != 3) reply = "ERR PREWARM8_FORMAT use: PREWARM8 m k n";
                        else reply = NpuRuntime.prewarmMatMulInt8(
                                Integer.parseInt(pp[0]), Integer.parseInt(pp[1]), Integer.parseInt(pp[2]));
                    } catch (Throwable t) {
                        reply = "ERR PREWARM8_EXCEPTION " + t.getClass().getSimpleName();
                    }
                    log("PREWARM8 result=" + reply);
                } else if (cmd.equals("SMOKE")) {
                    long t = System.nanoTime();
                    // Pass the real reply through. It already starts with
                    // "OK HTP_GRAPH_EXECUTE" on success and carries the timings,
                    // so the client's existing prefix check still works - and on
                    // failure it now says why instead of a constant.
                    reply = NpuRuntime.smokeDetail();
                    log("EXEC SMOKE result=" + reply + " elapsed_ms=" + ((System.nanoTime() - t) / 1_000_000.0));
                } else if (cmd.equals("BONSAI2_CAPS")) {
                    // Kernel-level capability only. This deliberately does not claim PQ2 model support.
                    reply = "OK BONSAI2_NPU/1 mode=KERNEL_BASELINE backend=HTP_V73 matmul=int8 binary=SUBMITBIN_MATMUL8 pq2=NOT_INTEGRATED max_matrix_bytes=67108864 memory_ceiling_bytes=4294967296";
                } else if (cmd.equals("CAPABILITIES")) {
                    reply = "OK MCNPU/1 backend=HTP_V73 ops=ADD,MATMUL,MATMUL16,MATMUL8,SUBMIT8,SUBMITBIN8,PREWARM8,ADDPROBE,BONSAI2_CAPS max_elements=" + NpuRuntime.maxAddElements();
                } else if (cmd.equals("PERLIN_CAP")) {
                    reply = NpuRuntime.perlinCap();
                    log("EXEC PERLIN_CAP " + reply);
                } else if (cmd.startsWith("PERLIN ")) {
                    // PERLIN <n> - build, run, and verify against the CPU reference.
                    int n = 4096;
                    try { n = Integer.parseInt(cmd.substring(7).trim()); } catch (Throwable ignored) { }
                    long t = System.nanoTime();
                    reply = NpuRuntime.perlinBench(n);
                    log("EXEC PERLIN elapsed_ms=" + ((System.nanoTime() - t) / 1_000_000.0)
                            + " " + reply);
                } else if (cmd.equals("PERLIN")) {
                    long t = System.nanoTime();
                    reply = NpuRuntime.perlinBench(4096);
                    log("EXEC PERLIN elapsed_ms=" + ((System.nanoTime() - t) / 1_000_000.0)
                            + " " + reply);
                } else if (cmd.equals("OPPROBE")) {
                    // Asks the backend what it supports. No graph is built and no
                    // device work is queued, so this must not abort an in-flight
                    // probe - it is a query, not work.
                    reply = NpuRuntime.opProbe();
                    log("EXEC OPPROBE " + reply.replace('\n', ' '));
                } else if (cmd.equals("ADDPROBE")) {
                    long t = System.nanoTime();
                    reply = NpuRuntime.addProbe();
                    log("EXEC ADDPROBE elapsed_ms=" + ((System.nanoTime() - t) / 1_000_000.0)
                            + "\n" + reply);
                } else if (cmd.startsWith("EXEC_ADD ")) {
                    reply = handleAdd(cmd.substring(9));
                } else if (cmd.startsWith("ADD ")) {
                    reply = handleAdd(cmd.substring(4));
                } else if (cmd.startsWith("EXEC_MATMUL ")) {
                    reply = handleMatMul(cmd.substring(12));
                } else if (cmd.startsWith("MATMUL ")) {
                    reply = handleMatMul(cmd.substring(7));
                } else if (cmd.startsWith("EXEC_MATMUL16 ")) {
                    reply = handleMatMul16(cmd.substring(14));
                } else if (cmd.startsWith("MATMUL16 ")) {
                    reply = handleMatMul16(cmd.substring(9));
                } else if (cmd.startsWith("EXEC_MATMUL8 ")) {
                    reply = handleMatMul8(cmd.substring(13));
                } else if (cmd.startsWith("MATMUL8 ")) {
                    reply = handleMatMul8(cmd.substring(8));
                } else if (cmd.startsWith("EXEC_XFORM ") || cmd.startsWith("XFORM ")) {
                    try {
                        String[] pp = cmd.substring(cmd.startsWith("EXEC_XFORM ") ? 11 : 6).trim().split("[, ]+");
                        if (pp.length != 2) reply = "ERR XFORM_FORMAT use: EXEC_XFORM op n";
                        else reply = NpuRuntime.xform(Integer.parseInt(pp[0]), Integer.parseInt(pp[1]));
                    } catch (Throwable t) { reply = "ERR XFORM_EXCEPTION " + t.getClass().getSimpleName(); }
                    log("EXEC XFORM result=" + reply);
                } else if (cmd.startsWith("SUBMIT_MATMUL8 ")) {
                    reply = handleSubmitMatMul8(cmd.substring(15));
                } else if (cmd.equals("QUIT")) {
                    writeLineUtf8(out, "BYE");
                    break;
                } else {
                    reply = "ERR UNKNOWN_COMMAND";
                }
                writeLineUtf8(out, reply);
            }
        } catch (Throwable t) {
            log("IPC client closed: " + t);
        }
    }

    /**
     * Reads one UTF-8 line in blocks so binary payloads are never prefetched.
     *
     * The previous byte-at-a-time loop existed to avoid consuming the raw tensor
     * bytes that follow a SUBMITBIN header. It also meant one read() per byte, so
     * a 360 KB ADD command cost ~360K calls. Blocks plus pushback keep the same
     * guarantee: everything after the newline goes back to the stream untouched.
     *
     * The 256 KB ceiling was the second problem. A n=16384 ADD command is about
     * 360 KB of decimal text, so it was rejected as "line too long" even though
     * 16384 is a size the HTP accepts - the imported test then reported INVALID
     * for a shape that was never actually sent.
     */
    private static String readLineUtf8(java.io.PushbackInputStream in, int maxBytes) throws IOException {
        java.io.ByteArrayOutputStream buf = new java.io.ByteArrayOutputStream(256);
        byte[] block = new byte[LINE_BLOCK];
        while (true) {
            int n = in.read(block);
            if (n < 0) break;
            int nl = -1;
            for (int i = 0; i < n; i++) if (block[i] == '\n') { nl = i; break; }
            if (nl >= 0) {
                // Push back everything after the newline, including any binary payload.
                if (nl + 1 < n) in.unread(block, nl + 1, n - (nl + 1));
                if (nl > 0 && block[nl - 1] != '\r') buf.write(block, 0, nl);
                else if (nl > 0) buf.write(block, 0, nl - 1);
                return new String(buf.toByteArray(), java.nio.charset.StandardCharsets.UTF_8);
            }
            buf.write(block, 0, n);
            if (buf.size() > maxBytes) throw new IOException("line too long >" + maxBytes + " bytes");
        }
        if (buf.size() == 0) return null;
        return new String(buf.toByteArray(), java.nio.charset.StandardCharsets.UTF_8);
    }

    private static void writeLineUtf8(OutputStream out, String s) throws IOException {
        out.write((s + "\n").getBytes(java.nio.charset.StandardCharsets.UTF_8));
        out.flush();
    }

    private static void readFully(InputStream in, byte[] dst, int len) throws IOException {
        int off = 0;
        while (off < len) {
            int n = in.read(dst, off, len - off);
            if (n < 0) throw new IOException("eof after " + off + " of " + len + " bytes");
            off += n;
        }
    }

    /**
     * SUBMITBIN_MATMUL8 m k n alen blen\n<A raw bytes><B raw bytes>
     * True binary data plane: no base64, no string parsing of tensor data.
     * Reply: one text header line, then the raw int8 result bytes.
     */
    /**
     * connQueueUs is accept -> dispatch, measured once per connection in handle().
     * readUs is how long we blocked waiting for this command's bytes. Both belong to
     * the caller's frame, so they are passed in rather than re-derived here.
     */
    private void handleSubmitBinMatMul8(InputStream in, OutputStream out, String payload,
                                        long connQueueUs, long readUs) throws IOException {
        String[] p = payload.trim().split(" ");
        if (p.length != 5) { writeLineUtf8(out, "ERR BIN_FORMAT use: SUBMITBIN_MATMUL8 m k n alen blen"); return; }
        int m = Integer.parseInt(p[0]), k = Integer.parseInt(p[1]), n = Integer.parseInt(p[2]);
        int alen = Integer.parseInt(p[3]), blen = Integer.parseInt(p[4]);
        if ((long) alen != (long) m * k || (long) blen != (long) k * n) {
            writeLineUtf8(out, "ERR BIN_SIZE expect alen=" + (m * k) + " blen=" + (k * n));
            return;
        }
        byte[] A = new byte[alen], B = new byte[blen];
        readFully(in, A, alen);
        readFully(in, B, blen);
        long t0 = System.nanoTime();
        byte[] res = NpuRuntime.matMulInt8Buf(A, B, m, k, n);
        long us = (System.nanoTime() - t0) / 1000;
        if (res == null || res.length < 5) {
            writeLineUtf8(out, "ERR BIN_SUBMIT_FAILED " + NpuRuntime.getLastNativeError());
            return;
        }
        java.nio.ByteBuffer bb = java.nio.ByteBuffer.wrap(res, 0, 4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        float scaleC = bb.getFloat();
        int cbytes = res.length - 4;
        log("SUBMITBIN_MATMUL8 m=" + m + " k=" + k + " n=" + n
                + " scaleC=" + scaleC + " cbytes=" + cbytes
                + " conn_queue_us=" + connQueueUs + " read_us=" + readUs
                + " npu_service_us=" + us);
        // One buffered flush for header + tensor. The old path flushed the header first,
        // forcing a second transport boundary before the result bytes.
        out.write(("OK BIN_SUBMIT m=" + m + " k=" + k + " n=" + n + " scaleC=" + scaleC
                + " cbytes=" + cbytes + " us=" + us + " conn_queue_us=" + connQueueUs
                + " read_us=" + readUs + " binary=1\n")
                .getBytes(java.nio.charset.StandardCharsets.UTF_8));
        out.write(res, 4, cbytes);
        out.flush();
    }

    /** EXEC_MATMUL m k n  -> deterministic fp32 matmul on HTP + CPU baseline. */
    private String handleMatMul(String payload) {
        try {
            String[] p = payload.trim().split("[x*, ]+");
            if (p.length != 3) return "ERR MATMUL_FORMAT use: EXEC_MATMUL m k n";
            int m = Integer.parseInt(p[0]), k = Integer.parseInt(p[1]), n = Integer.parseInt(p[2]);
            String result = NpuRuntime.matMul(m, k, n);
            log("EXEC MATMUL result=" + result);
            return result;
        } catch (Throwable t) {
            log("EXEC MATMUL exception=" + t);
            return "ERR MATMUL_EXCEPTION " + t.getClass().getSimpleName();
        }
    }

    /** EXEC_MATMUL16 m k n -> same matmul with fp16 tensors. */
    private String handleMatMul16(String payload) {
        try {
            String[] p = payload.trim().split("[x*, ]+");
            if (p.length != 3) return "ERR MATMUL16_FORMAT use: EXEC_MATMUL16 m k n";
            int m = Integer.parseInt(p[0]), k = Integer.parseInt(p[1]), n = Integer.parseInt(p[2]);
            String result = NpuRuntime.matMulFp16(m, k, n);
            log("EXEC MATMUL16 result=" + result);
            return result;
        } catch (Throwable t) {
            log("EXEC MATMUL16 exception=" + t);
            return "ERR MATMUL16_EXCEPTION " + t.getClass().getSimpleName();
        }
    }

    /** EXEC_MATMUL8 m k n -> int8 quantized matmul (HTP native datatype). */
    private String handleMatMul8(String payload) {
        try {
            String[] p = payload.trim().split("[x*, ]+");
            if (p.length != 3) return "ERR MATMUL8_FORMAT use: EXEC_MATMUL8 m k n";
            int m = Integer.parseInt(p[0]), k = Integer.parseInt(p[1]), n = Integer.parseInt(p[2]);
            String result = NpuRuntime.matMulInt8(m, k, n);
            log("EXEC MATMUL8 result=" + result);
            return result;
        } catch (Throwable t) {
            log("EXEC MATMUL8 exception=" + t);
            return "ERR MATMUL8_EXCEPTION " + t.getClass().getSimpleName();
        }
    }

    /**
     * SUBMIT_MATMUL8 m k n <base64 A> <base64 B>
     * Real data path: caller-side int8 tensors (normalized to [-1,1]) go straight
     * into the cached HTP graph; the reply carries scaleC plus the raw int8 result.
     */
    private String handleSubmitMatMul8(String payload) {
        try {
            String[] p = payload.trim().split(" ");
            if (p.length != 5) return "ERR SUBMIT_FORMAT use: SUBMIT_MATMUL8 m k n <b64A> <b64B>";
            int m = Integer.parseInt(p[0]), k = Integer.parseInt(p[1]), n = Integer.parseInt(p[2]);
            byte[] A = java.util.Base64.getDecoder().decode(p[3]);
            byte[] B = java.util.Base64.getDecoder().decode(p[4]);
            long t0 = System.nanoTime();
            byte[] out = NpuRuntime.matMulInt8Buf(A, B, m, k, n);
            long us = (System.nanoTime() - t0) / 1000;
            if (out == null || out.length < 5) return "ERR SUBMIT_FAILED (see logcat for native reason)";
            java.nio.ByteBuffer bb = java.nio.ByteBuffer.wrap(out, 0, 4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            float scaleC = bb.getFloat();
            String b64 = java.util.Base64.getEncoder().encodeToString(java.util.Arrays.copyOfRange(out, 4, out.length));
            log("SUBMIT_MATMUL8 m=" + m + " k=" + k + " n=" + n + " scaleC=" + scaleC + " us=" + us);
            return "OK SUBMIT m=" + m + " k=" + k + " n=" + n + " scaleC=" + scaleC
                    + " cbytes=" + (out.length - 4) + " roundtrip_us=" + us + " c=" + b64;
        } catch (Throwable t) {
            log("SUBMIT_MATMUL8 exception=" + t);
            return "ERR SUBMIT_EXCEPTION " + t.getClass().getSimpleName();
        }
    }

    /**
     * BINADD n case_count\n<case_count * 2 * n little-endian float32>
     *
     * The text ADD path sends each value as decimal text: n=16384 costs about
     * 360 KB of ASCII plus two Float.toString passes and two String.split calls
     * per case. This path sends the same values as 128 KB of raw float32 that
     * goes straight into the JNI call, which is what the multi-chunk batches
     * actually need.
     *
     * The reply is still a text header because nativeAdd returns a string; making
     * the result binary too needs a new native entry that fills a float[].
     */
    /**
     * Measures the real ADD size ceiling on this device, off the IPC path.
     *
     * The 16384 in the native ladder was only the largest size we had verified -
     * it lives in our own .so, not in the HTP - and every layer above it copied
     * that number. Until it is measured on the device, a phone that accepts more
     * will silently never use it. The result lands in the log as ADD_PROBE.
     */
    /** True while the startup probe still owns candidate graph builds. */
    private volatile boolean addProbeRunning = false;

    private void startAddProbe() {
        Thread t = new Thread(() -> {
            try {
                // Give MC's first requests room. The probe builds a graph per
                // candidate size and must not compete with real work at startup.
                Thread.sleep(3000);
                long t0 = System.nanoTime();
                addProbeRunning = true;
                String r = NpuRuntime.addProbe();
                log("ADD_PROBE elapsed_ms=" + ((System.nanoTime() - t0) / 1_000_000.0) + "\n" + r);
                log("ADD max_elements=" + NpuRuntime.maxAddElements());

                // Capability alone never proved the kernel works - it only said
                // the ops exist. This actually builds the noise graph, runs it and
                // diffs every point against a CPU reference written the textbook
                // way, so the answer shows up in the log without anyone having to
                // know a command exists. Runs here, not before the listener: it
                // builds graphs, and MC's first connect must not wait on that.
                try {
                    long tp = System.nanoTime();
                    String p = NpuRuntime.perlinBench(4096);
                    log("PERLIN_AUTO elapsed_ms=" + ((System.nanoTime() - tp) / 1_000_000.0) + " " + p);
                } catch (Throwable pe) {
                    log("PERLIN_AUTO FAILED " + pe);
                }
            } catch (Throwable e) {
                log("ADD_PROBE exception=" + e);
            } finally {
                addProbeRunning = false;
            }
        }, "mcnpu-add-probe");
        t.setPriority(Thread.MIN_PRIORITY);
        t.setDaemon(true);
        t.start();
    }

    private void handleBinAdd(java.io.PushbackInputStream in, OutputStream out, String args,
                              long connQueueUs, long readUs) throws IOException {
        long t0 = System.nanoTime();
        String[] pp = args.trim().split("[ ]+");
        if (pp.length != 2) { writeLineUtf8(out, "ERR BINADD_FORMAT use: BINADD n case_count"); return; }
        int n, cases;
        try {
            n = Integer.parseInt(pp[0]);
            cases = Integer.parseInt(pp[1]);
        } catch (NumberFormatException e) {
            writeLineUtf8(out, "ERR BINADD_FORMAT non-numeric"); return;
        }
        int maxAdd = NpuRuntime.maxAddElements();
        if (n <= 0 || n > maxAdd || cases <= 0 || cases > 4096) {
            writeLineUtf8(out, "ERR BINADD_RANGE n=" + n + " cases=" + cases); return;
        }
        long bodyBytes = 4L * 2L * n * cases;
        if (bodyBytes > MAX_BIN_BODY) {
            writeLineUtf8(out, "ERR BINADD_TOO_LARGE body=" + bodyBytes + " cap=" + MAX_BIN_BODY); return;
        }
        log("BINADD recv n=" + n + " cases=" + cases + " body_bytes=" + bodyBytes);
        byte[] body = new byte[(int) bodyBytes];
        readFully(in, body, body.length);
        log("BINADD body read ok n=" + n + " cases=" + cases);
        java.nio.ByteBuffer bb = java.nio.ByteBuffer.wrap(body).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        float[] a = new float[n], b = new float[n];
        int ok = 0, cached = 0;
        long npuUs = 0;
        StringBuilder first = new StringBuilder();
        byte[] results = new byte[4 * n * cases];
        java.nio.ByteBuffer rb = java.nio.ByteBuffer.wrap(results).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        // Named outBuf, not out: this method's reply stream is already called out,
        // and shadowing it turns every write below into a type error that reads as
        // if the result buffer were the problem.
        // Filled with the same -999 sentinel the native side uses, so a way the
        // HTP never wrote reads back as -999 instead of as a plausible 0. That is
        // the whole difference between "the device returned zeros" and "the device
        // never ran", and a zero-filled buffer cannot tell them apart.
        float[] outBuf = new float[n];
        for (int c = 0; c < cases; c++) {
            java.util.Arrays.fill(outBuf, -999f);
            for (int i = 0; i < n; i++) a[i] = bb.getFloat();
            for (int i = 0; i < n; i++) b[i] = bb.getFloat();
            long c0 = System.nanoTime();
            // verify=false: the per-element CPU compare would cost more than the
            // graph execute it is meant to be measuring.
            String r = NpuRuntime.addInto(a, b, outBuf, false);
            long us = (System.nanoTime() - c0) / 1000L;
            npuUs += us;
            if (r != null && r.startsWith("OK")) ok++;
            if (r != null && r.contains("graph_cached=true")) cached++;
            if (c == 0) {
                first = new StringBuilder(r == null ? "null" : r);
                // The first case carries the graph build. Saying so separates
                // "the device is slow to build" from "the device never answered",
                // which is the only question that matters on a stalled run.
                log("BINADD case0 n=" + n + " us=" + us + " r=" + first);
            } else if (c % 8 == 0) {
                log("BINADD progress n=" + n + " case=" + c + "/" + cases + " us=" + us);
            }
            for (int i = 0; i < n; i++) rb.putFloat(outBuf[i]);
        }
        long totalUs = (System.nanoTime() - t0) / 1000L;
        // The prefix has to follow ok, not the fact that the call was understood.
        // A header reading "OK BINADD ... ok=0/16" made the client copy a buffer
        // the device had never written, and the imported test then reported a
        // perfect-looking max_abs of max|a+b| with every element wrong. This is
        // the same black hole as smoke()'s boolean: the reason is dropped at the
        // boundary and the failure arrives dressed as success.
        String reason = first.length() == 0 ? "none" : first.toString();
        if (reason.length() > 300) reason = reason.substring(0, 300) + "...";
        writeLineUtf8(out, (ok == cases ? "OK BINADD" : "ERR BINADD_PARTIAL")
                + " n=" + n + " cases=" + cases + " ok=" + ok + "/" + cases
                + " cached=" + cached + " body_bytes=" + bodyBytes
                + " npu_us=" + npuUs + " conn_queue_us=" + connQueueUs
                + " read_us=" + readUs + " total_us=" + totalUs
                + " out_bytes=" + results.length + " case0=" + reason);
        // Results follow the header as raw little-endian float32. The header stays
        // text so a failure is still readable without knowing the binary layout.
        out.write(results);
        out.flush();
        log("BINADD n=" + n + " cases=" + cases + " ok=" + ok + "/" + cases
                + " body_bytes=" + bodyBytes + " npu_us=" + npuUs + " total_us=" + totalUs
                + (ok == cases ? "" : " case0=" + reason));
    }

    private String handleAdd(String payload) {
        long t0 = System.nanoTime();
        try {
            String[] parts = payload.split("\\|", -1);
            if (parts.length != 2) return "ERR ADD_FORMAT";
            String[] as = parts[0].split(",", -1);
            String[] bs = parts[1].split(",", -1);
            if (as.length == 0 || as.length != bs.length || as.length > NpuRuntime.maxAddElements()) return "ERR ADD_SIZE";
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