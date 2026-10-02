package bslsjdk.mcnpu;

import android.app.*;
import android.content.Intent;
import android.os.IBinder;
import android.os.Build;
import android.net.LocalServerSocket;
import android.net.LocalSocket;
import android.net.LocalSocketAddress;
import java.io.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class NpuService extends Service {
    public static final String SOCKET_NAME = "mcnpu_v1";
    private static final int NOTIFICATION_ID = 1001;
    private static final String CHANNEL = "mcnpu";
    private final ExecutorService clients = Executors.newCachedThreadPool();
    private volatile boolean running;
    private LocalServerSocket server;

    @Override public void onCreate() {
        super.onCreate();
        createChannel();
        startForeground(NOTIFICATION_ID, notification("MC NPU: starting"));
        running = true;
        clients.execute(this::serverLoop);
    }

    @Override public int onStartCommand(Intent intent, int flags, int startId) { return START_STICKY; }

    private void serverLoop() {
        if (!NpuRuntime.init()) updateNotification("MC NPU: HTP init failed");
        else updateNotification("MC NPU: HTP V73 ready");
        try {
            server = new LocalServerSocket(new LocalSocketAddress(SOCKET_NAME, LocalSocketAddress.Namespace.ABSTRACT));
            while (running) {
                LocalSocket socket = server.accept();
                clients.execute(() -> handle(socket));
            }
        } catch (Throwable t) {
            if (running) updateNotification("MC NPU: IPC stopped " + t.getClass().getSimpleName());
        }
    }

    private void handle(LocalSocket socket) {
        try (LocalSocket s = socket;
             BufferedReader in = new BufferedReader(new InputStreamReader(s.getInputStream()));
             BufferedWriter out = new BufferedWriter(new OutputStreamWriter(s.getOutputStream()))) {
            String line;
            while ((line = in.readLine()) != null) {
                String cmd = line.trim();
                if (cmd.equals("PING")) reply(out, "PONG MCNPU/1");
                else if (cmd.equals("STATUS")) reply(out, NpuRuntime.status());
                else if (cmd.equals("SMOKE")) reply(out, NpuRuntime.smoke() ? "OK HTP_GRAPH_EXECUTE" : "ERR HTP_GRAPH_EXECUTE");
                else if (cmd.startsWith("ADD ")) reply(out, handleAdd(cmd.substring(4)));
                else if (cmd.equals("QUIT")) { reply(out, "BYE"); break; }
                else reply(out, "ERR UNKNOWN_COMMAND");
            }
        } catch (Throwable ignored) {}
    }

    private String handleAdd(String payload) {
        try {
            String[] parts = payload.split("\\|");
            if (parts.length != 2) return "ERR ADD_FORMAT";
            String[] as = parts[0].split(",");
            String[] bs = parts[1].split(",");
            if (as.length == 0 || as.length != bs.length || as.length > 1024) return "ERR ADD_SIZE";
            float[] a = new float[as.length], b = new float[bs.length];
            for (int i=0;i<as.length;i++) { a[i]=Float.parseFloat(as[i]); b[i]=Float.parseFloat(bs[i]); }
            return NpuRuntime.add(a,b);
        } catch (Throwable t) { return "ERR ADD_EXCEPTION " + t.getClass().getSimpleName(); }
    }

    private static void reply(BufferedWriter out, String s) throws IOException {
        out.write(s); out.write("\n"); out.flush();
    }

    private void createChannel() {
        if (Build.VERSION.SDK_INT >= 26) {
            NotificationManager nm = getSystemService(NotificationManager.class);
            nm.createNotificationChannel(new NotificationChannel(CHANNEL, "MC NPU", NotificationManager.IMPORTANCE_LOW));
        }
    }

    private Notification notification(String text) {
        Notification.Builder b = Build.VERSION.SDK_INT >= 26 ? new Notification.Builder(this, CHANNEL) : new Notification.Builder(this);
        return b.setContentTitle("MC NPU").setContentText(text)
                .setSmallIcon(android.R.drawable.stat_sys_download_done).setOngoing(true).build();
    }
    private void updateNotification(String text) {
        getSystemService(NotificationManager.class).notify(NOTIFICATION_ID, notification(text));
    }

    @Override public void onDestroy() {
        running=false;
        try { if(server!=null) server.close(); } catch(Throwable ignored){}
        clients.shutdownNow();
        NpuRuntime.shutdown();
        super.onDestroy();
    }
    @Override public IBinder onBind(Intent intent) { return null; }
}
