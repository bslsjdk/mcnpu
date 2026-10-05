package bslsjdk.mcnpu;

import android.content.Context;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Method;
import java.util.ArrayList;

/**
 * Keeps the MCNPU process alive long enough for a Minecraft world load.
 *
 * Why this exists:
 *   Creating a world allocates hundreds of megabytes. Android's LMK reaps background
 *   processes under that pressure, and MCNPU is a background service of an app the user
 *   is not looking at. A captured run shows the service killed and rebuilt by
 *   START_STICKY 2m40s later - long after the world load had stopped waiting for it.
 *
 * What this does, in descending order of value:
 *   1. Adds the package to the deviceidle (doze) whitelist - the most effective measure
 *      available without root, because doze and app-standby are what actually stop
 *      background work on modern Android.
 *   2. Allows the AppOps ColorOS/realme checks before freezing a background app:
 *      RUN_IN_BACKGROUND / RUN_ANY_IN_BACKGROUND / WAKE_LOCK.
 *   3. Pins the standby bucket to ACTIVE so standby quotas do not throttle it.
 *
 * All of that needs a shell-level identity, which is what Shizuku provides.
 *
 * Shizuku is invoked reflectively on purpose: the exact process API varies between
 * Shizuku releases, and a compile-time dependency on it would turn an optional
 * enhancement into a hard build break. If the API is absent this class logs what it
 * would have done and stops. It never throws into the service startup path.
 *
 * Every step is read back and verified rather than assumed.
 */
public final class NpuKeepAlive {
    private static final String TAG = "MCNPU";
    private static final String PKG = "bslsjdk.mcnpu";
    /** ROMs reset the standby bucket and appops on their own schedule; re-apply. */
    private static final long REAPPLY_MS = 10 * 60 * 1000L;
    /** Poll interval while waiting for the Shizuku permission to be granted. */
    private static final long WAIT_SHIZUKU_MS = 10 * 1000L;

    private static volatile String lastReport = "未执行";
    private static volatile boolean applied = false;
    private static volatile boolean running = false;
    /** First failure reason from the most recent pass, surfaced in the report. */
    private static volatile String lastError = null;

    private NpuKeepAlive() {}

    /** Fire-and-forget; safe to call from the service startup path. */
    public static synchronized void apply(final Context ctx) {
        if (running) {
            android.util.Log.i(TAG, "KEEPALIVE already running, skip");
            return;
        }
        running = true;
        Thread t = new Thread(new Runnable() {
            @Override public void run() {
                try {
                    runLoop();
                } catch (Throwable e) {
                    android.util.Log.e(TAG, "KEEPALIVE crashed", e);
                    lastReport = "崩溃: " + e;
                } finally {
                    running = false;
                }
            }
        }, "mcnpu-keepalive");
        t.setDaemon(true);
        t.start();
    }

    /** Re-check now instead of waiting for the next poll. */
    public static void kick(final Context ctx) { apply(ctx); }

    private static void runLoop() {
        // Wait for Shizuku rather than giving up. This thread starts when the
        // service starts, which is normally BEFORE the user has accepted the
        // Shizuku prompt - and returning here was permanent, so the whitelist
        // was never applied even though the same diagnostic later reported
        // "已授权". Observed as: KEEPALIVE "未授权" next to AUTH_RESULT "已授权".
        while (!ShizukuHelper.available() || !ShizukuHelper.granted()) {
            lastReport = ShizukuHelper.available()
                    ? "等待 Shizuku 授权（暂时无法加入省电白名单）"
                    : "等待 Shizuku 启动（暂时无法加入省电白名单）";
            android.util.Log.i(TAG, "KEEPALIVE wait: " + lastReport);
            if (!sleepQuietly(WAIT_SHIZUKU_MS)) return;
        }
        while (true) {
            ArrayList<String> results = new ArrayList<String>();
            lastError = null;
            String uid = shell("id", "-u");
            results.add("uid=" + (uid == null ? "?" : uid.trim()));

            // 1. Doze whitelist - the one that actually matters.
            shell("cmd", "deviceidle", "whitelist", "+" + PKG);
            boolean whitelisted = verifyWhitelist();
            results.add("doze_whitelist=" + (whitelisted ? "YES" : "no"));

            // 2. AppOps ColorOS checks before freezing a background app.
            String[] ops = {"RUN_IN_BACKGROUND", "RUN_ANY_IN_BACKGROUND", "WAKE_LOCK"};
            for (int i = 0; i < ops.length; i++) {
                String r = shell("cmd", "appops", "set", PKG, ops[i], "allow");
                results.add(ops[i] + "=" + (r == null ? "?" : (r.trim().length() == 0 ? "ok" : r.trim())));
            }

            // 3. Standby bucket: ACTIVE keeps it out of standby quota throttling.
            shell("am", "set-standby-bucket", PKG, "active");

            // Every command is read back rather than assumed. If the shell identity is
            // broken they all return null, and reporting "已保活" over a list of "?" would
            // claim success for a pass that applied nothing at all. The header must
            // follow the one result that matters, not the fact that we tried.
            applied = whitelisted;
            int unknown = 0;
            for (int i = 0; i < results.size(); i++) {
                if (results.get(i).endsWith("=?")) unknown++;
            }
            String head;
            if (unknown == results.size()) {
                head = "保活失败(所有命令未执行)";
            } else if (!whitelisted) {
                head = "部分生效(省电白名单未加入)";
            } else {
                head = "已保活";
            }
            lastReport = head + " · " + join(results)
                    + (lastError == null ? "" : " err=" + lastError);
            android.util.Log.i(TAG, "KEEPALIVE " + lastReport);

            if (!sleepQuietly(REAPPLY_MS)) return;
        }
    }

    /** @return false if interrupted. */
    private static boolean sleepQuietly(long ms) {
        try { Thread.sleep(ms); return true; }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); return false; }
    }

    private static String join(ArrayList<String> parts) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < parts.size(); i++) {
            if (i > 0) sb.append(' ');
            sb.append(parts.get(i));
        }
        return sb.toString();
    }

    private static boolean verifyWhitelist() {
        String out = shell("cmd", "deviceidle", "whitelist");
        return out != null && out.contains(PKG);
    }

    private static String shell(String... cmd) {
        try {
            Class<?> shizuku = Class.forName("rikka.shizuku.Shizuku");
            Method m = shizuku.getMethod("newProcess", String[].class, String[].class, String.class);
            Object proc = m.invoke(null, cmd, null, null);
            String out = drain((InputStream) proc.getClass().getMethod("getInputStream").invoke(proc))
                    + drain((InputStream) proc.getClass().getMethod("getErrorStream").invoke(proc));
            int rc = ((Integer) proc.getClass().getMethod("waitFor").invoke(proc)).intValue();
            android.util.Log.i(TAG, "KEEPALIVE rc=" + rc + " out=" + out.trim());
            return out;
        } catch (Throwable t) {
            String why = t.getClass().getSimpleName()
                    + (t.getMessage() == null ? "" : ": " + t.getMessage());
            if (lastError == null) lastError = why;
            android.util.Log.w(TAG, "KEEPALIVE cmd failed: " + t);
            return null;
        }
    }

    private static String drain(InputStream in) {
        if (in == null) return "";
        StringBuilder sb = new StringBuilder();
        BufferedReader r = null;
        try {
            r = new BufferedReader(new InputStreamReader(in));
            String line;
            while ((line = r.readLine()) != null) sb.append(line).append('\n');
        } catch (Throwable ignored) {
        } finally {
            if (r != null) try { r.close(); } catch (Throwable ignored) {}
        }
        return sb.toString();
    }

    public static boolean isApplied() { return applied; }
    public static String lastReport() { return lastReport; }
}
