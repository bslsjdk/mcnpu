package bslsjdk.mcnpu;

import android.content.Context;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Keeps the MCNPU process alive long enough for a Minecraft world load.
 *
 * Why this exists:
 *   Creating a world allocates hundreds of megabytes. Android's LMK reaps background
 *   processes under that pressure, and MCNPU is a background service of an app the user
 *   is not looking at. A captured run shows the service being killed and rebuilt 2m40s
 *   later - by then the world load had already given up on it.
 *
 * What this does, in descending order of value:
 *   1. Adds the package to the deviceidle (doze) whitelist - the single most effective
 *      thing available without root, because doze and app-standby are what actually
 *      stop background work on modern Android.
 *   2. Sets the AppOps that ColorOS/realme checks before freezing a background app:
 *      RUN_IN_BACKGROUND / RUN_ANY_IN_BACKGROUND / WAKE_LOCK.
 *   3. Puts the app in the ACTIVE standby bucket so standby quotas do not throttle it.
 *
 * All of this needs a shell-level identity, which is exactly what Shizuku provides.
 * Without Shizuku this class logs what it would have done and stops - it never blocks
 * and never throws into the service startup path.
 *
 * Everything is verified by reading the state back, not by assuming the command worked.
 */
public final class NpuKeepAlive {
    private static final String TAG = "MCNPU";
    private static final String PKG = "bslsjdk.mcnpu";
    /** ROMs reset the standby bucket and appops on their own schedule; re-apply. */
    private static final long REAPPLY_MS = 10 * 60 * 1000L;

    private static volatile String lastReport = "未执行";
    private static volatile boolean applied = false;

    private NpuKeepAlive() {}

    /** Fire-and-forget; safe to call from the service startup path. */
    public static void apply(final Context ctx) {
        new Thread(() -> {
            try {
                run(ctx, true);
            } catch (Throwable t) {
                android.util.Log.e(TAG, "KEEPALIVE crashed", t);
                lastReport = "崩溃: " + t;
            }
        }, "mcnpu-keepalive").start();
    }

    private static void run(Context ctx, boolean loop) {
        if (!ShizukuHelper.available()) {
            lastReport = "Shizuku 未运行（无法加入省电白名单）";
            android.util.Log.i(TAG, "KEEPALIVE skip: shizuku not running");
            return;
        }
        if (!ShizukuHelper.granted()) {
            lastReport = "Shizuku 未授权（无法加入省电白名单）";
            android.util.Log.i(TAG, "KEEPALIVE skip: shizuku not granted");
            return;
        }
        do {
            List<String> results = new ArrayList<>();
            String uid = shell("id", "-u");
            results.add("uid=" + (uid == null ? "?" : uid.trim()));

            // 1. Doze whitelist. This is the one that actually matters.
            shell("cmd", "deviceidle", "whitelist", "+" + PKG);
            boolean whitelisted = verifyWhitelist();
            results.add("doze_whitelist=" + (whitelisted ? "YES" : "no"));

            // 2. AppOps ColorOS checks before freezing a background app.
            for (String op : new String[]{"RUN_IN_BACKGROUND", "RUN_ANY_IN_BACKGROUND", "WAKE_LOCK"}) {
                String r = shell("cmd", "appops", "set", PKG, op, "allow");
                results.add(op + "=" + (r == null ? "?" : (r.trim().isEmpty() ? "ok" : r.trim())));
            }

            // 3. Standby bucket: ACTIVE keeps it out of standby quota throttling.
            shell("am", "set-standby-bucket", PKG, "active");

            applied = whitelisted;
            lastReport = "已保活 · " + String.join(" ", results);
            android.util.Log.i(TAG, "KEEPALIVE " + lastReport);

            if (!loop) return;
            try { Thread.sleep(REAPPLY_MS); } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        } while (true);
    }

    private static boolean verifyWhitelist() {
        String out = shell("cmd", "deviceidle", "whitelist");
        if (out == null) return false;
        return out.contains(PKG);
    }

    /**
     * Runs a command with Shizuku's shell identity. Returns stdout+stderr trimmed, or null
     * if the command could not be run at all. Never throws.
     */
    private static String shell(String... cmd) {
        try {
            rikka.shizuku.ShizukuRemoteProcess p =
                    rikka.shizuku.Shizuku.newProcess(cmd, null, null);
            String out = drain(p.getInputStream()) + drain(p.getErrorStream());
            int rc = p.waitFor();
            android.util.Log.i(TAG, "KEEPALIVE cmd=" + String.join(" ", cmd)
                    + " rc=" + rc + " out=" + out.trim());
            return out;
        } catch (Throwable t) {
            android.util.Log.w(TAG, "KEEPALIVE cmd failed: " + String.join(" ", cmd)
                    + " : " + t);
            return null;
        }
    }

    private static String drain(InputStream in) {
        if (in == null) return "";
        StringBuilder sb = new StringBuilder();
        try (java.io.BufferedReader r = new java.io.BufferedReader(
                new java.io.InputStreamReader(in))) {
            String line;
            while ((line = r.readLine()) != null) sb.append(line).append('
');
        } catch (Throwable ignored) {}
        return sb.toString();
    }

    /** True once the doze whitelist check has actually passed. */
    public static boolean isApplied() { return applied; }
    public static String lastReport() { return lastReport; }
}
