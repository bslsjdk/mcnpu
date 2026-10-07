package bslsjdk.mcnpu;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.PowerManager;
import android.provider.Settings;
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
    /**
     * Set once the shell factory is found to be missing. That is a property of the Shizuku
     * build on the device, not a transient failure, so no later pass can fix it.
     */
    private static volatile boolean unsupported = false;
    /**
     * Application context captured from the startup call. Only the application
     * context is kept, never the service or an activity, so holding it cannot
     * leak one.
     */
    private static volatile Context sAppCtx = null;

    private NpuKeepAlive() {}

    /** Fire-and-forget; safe to call from the service startup path. */
    public static synchronized void apply(final Context ctx) {
        if (ctx != null && sAppCtx == null) sAppCtx = ctx.getApplicationContext();
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

            // Retrying exists to fight ROMs that quietly undo these settings. It cannot fight a
            // Shizuku build that has no way to run a command, so a permanent failure reports
            // itself once and the thread exits rather than spinning every ten minutes forever.
            if (unsupported) {
                // Shizuku has no way to run a command, so the doze whitelist is
                // out of reach: only a shell identity can write it. What an app
                // can still obtain on its own is the battery-optimization
                // opt-out, granted once by the user in a system dialog. It is
                // not the doze whitelist and does not replace it - it does not
                // stop doze - but it is the only measure left that needs no
                // shell, and with none of them the service stays a candidate
                // for LMK while a world load is allocating hundreds of MB.
                lastReport = fallbackBatteryOpt()
                        + (lastError == null ? "" : " err=" + lastError);
                android.util.Log.i(TAG, "KEEPALIVE " + lastReport);
                return;
            }

            if (!sleepQuietly(REAPPLY_MS)) return;
        }
    }

    /**
     * The no-shell path: ask the system for a battery-optimization exemption.
     *
     * Deliberately tried only once per process. The dialog is a user-visible
     * interruption and re-raising it on every re-apply pass would turn a
     * background service into a notification nuisance.
     */
    private static synchronized String fallbackBatteryOpt() {
        if (sFallbackDone) return lastReport;
        sFallbackDone = true;

        Context c = sAppCtx;
        if (c == null) return "Shizuku 不提供 shell 接口，且无 Context 可发起省电豁免请求";

        boolean ignoring = false;
        try {
            PowerManager pm = (PowerManager) c.getSystemService(Context.POWER_SERVICE);
            ignoring = pm != null && pm.isIgnoringBatteryOptimizations(PKG);
        } catch (Throwable t) {
            android.util.Log.w(TAG, "KEEPALIVE isIgnoringBatteryOptimizations failed", t);
        }
        if (ignoring) {
            applied = true;
            return "Shizuku 不提供 shell 接口 · 已在系统省电豁免名单(doze 白名单仍无法加入)";
        }

        try {
            Intent i = new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                    Uri.parse("package:" + PKG));
            i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            c.startActivity(i);
            return "Shizuku 不提供 shell 接口 · 已弹出系统省电豁免请求，请点「允许」(doze 白名单仍需 shell)";
        } catch (Throwable t) {
            // Most likely the background-activity-start restriction on Android
            // 10+. Say exactly that instead of "failed", so the next pass knows
            // the dialog itself is unreachable and the user must go via Settings.
            return "Shizuku 不提供 shell 接口，且无法弹出豁免请求("
                    + t.getClass().getSimpleName()
                    + ") · 请手动: 设置 → 应用 → MCNPU → 电池 → 不受限制";
        }
    }

    /**
     * Fallback already attempted in this process. Without it every re-apply
     * pass would raise the system dialog again.
     */
    private static volatile boolean sFallbackDone = false;

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
            Object proc = newProcess(shizuku, cmd);
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

    /**
     * Shizuku ships newProcess with a different signature across versions and the one this
     * was written against does not exist on every build - on this device it throws
     * NoSuchMethodException for (String[],String[],String). Rather than keep guessing,
     * enumerate the overloads that are actually present and adapt to whichever fits.
     *
     * When none fit, the error names the signatures that DO exist. A failure that reports the
     * real shape is fixable on the next pass; a bare NoSuchMethodException is just a dead end.
     */
    private static Object newProcess(Class<?> shizuku, String[] cmd) throws Exception {
        StringBuilder seen = new StringBuilder();
        for (Method m : shizuku.getMethods()) {
            if (!"newProcess".equals(m.getName())) continue;
            Class<?>[] p = m.getParameterTypes();
            seen.append("(");
            for (Class<?> x : p) seen.append(x.getSimpleName()).append(" ");
            seen.append(") ");
            Object[] args = argsFor(p, cmd);
            if (args == null) continue;
            m.setAccessible(true);
            return m.invoke(null, args);
        }
        // Nothing on Shizuku itself. On API 13+ the factory was dropped and the process class
        // is constructed directly, so try that before giving up - it is the same shell identity
        // and the same command, just reached by a different door.
        try {
            Class<?> rp = Class.forName("rikka.shizuku.ShizukuRemoteProcess");
            for (java.lang.reflect.Constructor<?> c : rp.getConstructors()) {
                Class<?>[] p = c.getParameterTypes();
                seen.append("[ctor(");
                for (Class<?> x : p) seen.append(x.getSimpleName()).append(" ");
                seen.append(")] ");
                Object[] args = argsFor(p, cmd);
                if (args == null) continue;
                c.setAccessible(true);
                return c.newInstance(args);
            }
        } catch (ClassNotFoundException e) {
            seen.append("[no ShizukuRemoteProcess] ");
        }
        // A missing factory is not something a later pass can fix, so say so once and stop
        // instead of retrying every ten minutes for the life of the process.
        unsupported = true;
        throw new NoSuchMethodException(
                "newProcess(String[],String[],String) - overloads present: " + seen);
    }

    /** Arguments for one overload, or null when its shape cannot take a plain command. */
    private static Object[] argsFor(Class<?>[] p, String[] cmd) {
        if (p.length < 2 || p[0] != String[].class) return null;
        switch (p.length) {
            case 2:
                return new Object[]{cmd, null};
            case 3:
                if (p[1] != String[].class || p[2] != String.class) return null;
                return new Object[]{cmd, null, null};
            case 4:
                if (p[1] != String[].class || p[2] != String.class) return null;
                Object tail = p[3] == boolean.class ? (Object) Boolean.FALSE
                        : p[3] == int.class ? (Object) Integer.valueOf(0) : null;
                if (tail == null) return null;
                return new Object[]{cmd, null, null, tail};
            default:
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
