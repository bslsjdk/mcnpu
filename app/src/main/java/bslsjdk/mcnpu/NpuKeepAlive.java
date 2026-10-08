package com.bslsjdk.mcnpu.next;

import android.app.AppOpsManager;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.PowerManager;
import android.os.Process;
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
 *   is not looking at.
 *
 * What can actually be achieved, in descending order of value:
 *   1. Deviceidle (doze) whitelist - needs a shell identity, so only Shizuku can
 *      write it. On a Shizuku build without newProcess this is permanently out of
 *      reach and no amount of retrying changes that.
 *   2. Battery-optimization opt-out - any app can request it with a system dialog.
 *   3. AppOps RUN_IN_BACKGROUND / RUN_ANY_IN_BACKGROUND / WAKE_LOCK - the ROM-level
 *      freezes. Writing them needs a shell, but READING them does not, so their
 *      current state is always knowable.
 *
 * The previous version derived every field from a shell command. On this device
 * there is no shell, so the report was "uid=? doze_whitelist=no RUN_IN_BACKGROUND=?
 * ..." - four unknowns - and the header still said "部分生效", which claims a
 * partial success for a pass that applied nothing. A report has to be built from
 * what was verified, and the shell-free half of this was never being asked.
 *
 * Every step is read back rather than assumed.
 */
public final class NpuKeepAlive {
    private static final String TAG = "MCNPU";
    private static final String PKG = "com.bslsjdk.mcnpu.next";
    /** ROMs reset the standby bucket and appops on their own schedule; re-apply. */
    private static final long REAPPLY_MS = 10 * 60 * 1000L;
    /** Poll interval while waiting for the Shizuku permission to be granted. */
    private static final long WAIT_SHIZUKU_MS = 10 * 1000L;
    /**
     * Bounded: waiting for Shizuku is only worth doing while it might still turn up.
     * Past this the shell-free measures are what is left, and they can be applied
     * without it, so the thread stops waiting and applies them.
     */
    private static final int MAX_SHIZUKU_WAITS = 3;

    // AppOpsManager.MODE_* values, fixed in the platform.
    private static final int MODE_ALLOWED = 0;
    private static final int MODE_IGNORED = 1;
    private static final int MODE_ERRORED = 2;
    private static final int MODE_DEFAULT = 3;
    private static final int MODE_FOREGROUND = 4;

    private static volatile String lastReport = "未执行";
    private static volatile boolean applied = false;
    private static volatile boolean running = false;
    /** First failure reason from the most recent pass, surfaced at most once. */
    private static volatile String lastError = null;
    /**
     * Set once the shell factory is found to be missing. That is a property of the
     * Shizuku build on the device, not a transient failure, so no later pass can
     * fix it and retrying would only spin.
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
        // Wait for Shizuku rather than giving up immediately: this thread starts
        // when the service starts, which is normally BEFORE the user has accepted
        // the Shizuku prompt. Bounded, though - the shell-free measures below do
        // not need Shizuku, and waiting forever for a shell that may never come
        // would leave them unapplied and the report permanently stale.
        int waited = 0;
        while (waited < MAX_SHIZUKU_WAITS && (!ShizukuHelper.available() || !ShizukuHelper.granted())) {
            // Report the shell-free state while waiting, so the diagnostic shows
            // something real instead of "未执行" for the first thirty seconds.
            lastReport = "等待 Shizuku(" + (ShizukuHelper.available() ? "待授权" : "未运行")
                    + ") · " + join(shellFreeState());
            android.util.Log.i(TAG, "KEEPALIVE wait: " + lastReport);
            waited++;
            if (!sleepQuietly(WAIT_SHIZUKU_MS)) return;
        }

        while (true) {
            ArrayList<String> results = new ArrayList<String>();
            lastError = null;

            // Everything here is knowable without a shell.
            results.addAll(shellFreeState());

            // The doze whitelist is the one measure that needs a shell identity.
            // Attempted only if one is reachable; otherwise the field says why it
            // is missing rather than reporting a bare "no".
            String whitelist;
            if (unsupported) {
                whitelist = "需shell";
            } else {
                shell("cmd", "deviceidle", "whitelist", "+" + PKG);
                whitelist = verifyWhitelist() ? "yes" : "no";
            }
            results.add("doze=" + whitelist);

            // Only the whitelist decides "applied": it is the only measure strong
            // enough to matter on its own. The shell-free ones are reported either
            // way, but claiming success over a list of unknowns is what produced
            // the "部分生效" over four "?" in the first place.
            int measures = countMeasures(results);
            String head;
            if ("yes".equals(whitelist)) {
                head = "已保活(doze白名单已加入)";
            } else if (measures == 0) {
                head = "未生效(无任何措施在读回中确认)";
            } else {
                head = "部分生效(" + measures + " 项应用层措施, doze 白名单" + whitelist + ")";
            }
            lastReport = head + " · " + join(results)
                    + (lastError == null ? "" : " err=" + lastError);
            android.util.Log.i(TAG, "KEEPALIVE " + lastReport);

            // A Shizuku build with no way to run a command cannot be fixed by a
            // later pass: report the fallback once and stop instead of spinning
            // every ten minutes for the life of the process.
            if (unsupported) {
                lastError = null;
                lastReport = fallbackBatteryOpt();
                android.util.Log.i(TAG, "KEEPALIVE " + lastReport);
                return;
            }

            if (!sleepQuietly(REAPPLY_MS)) return;
        }
    }

    /** How many of the reported measures are actually in force. */
    private static int countMeasures(ArrayList<String> results) {
        int n = 0;
        for (int i = 0; i < results.size(); i++) {
            String r = results.get(i);
            if (r.startsWith("battery_opt= exempt")) n++;
            else if (r.startsWith("RUN_IN_BACKGROUND=allow")) n++;
            else if (r.startsWith("RUN_ANY_IN_BACKGROUND=allow")) n++;
            else if (r.startsWith("WAKE_LOCK=allow")) n++;
            else if (r.startsWith("standby=ACTIVE")) n++;
        }
        return n;
    }

    /**
     * Everything that can be read back without a shell.
     *
     * These used to come from "cmd appops get", which needs the same shell identity
     * the device does not have, so they came back as "?". AppOpsManager reads them
     * directly for the app's own uid, which needs no permission at all.
     */
    private static ArrayList<String> shellFreeState() {
        ArrayList<String> out = new ArrayList<String>();
        int uid = Process.myUid();
        out.add("uid=" + uid);

        Context c = sAppCtx;
        if (c == null) {
            out.add("battery_opt=?");
            out.add("RUN_IN_BACKGROUND=?");
            out.add("RUN_ANY_IN_BACKGROUND=?");
            out.add("WAKE_LOCK=?");
            out.add("standby=?");
            return out;
        }

        // Battery optimization: the one measure an app can obtain on its own.
        String batt;
        try {
            PowerManager pm = (PowerManager) c.getSystemService(Context.POWER_SERVICE);
            batt = (pm != null && pm.isIgnoringBatteryOptimizations(PKG)) ? "exempt" : "restricted";
        } catch (Throwable t) {
            batt = "?";
        }
        out.add("battery_opt=" + batt);

        AppOpsManager ao = null;
        try {
            ao = (AppOpsManager) c.getSystemService(Context.APP_OPS_SERVICE);
        } catch (Throwable ignored) {
        }
        out.add("RUN_IN_BACKGROUND=" + opState(ao, "OPSTR_RUN_IN_BACKGROUND", uid));
        out.add("RUN_ANY_IN_BACKGROUND=" + opState(ao, "OPSTR_RUN_ANY_IN_BACKGROUND", uid));
        out.add("WAKE_LOCK=" + opState(ao, "OPSTR_WAKE_LOCK", uid));

        // Standby bucket. ACTIVE keeps it out of standby quota throttling. Read
        // reflectively because the constant is not on every compile SDK, and a
        // compile break here would cost more than the field is worth.
        String bucket = "?";
        try {
            Object usm = c.getSystemService("usagestats");
            if (usm != null) {
                Method m = usm.getClass().getMethod("getAppStandbyBucket");
                Object v = m.invoke(usm);
                int b = v == null ? -1 : ((Integer) v).intValue();
                bucket = b == 5 ? "EXEMPTED" : b == 10 ? "ACTIVE" : b == 20 ? "WORKING_SET"
                        : b == 30 ? "FREQUENT" : b == 40 ? "RARE" : b == 50 ? "RESTRICTED"
                        : b < 0 ? "?" : "bucket" + b;
            }
        } catch (Throwable ignored) {
        }
        out.add("standby=" + bucket);
        return out;
    }

    /**
     * One AppOps mode, read through the public checker.
     *
     * The OPSTR_ constants are looked up by name: they exist from API 28 but are
     * not on every compile SDK this project has been built against, and a hard
     * reference to a missing constant is a build break rather than a "?".
     */
    private static String opState(AppOpsManager ao, String constName, int uid) {
        if (ao == null) return "?";
        String op;
        try {
            op = (String) AppOpsManager.class.getField(constName).get(null);
        } catch (Throwable t) {
            return "n/a";
        }
        if (op == null) return "n/a";
        try {
            int m = ao.checkOpNoThrow(op, uid, PKG);
            switch (m) {
                case MODE_ALLOWED: return "allow";
                case MODE_IGNORED: return "ignore";
                case MODE_ERRORED: return "deny";
                case MODE_DEFAULT: return "default";
                case MODE_FOREGROUND: return "foreground";
                default: return "mode" + m;
            }
        } catch (Throwable t) {
            return "?";
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
            return "部分生效(系统省电豁免已获得) · doze 白名单需 shell 无法加入 · "
                    + join(shellFreeState());
        }

        try {
            Intent i = new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                    Uri.parse("package:" + PKG));
            i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            c.startActivity(i);
            return "等待用户确认系统省电豁免 · doze 白名单需 shell 无法加入 · "
                    + join(shellFreeState());
        } catch (Throwable t) {
            // Most likely the background-activity-start restriction on Android
            // 10+. Say exactly that instead of "failed", so the next pass knows
            // the dialog itself is unreachable and the user must go via Settings.
            return "未生效(无法弹出豁免请求: " + t.getClass().getSimpleName()
                    + ") · 请手动: 设置 → 应用 → MCNPU → 电池 → 不受限制 · "
                    + join(shellFreeState());
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
