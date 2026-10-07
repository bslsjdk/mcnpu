package bslsjdk.mcnpu;

import android.content.pm.PackageManager;
import rikka.shizuku.Shizuku;

public final class ShizukuHelper {
    public static final int REQUEST_CODE = 2401;
    private static volatile String lastResult = "尚未申请";
    /** True once this process has actually seen a permission result. */
    private static volatile boolean fired = false;
    private static Runnable callback;

    private static final Shizuku.OnRequestPermissionResultListener LISTENER =
            (requestCode, result) -> {
                if (requestCode != REQUEST_CODE) return;
                lastResult = result == PackageManager.PERMISSION_GRANTED ? "已授权" : "未授权";
                fired = true;
                android.util.Log.i("MCNPU", "Shizuku permission result=" + result);
                if (callback != null) callback.run();
            };

    private ShizukuHelper() {}

    public static void init() {
        try {
            Shizuku.removeRequestPermissionResultListener(LISTENER);
        } catch (Throwable ignored) {}
        try {
            Shizuku.addRequestPermissionResultListener(LISTENER);
        } catch (Throwable t) {
            android.util.Log.e("MCNPU", "Shizuku listener init failed", t);
        }
    }

    public static void setCallback(Runnable r) { callback = r; }

    public static boolean available() {
        try { return Shizuku.pingBinder(); }
        catch (Throwable t) { return false; }
    }

    public static boolean granted() {
        try {
            return available()
                    && Shizuku.checkSelfPermission() == PackageManager.PERMISSION_GRANTED;
        } catch (Throwable t) { return false; }
    }

    public static void requestPermission() {
        if (!available() || granted()) return;
        Shizuku.requestPermission(REQUEST_CODE);
    }

    public static String status() {
        if (!available()) return "Shizuku：未运行";
        if (granted()) return "Shizuku：MC NPU 已授权";
        try {
            if (Shizuku.shouldShowRequestPermissionRationale()) return "Shizuku：曾拒绝，请重新授权";
        } catch (Throwable ignored) {
            return "Shizuku：已连接，授权状态暂不可用";
        }
        return "Shizuku：已连接，等待授权";
    }

    /**
     * What the last permission request produced - but not at the cost of
     * contradicting the current state.
     *
     * lastResult only changes when the result listener fires, and that only
     * happens if this process both asked and got an answer. On a device where
     * the permission was granted in an earlier run, or granted by Shizuku's own
     * UI, the listener never fires and the field stayed "尚未申请" forever while
     * status() correctly said "已授权". Two lines of the same diagnostic
     * disagreeing about the same fact is worse than either being vague, so the
     * live state wins whenever the listener has nothing to add.
     */
    public static String result() {
        if (granted()) return fired ? "已授权(本次会话)" : "已授权(此前已授予)";
        if (!available()) return "Shizuku 未运行";
        return lastResult;
    }
}