package bslsjdk.mcnpu;

import android.content.pm.PackageManager;
import rikka.shizuku.Shizuku;

public final class ShizukuHelper {
    public static final int REQUEST_CODE = 2401;
    private static volatile String lastResult = "尚未申请";
    private static Runnable callback;

    private static final Shizuku.OnRequestPermissionResultListener LISTENER =
            (requestCode, result) -> {
                if (requestCode != REQUEST_CODE) return;
                lastResult = result == PackageManager.PERMISSION_GRANTED ? "已授权" : "未授权";
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
        if (Shizuku.shouldShowRequestPermissionRationale()) return "Shizuku：曾拒绝，请重新授权";
        return "Shizuku：已连接，等待授权";
    }

    public static String result() { return lastResult; }
}