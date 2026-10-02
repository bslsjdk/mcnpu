package bslsjdk.mcnpu;

import android.content.pm.PackageManager;
import rikka.shizuku.Shizuku;

public final class ShizukuHelper {
    public static final int REQUEST_CODE = 2401;

    private static final Shizuku.OnRequestPermissionResultListener LISTENER =
            (requestCode, result) -> {
                if (requestCode == REQUEST_CODE) {
                    android.util.Log.i("MCNPU", "Shizuku permission result=" + result);
                }
            };

    private ShizukuHelper() {}

    public static void init() {
        try {
            Shizuku.addRequestPermissionResultListener(LISTENER);
        } catch (Throwable t) {
            android.util.Log.e("MCNPU", "Shizuku listener init failed", t);
        }
    }

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
        if (!available()) return "SHIZUKU_OFFLINE";
        if (granted()) return "SHIZUKU_GRANTED";
        return "SHIZUKU_NOT_GRANTED";
    }
}
