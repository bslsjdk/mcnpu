package bslsjdk.mcnpu;

import android.content.pm.PackageManager;

import rikka.shizuku.Shizuku;

public final class ShizukuHelper {
    public static final int REQUEST_CODE = 2401;

    private ShizukuHelper() {}

    public static boolean available() {
        try {
            return Shizuku.pingBinder();
        } catch (Throwable t) {
            return false;
        }
    }

    public static boolean granted() {
        try {
            return available()
                    && Shizuku.checkSelfPermission() == PackageManager.PERMISSION_GRANTED;
        } catch (Throwable t) {
            return false;
        }
    }

    public static void requestPermission() {
        if (!available()) return;
        if (granted()) return;
        Shizuku.requestPermission(REQUEST_CODE);
    }

    public static String status() {
        if (!available()) return "SHIZUKU_OFFLINE";
        if (granted()) return "SHIZUKU_GRANTED";
        return "SHIZUKU_NOT_GRANTED";
    }
}
