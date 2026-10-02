package bslsjdk.mcnpu;

import android.content.Context;
import java.io.File;
import java.io.InputStream;
import java.io.FileOutputStream;
import java.util.zip.GZIPInputStream;

public final class NpuRuntime {
    private static volatile boolean ready;
    private static volatile String lastError = "not initialized";
    private NpuRuntime() {}

    public static synchronized boolean init(Context context) {
        if (ready) return true;
        try {
            System.loadLibrary("mcnpu");
            String qnnDir = extractQnnLibs(context.getApplicationContext());
            File work = new File(context.getFilesDir(), "qnnwork");
            if (!work.exists() && !work.mkdirs()) throw new IllegalStateException("mkdir qnnwork failed");
            String workDir = work.getAbsolutePath();
            nativeConfigure("logLevel=DEBUG;deviceRetries=1;qnnDir=" + qnnDir + ";workDir=" + workDir);
            ready = nativeInit(qnnDir, workDir);
            lastError = ready ? "" : nativeGetDeviceInfo();
        } catch (Throwable t) {
            ready = false;
            lastError = t.toString();
        }
        return ready;
    }
    private static String extractQnnLibs(Context context) throws Exception {
        File dst = new File(context.getFilesDir(), "qnnlibs");
        if (!dst.exists() && !dst.mkdirs()) throw new IllegalStateException("mkdir qnnlibs failed");
        String[] names = context.getAssets().list("qnnlibs");
        if (names == null || names.length == 0) throw new IllegalStateException("assets/qnnlibs empty");
        for (String name : names) {
            File out = new File(dst, name.endsWith(".gz") ? name.substring(0, name.length()-3) : name);
            if (out.exists() && out.length() > 0) continue;
            try (InputStream raw = context.getAssets().open("qnnlibs/" + name);
                 InputStream in = name.endsWith(".gz") ? new GZIPInputStream(raw) : raw;
                 FileOutputStream fos = new FileOutputStream(out)) {
                byte[] buf = new byte[65536];
                int n;
                while ((n = in.read(buf)) != -1) fos.write(buf, 0, n);
            }
            out.setReadable(true, false);
            out.setExecutable(true, false);
        }
        return dst.getAbsolutePath();
    }

    public static boolean isReady() { return ready; }
    public static String getLastError() { return lastError; }
    public static String status() { return ready ? nativeGetDeviceInfo() : "NPU_OFFLINE " + lastError; }
    public static boolean smoke() { return ready && nativeTest(); }
    public static String add(float[] a, float[] b) { return ready ? nativeAdd(a,b) : "ERR " + lastError; }
    public static void shutdown() { if (ready) { nativeShutdown(); ready=false; } }

    private static native void nativeConfigure(String tuning);
    private static native boolean nativeInit(String qnnDir, String workDir);
    private static native String nativeGetDeviceInfo();
    private static native boolean nativeTest();
    private static native String nativeAdd(float[] a, float[] b);
    private static native void nativeShutdown();
}
