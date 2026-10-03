package bslsjdk.mcnpu;

import android.content.Context;
import java.io.*;
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
            ready = nativeInit(qnnDir, work.getAbsolutePath());
            lastError = ready ? "" : nativeGetDeviceInfo();
        } catch (Throwable t) {
            ready = false;
            lastError = t.toString();
        }
        return ready;
    }

    private static String extractQnnLibs(Context context) throws Exception {
        File dst = new File(context.getFilesDir(), "qnnlibs");
        File stamp = new File(dst, ".stack-version");
        String stackVersion = readAssetText(context, "qnn-stack-version.txt");
        String installed = stamp.isFile() ? readFileText(stamp) : "";
        String[] names = context.getAssets().list("qnnlibs");
        if (names == null || names.length == 0) throw new IllegalStateException("assets/qnnlibs empty");

        boolean complete = stackVersion.equals(installed);
        if (complete) {
            for (String name : names) {
                File out = new File(dst, outputName(name));
                if (!out.isFile() || out.length() <= 0) { complete = false; break; }
            }
        }
        if (complete) return dst.getAbsolutePath();

        File parent = context.getFilesDir();
        File tmp = new File(parent, "qnnlibs.tmp");
        deleteRecursively(tmp);
        if (!tmp.mkdirs()) throw new IOException("mkdir qnn temp failed: " + tmp);

        try {
            for (String name : names) {
                File out = new File(tmp, outputName(name));
                try (InputStream raw = context.getAssets().open("qnnlibs/" + name);
                     InputStream in = name.endsWith(".gz") ? new GZIPInputStream(raw) : raw;
                     FileOutputStream fos = new FileOutputStream(out)) {
                    byte[] buf = new byte[65536];
                    int n;
                    while ((n = in.read(buf)) != -1) fos.write(buf, 0, n);
                }
                if (!out.isFile() || out.length() <= 0) throw new IOException("QNN copy incomplete: " + name);
                out.setReadable(true, false);
                out.setExecutable(true, false);
            }
            File tmpStamp = new File(tmp, ".stack-version");
            try (FileOutputStream out = new FileOutputStream(tmpStamp)) {
                out.write(stackVersion.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            }

            File backup = new File(parent, "qnnlibs.old");
            deleteRecursively(backup);
            if (dst.exists() && !dst.renameTo(backup)) {
                deleteRecursively(dst);
                if (dst.exists()) throw new IOException("cannot replace old qnnlibs");
            }
            if (!tmp.renameTo(dst)) {
                if (backup.exists()) backup.renameTo(dst);
                throw new IOException("cannot activate new qnnlibs");
            }
            deleteRecursively(backup);
        } catch (Throwable t) {
            deleteRecursively(tmp);
            throw t;
        }
        return dst.getAbsolutePath();
    }

    private static String outputName(String name) {
        return name.endsWith(".gz") ? name.substring(0, name.length() - 3) : name;
    }

    private static void deleteRecursively(File f) {
        if (!f.exists()) return;
        File[] children = f.listFiles();
        if (children != null) for (File child : children) deleteRecursively(child);
        if (!f.delete() && f.exists()) throw new IllegalStateException("delete failed: " + f);
    }

    private static String readAssetText(Context context, String path) throws IOException {
        try (InputStream in = context.getAssets().open(path)) {
            return readText(in);
        }
    }

    private static String readFileText(File f) throws IOException {
        try (InputStream in = new FileInputStream(f)) { return readText(in); }
    }

    private static String readText(InputStream in) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buf = new byte[4096];
        int n;
        while ((n = in.read(buf)) != -1) out.write(buf, 0, n);
        return out.toString("UTF-8").trim();
    }

    public static boolean isReady() { return ready; }
    public static String getLastError() { return lastError; }
    public static synchronized String status() {
        return ready ? nativeGetDeviceInfo() : "NPU_OFFLINE " + lastError;
    }
    public static synchronized boolean smoke() { return ready && nativeTest(); }
    public static synchronized String add(float[] a, float[] b) {
        if (!ready) return "ERR " + lastError;
        return nativeAdd(a, b);
    }

    /** m x k times k x n on the HTP; result also reports cpu_us and speedup. */
    public static synchronized String matMul(int m, int k, int n) {
        if (!ready) return "ERR " + lastError;
        return nativeMatMul(m, k, n);
    }
    public static synchronized void shutdown() {
        if (!ready) return;
        nativeShutdown();
        ready = false;
    }

    private static native boolean nativeInit(String qnnDir, String workDir);
    private static native String nativeGetDeviceInfo();
    private static native boolean nativeTest();
    private static native String nativeAdd(float[] a, float[] b);
    /** Deterministic matmul on HTP with CPU reference; data is generated in-service. */
    private static native String nativeMatMul(int m, int k, int n);
    private static native void nativeShutdown();
}
