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

    /** Exact native failure for the last binary INT8 submission. */
    public static String getLastNativeError() {
        try { return nativeGetLastError(); } catch (Throwable t) { return "ERR NATIVE_ERROR_QUERY " + t.getClass().getSimpleName(); }
    }
    public static synchronized String status() {
        return ready ? nativeGetDeviceInfo() : "NPU_OFFLINE " + lastError;
    }
    public static synchronized boolean smoke() { return ready && nativeTest(); }

    /**
     * Largest ADD length this device accepted.
     *
     * The 16384 default is not a device limit - it is only the largest size we had
     * verified, and it lives in our own native code. The real ceiling is measured
     * by addProbe(). Everything that caps an ADD length must read this instead of
     * hardcoding 16384, or a device that accepts more will silently never use it.
     */
    public static int maxAddElements() {
        if (!ready) return 16384;
        try {
            String s = nativeAddMax();
            if (s != null) return Integer.parseInt(s.trim());
        } catch (Throwable t) { /* fall through to the default */ }
        return 16384;
    }

    /**
     * Walks candidate ADD sizes on the real HTP and reports which ones build.
     *
     * This replaces a guess with a measurement: the ladder used to reject anything
     * above 16384 before it could reach the device, so "the device caps ADD at
     * 16384" was a conclusion our own code produced, not something the HTP said.
     * Runs one graph build per candidate size and then flushes the graph cache,
     * so it is a diagnostic, not something to call on a hot path.
     */
    // Not synchronized on purpose. The native side already serializes on
    // gRuntimeMutex, and holding the class lock across a probe that builds one
    // graph per candidate size would stall every IPC request for its duration.
    public static String addProbe() {
        if (!ready) return "ERR NPU_NOT_READY";
        try {
            String r = nativeAddProbe();
            return r == null ? "ERR ADD_PROBE_NULL" : r;
        } catch (Throwable t) {
            return "ERR ADD_PROBE_EXCEPTION " + t.getClass().getSimpleName();
        }
    }
    private static native String nativeAddProbe();

    /**
     * Every op the HTP backend actually registered, including its built-in package.
     *
     * This answers "can Perlin live on the NPU?" with a fact instead of a guess.
     * The noise kernel needs a table lookup per corner; if the backend has no
     * gather, the only substitutes cost hundreds of MB per chunk section and the
     * whole noise route is not worth writing. We used to infer op support from
     * addNode failures, but those cannot distinguish "op does not exist" from
     * "we passed the wrong params", which is how a working op gets written off.
     */
    public static String opProbe() {
        if (!ready) return "ERR OPPROBE NPU_NOT_READY";
        try {
            String r = nativeOpProbe();
            return r == null ? "ERR OPPROBE_NULL" : r;
        } catch (Throwable t) {
            return "ERR OPPROBE_EXCEPTION " + t.getClass().getSimpleName();
        }
    }
    private static native String nativeOpProbe();

    /**
     * Which noise path this backend can actually support.
     *
     * A: the whole Perlin evaluation as one graph. Needs Gather, ElementWiseUnary
     *    and ElementWiseBinary.
     * C: the host resolves perm[] and the gradient table - both pure lookups the
     *    CPU already does well - and the graph does only the arithmetic: dot
     *    products, the fade polynomial and the trilinear blend. Needs
     *    ElementWiseBinary alone, which is far more basic than Gather.
     *
     * Reported separately from opProbe because the interesting answer here is not
     * the op list but the path it selects.
     */
    public static String perlinCap() {
        if (!ready) return "ERR PERLIN_CAP NPU_NOT_READY";
        try {
            String r = nativePerlinCap();
            return r == null ? "ERR PERLIN_CAP_NULL" : r;
        } catch (Throwable t) {
            return "ERR PERLIN_CAP_EXCEPTION " + t.getClass().getSimpleName();
        }
    }
    private static native String nativePerlinCap();

    /**
     * Build the noise graph, run it, and compare every point against a CPU
     * reference that is written the textbook way rather than the way the graph is,
     * so a bug shared by both cannot produce a green result.
     */
    public static String perlinBench(int n) {
        if (!ready) return "ERR PERLIN NPU_NOT_READY";
        try {
            String r = nativePerlinBench(n);
            return r == null ? "ERR PERLIN_NULL" : r;
        } catch (Throwable t) {
            return "ERR PERLIN_EXCEPTION " + t.getClass().getSimpleName();
        }
    }
    private static native String nativePerlinBench(int n);
    private static native String nativeAddMax();

    /**
     * Drop every cached graph and rebuild the QNN context.
     *
     * The graph budget is small on purpose and a benchmark builds one graph per
     * candidate shape, so a sweep can fill the cache before the game submits real
     * work - and then the first production shape is the one that pays the teardown.
     */
    public static String flushGraphs() {
        if (!ready) return "ERR not ready";
        try {
            String r = nativeFlushGraphs();
            return r == null ? "ERR FLUSH_GRAPHS_NULL" : r;
        } catch (Throwable t) {
            return "ERR FLUSH_GRAPHS_EXCEPTION " + t.getClass().getSimpleName();
        }
    }
    private static native String nativeFlushGraphs();

    /**
     * Ends an in-flight probe at its next candidate.
     *
     * The probe builds one graph per size and holds the device lock for each, so
     * a real request that arrives mid-probe would otherwise wait for a diagnostic
     * it has no interest in. The probe reports what it learned up to that point
     * instead of nothing.
     */
    public static void abortProbe() {
        try { nativeAbortProbe(); } catch (Throwable ignored) { }
    }
    private static native void nativeAbortProbe();

    /**
     * Self test that keeps the reason.
     *
     * smoke() returns a bit, so a failing self test reaches the client as one fixed
     * string no matter what actually went wrong. This returns the underlying reply
     * instead, so the client can show whether it was graph creation, execution, or
     * output verification - and with what values.
     */
    public static synchronized String smokeDetail() {
        if (!ready) return "ERR NPU_NOT_READY";
        String r = nativeTestDetail();
        return r == null ? "ERR NATIVE_NULL" : r;
    }
    /**
     * Adds a and b on the HTP and writes the result into out.
     *
     * verify=true also compares every element against a CPU reference, which is
     * a full CPU pass over the data - correct for a test, wrong for a data path,
     * because it makes the call look no faster than the CPU it replaces.
     */
    public static synchronized String addInto(float[] a, float[] b, float[] out, boolean verify) {
        if (!ready) return "ERR " + lastError;
        if (a == null || b == null || out == null) return "ERR NULL";
        if (a.length != b.length || out.length < a.length) return "ERR SIZE";
        return nativeAddInto(a, b, out, verify);
    }

    public static synchronized String add(float[] a, float[] b) {
        if (!ready) return "ERR " + lastError;
        return nativeAdd(a, b);
    }

    /** m x k times k x n on the HTP; result also reports cpu_us and speedup. */
    public static synchronized String matMul(int m, int k, int n) {
        if (!ready) return "ERR " + lastError;
        return nativeMatMul(m, k, n);
    }

    /** Same matmul, but with fp16 tensors (the HTP-native datatype). */
    public static synchronized String matMulFp16(int m, int k, int n) {
        if (!ready) return "ERR " + lastError;
        return nativeMatMulFp16(m, k, n);
    }

    /** INT8 quantized matmul: the datatype HTP actually accelerates natively. */
    public static synchronized String matMulInt8(int m, int k, int n) {
        if (!ready) return "ERR " + lastError;
        return nativeMatMulInt8(m, k, n);
    }

    /** Element-wise int8 batch transform. op 0 = add, 1 = multiply. */
    public static synchronized String xform(int op, int n) {
        if (n <= 0 || n > 65536) return "ERR SIZE_UNSUPPORTED max=65536";
        return nativeXform(op, n);
    }

    private static native String nativeXform(int op, int n);

    /** Prebuild one INT8 graph so the first real terrain request avoids cold graph setup. */
    public static synchronized String prewarmMatMulInt8(int m, int k, int n) {
        if (!ready) return "ERR " + lastError;
        if (m <= 0 || k <= 0 || n <= 0) return "ERR SIZE";
        long aBytes = (long) m * k, bBytes = (long) k * n;
        if (aBytes > 16L * 1024L * 1024L || bBytes > 16L * 1024L * 1024L)
            return "ERR PREWARM_INPUT_TOO_LARGE";
        byte[] a = new byte[(int) aBytes];
        byte[] b = new byte[(int) bBytes];
        for (int i = 0; i < a.length; i++) a[i] = (byte) (((i * 13 + 7) & 31) - 16);
        for (int i = 0; i < b.length; i++) b[i] = (byte) (((i * 17 + 3) & 31) - 16);
        long t0 = System.nanoTime();
        byte[] out = nativeMatMulInt8Buf(a, b, m, k, n);
        long us = (System.nanoTime() - t0) / 1000L;
        return out == null ? "ERR PREWARM_FAILED elapsed_us=" + us
                : "OK PREWARM elapsed_us=" + us + " bytes=" + out.length;
    }

    /** Real data path: int8 tensors in, int8 result out (with scaleC prefix). */
    public static synchronized byte[] matMulInt8Buf(byte[] a, byte[] b, int m, int k, int n) {
        if (!ready) return null;
        return nativeMatMulInt8Buf(a, b, m, k, n);
    }
    public static synchronized void shutdown() {
        if (!ready) return;
        nativeShutdown();
        ready = false;
    }

    private static native String nativeGetLastError();
    private static native boolean nativeInit(String qnnDir, String workDir);
    private static native String nativeGetDeviceInfo();
    private static native boolean nativeTest();
    private static native String nativeTestDetail();
    private static native String nativeAdd(float[] a, float[] b);
    private static native String nativeAddInto(float[] a, float[] b, float[] out, boolean verify);
    /** Deterministic matmul on HTP with CPU reference; data is generated in-service. */
    private static native String nativeMatMul(int m, int k, int n);
    private static native String nativeMatMulFp16(int m, int k, int n);
    private static native String nativeMatMulInt8(int m, int k, int n);
    /** Returns [4-byte LE scaleC][m*n int8 result], or null on failure. */
    private static native byte[] nativeMatMulInt8Buf(byte[] a, byte[] b, int m, int k, int n);
    private static native void nativeShutdown();
}
