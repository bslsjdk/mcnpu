package bslsjdk.mcnpu;

public final class NpuRuntime {
    private static volatile boolean ready;
    private static volatile String lastError = "not initialized";
    private NpuRuntime() {}

    public static synchronized boolean init() {
        if (ready) return true;
        try {
            System.loadLibrary("mcnpu");
            nativeConfigure("logLevel=DEBUG;deviceRetries=1");
            ready = nativeInit();
            lastError = ready ? "" : nativeGetDeviceInfo();
        } catch (Throwable t) {
            ready = false;
            lastError = t.toString();
        }
        return ready;
    }
    public static boolean isReady() { return ready; }
    public static String getLastError() { return lastError; }
    public static String status() { return ready ? nativeGetDeviceInfo() : "NPU_OFFLINE " + lastError; }
    public static boolean smoke() { return ready && nativeTest(); }
    public static String add(float[] a, float[] b) { return ready ? nativeAdd(a,b) : "ERR " + lastError; }
    public static void shutdown() { if (ready) { nativeShutdown(); ready=false; } }

    private static native void nativeConfigure(String tuning);
    private static native boolean nativeInit();
    private static native String nativeGetDeviceInfo();
    private static native boolean nativeTest();
    private static native String nativeAdd(float[] a, float[] b);
    private static native void nativeShutdown();
}
