package bslsjdk.mcnpu;

import java.io.File;

/**
 * Runtime boundary for the local Ornith-1.5-9B model.
 *
 * This is deliberately a loader/validation boundary first. It does not fake
 * token generation and it does not expose a network server. The next native
 * layer can attach a real GGUF inference engine behind this stable interface.
 */
public final class Ornith15Runtime {
    private static volatile boolean loaded;
    private static volatile String lastInfo = "NOT_LOADED";

    private Ornith15Runtime() {}

    public static synchronized String load(String modelPath, long contextTokens) {
        if (modelPath == null || modelPath.isEmpty())
            return "ERR ORNITH15_RUNTIME null_model";
        if (contextTokens <= 0 || contextTokens > 262144)
            return "ERR ORNITH15_RUNTIME context=" + contextTokens;

        File f = new File(modelPath);
        if (!f.isFile() || f.length() <= 0)
            return "ERR ORNITH15_RUNTIME model_missing=" + modelPath;

        try {
            System.loadLibrary("mcnpu");
            String r = nativeLoad(modelPath, contextTokens);
            lastInfo = r == null ? "ERR ORNITH15_RUNTIME null_native_reply" : r;
            loaded = r != null && r.startsWith("OK ORNITH15_RUNTIME/1");
            return lastInfo;
        } catch (Throwable t) {
            loaded = false;
            lastInfo = "ERR ORNITH15_RUNTIME " + t.getClass().getSimpleName() + ": " + t.getMessage();
            return lastInfo;
        }
    }

    public static synchronized String info() {
        try {
            String r = nativeInfo();
            return r == null ? lastInfo : r;
        } catch (Throwable t) {
            return lastInfo;
        }
    }

    public static synchronized void unload() {
        try { nativeUnload(); } catch (Throwable ignored) {}
        loaded = false;
        lastInfo = "NOT_LOADED";
    }

    public static boolean isLoaded() {
        return loaded;
    }

    private static native String nativeLoad(String modelPath, long contextTokens);
    private static native String nativeInfo();
    private static native void nativeUnload();
}
