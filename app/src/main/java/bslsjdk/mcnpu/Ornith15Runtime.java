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
    private static volatile boolean mlxValidated;
    private static volatile String lastInfo = "NOT_LOADED";

    private Ornith15Runtime() {}

    public static synchronized String load(String modelPath, long contextTokens) {
        if (modelPath == null || modelPath.isEmpty())
            return "ERR ORNITH15_RUNTIME null_model";
        loaded = false;
        mlxValidated = false;
        if (contextTokens <= 0 || contextTokens > 262144)
            return "ERR ORNITH15_RUNTIME context=" + contextTokens;

        File f = new File(modelPath);
        if (!f.isFile() || f.length() <= 0)
            return "ERR ORNITH15_RUNTIME model_missing=" + modelPath;

        // The native llama.cpp bridge is retained as the GGUF regression
        // baseline. Never pass the actual MLX Safetensors target into it.
        if (modelPath.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors")) {
            String probe = Ornith15MlxProbe.inspect(modelPath, contextTokens);
            if (!probe.startsWith("OK ORNITH15_MLX_PROBE/1")) {
                loaded = false;
                mlxValidated = false;
                lastInfo = probe;
                return probe;
            }
            loaded = false;
            mlxValidated = true;
            lastInfo = probe + " executor=NOT_YET_ATTACHED";
            return lastInfo;
        }

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

    /**
     * Runs the real local llama.cpp baseline when the optional llama backend is
     * compiled. No network transport is involved. This is intentionally a
     * synchronous correctness path before MCNPU acceleration is inserted.
     */
    public static synchronized String generate(String prompt, int maxTokens) {
        if (mlxValidated)
            return "ERR ORNITH15_RUNTIME mlx_safetensors_executor_not_attached";
        if (!loaded) return "ERR ORNITH15_RUNTIME not_loaded";
        if (prompt == null || prompt.isEmpty()) return "ERR ORNITH15_RUNTIME empty_prompt";
        try {
            String r = nativeGenerate(prompt, maxTokens);
            return r == null ? "ERR ORNITH15_RUNTIME null_generate" : r;
        } catch (Throwable t) {
            return "ERR ORNITH15_RUNTIME generate_" + t.getClass().getSimpleName();
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
        mlxValidated = false;
        lastInfo = "NOT_LOADED";
    }

    public static boolean isLoaded() {
        return loaded;
    }

    /** True when the target MLX Safetensors file passed structural validation. */
    public static boolean isMlxValidated() {
        return mlxValidated;
    }

    private static native String nativeLoad(String modelPath, long contextTokens);
    private static native String nativeGenerate(String prompt, int maxTokens);
    private static native String nativeInfo();
    private static native void nativeUnload();
}
