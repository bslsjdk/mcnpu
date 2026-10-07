package bslsjdk.mcnpu;

/**
 * Minimal PQ2_0 format probe. This does not load model weights or allocate a
 * model-sized buffer; it only validates the 34-byte Prism block codec.
 */
public final class Bonsai2Pq2Probe {
    static {
        System.loadLibrary("mcnpu");
    }

    private Bonsai2Pq2Probe() {}

    public static String run() {
        try {
            String r = nativeRun();
            return r == null ? "ERR BONSAI2_PQ2_PROBE_NULL" : r;
        } catch (Throwable t) {
            return "ERR BONSAI2_PQ2_PROBE " + t.getClass().getSimpleName() + ": " + t.getMessage();
        }
    }

    private static native String nativeRun();

    public static String runHadamard() {
        try {
            String r = nativeHadamardRun();
            return r == null ? "ERR BONSAI2_HADAMARD_PROBE_NULL" : r;
        } catch (Throwable t) {
            return "ERR BONSAI2_HADAMARD_PROBE " + t.getClass().getSimpleName() + ": " + t.getMessage();
        }
    }

    private static native String nativeHadamardRun();

    /** Validates only GGUF header/metadata/tensor descriptors; never loads model weights. */
    public static String validateGguf(String path) {
        if (path == null || path.isEmpty()) return "ERR BONSAI2_PQ2_VALIDATE null_path";
        try {
            String r = nativeValidate(path);
            return r == null ? "ERR BONSAI2_PQ2_VALIDATE_NULL" : r;
        } catch (Throwable t) {
            return "ERR BONSAI2_PQ2_VALIDATE " + t.getClass().getSimpleName() + ": " + t.getMessage();
        }
    }

    private static native String nativeValidate(String path);
}
