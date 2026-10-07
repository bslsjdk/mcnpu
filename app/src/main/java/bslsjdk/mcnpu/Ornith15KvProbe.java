package bslsjdk.mcnpu;

/**
 * Native micro-probe for the Ornith-1.5 KV page codec.
 *
 * This intentionally allocates only a tiny synthetic page. It validates the
 * quantize/dequantize path and reports the page-based memory model without
 * allocating a 64K cache.
 */
public final class Ornith15KvProbe {
    public static final int F16 = 0;
    public static final int Q8_Q8 = 1;
    public static final int Q8_Q5 = 2;
    public static final int Q8_Q4 = 3;

    private Ornith15KvProbe() {}

    public static String probe(int tokens, int mode) {
        if (tokens <= 0 || tokens > 262144) return "ERR ORNITH15_KV bad_tokens";
        if (mode < F16 || mode > Q8_Q4) return "ERR ORNITH15_KV bad_mode";
        try {
            String r = nativeProbe(tokens, mode);
            return r == null ? "ERR ORNITH15_KV null" : r;
        } catch (Throwable t) {
            return "ERR ORNITH15_KV " + t.getClass().getSimpleName() + ": " + t.getMessage();
        }
    }

    public static String probeAll(int tokens) {
        StringBuilder b = new StringBuilder();
        for (int mode = F16; mode <= Q8_Q4; mode++) {
            if (mode != F16) b.append('\n');
            b.append(probe(tokens, mode));
        }
        return b.toString();
    }

    private static native String nativeProbe(int tokens, int mode);
}
