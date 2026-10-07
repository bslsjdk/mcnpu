package bslsjdk.mcnpu;

/**
 * Pure arithmetic planner for Ornith-1.5-9B long-context state.
 *
 * This class does not allocate KV memory. It models the actual page codec used by
 * Ornith15KvProbe, including per-block FP32 scale metadata, so the runtime can
 * budget the representation before allocation.
 */
public final class Ornith15MemoryPlanner {
    public static final long HARD_RAM_BYTES = 4L * 1024L * 1024L * 1024L;
    public static final long FULL_ATTN_LAYERS = 8L;
    public static final long KV_HEADS = 4L;
    public static final long HEAD_DIM = 256L;
    public static final long KV_STREAMS = 2L;

    private Ornith15MemoryPlanner() {}

    public enum Mode {
        FP16,
        Q8_Q8,
        Q8_Q5,
        Q8_Q4
    }

    public static long fp16BytesPerToken() {
        return FULL_ATTN_LAYERS * KV_STREAMS * KV_HEADS * HEAD_DIM * 2L;
    }

    /**
     * Planning bit-widths for K/V. These are storage estimates only.
     * Q8_Q5 means 8 bits for K and 5 bits for V before block metadata/alignment.
     */
    /** Actual bytes/token for the current page codec, including FP32 scale metadata. */
    public static long codecBytesPerToken(Mode mode) {
        if (mode == Mode.FP16) return fp16BytesPerToken();
        long elementsPerToken = FULL_ATTN_LAYERS * KV_HEADS * HEAD_DIM;
        int kBits = 8;
        int vBits = mode == Mode.Q8_Q8 ? 8 : (mode == Mode.Q8_Q5 ? 5 : 4);
        long blocksPerToken = (elementsPerToken + 31L) / 32L;
        long payload = (elementsPerToken * (long) (kBits + vBits)) / 8L;
        long scaleBytes = blocksPerToken * 2L * Float.BYTES;
        return payload + scaleBytes;
    }

    public static double bitsPerElement(Mode mode) {
        switch (mode) {
            case FP16: return 16.0;
            case Q8_Q8: return 8.0;
            case Q8_Q5: return 6.5;
            case Q8_Q4: return 6.0;
            default: throw new AssertionError(mode);
        }
    }

    public static long estimatedBytes(long tokens, Mode mode) {
        if (tokens <= 0) return 0;
        double bytes = (double) codecBytesPerToken(mode) * tokens;
        if (bytes >= Long.MAX_VALUE) return Long.MAX_VALUE;
        return (long) Math.ceil(bytes);
    }

    /**
     * Returns the largest token count whose estimated KV storage stays inside
     * the supplied budget. Metadata/alignment overhead is deliberately excluded.
     */
    public static long maxTokens(long budgetBytes, Mode mode) {
        if (budgetBytes <= 0) return 0;
        double perToken = codecBytesPerToken(mode);
        return (long) Math.floor(budgetBytes / perToken);
    }

    /**
     * Conservative runtime budget. We leave 1 GiB outside the KV estimate for
     * mapped-weight resident pages, recurrent state, tokenizer/runtime objects,
     * NPU staging and scratch buffers. This is a planning policy, not a guarantee.
     */
    public static long kvBudgetForRuntime(long runtimeBudgetBytes) {
        if (runtimeBudgetBytes <= 0) return 0;
        long reserve = 1L * 1024L * 1024L * 1024L;
        return runtimeBudgetBytes > reserve ? runtimeBudgetBytes - reserve : 0;
    }

    public static String plan(long contextTokens) {
        if (contextTokens <= 0 || contextTokens > 262144)
            return "ERR ORNITH15_PLAN context=" + contextTokens;

        long kvBudget = kvBudgetForRuntime(HARD_RAM_BYTES);
        StringBuilder s = new StringBuilder("OK ORNITH15_PLAN/1");
        s.append(" context=").append(contextTokens);
        s.append(" ram_ceiling=").append(HARD_RAM_BYTES);
        s.append(" kv_budget=").append(kvBudget);
        s.append(" codec_q8q8_bpt=").append(codecBytesPerToken(Mode.Q8_Q8));
        s.append(" codec_q8q5_bpt=").append(codecBytesPerToken(Mode.Q8_Q5));
        s.append(" codec_q8q4_bpt=").append(codecBytesPerToken(Mode.Q8_Q4));

        for (Mode mode : Mode.values()) {
            s.append(" ").append(mode.name().toLowerCase())
                    .append("_bytes=").append(estimatedBytes(contextTokens, mode))
                    .append(" max_tokens=").append(maxTokens(kvBudget, mode));
        }

        // The planner prefers the least aggressive representation that fits.
        Mode selected = null;
        for (Mode mode : Mode.values()) {
            if (estimatedBytes(contextTokens, mode) <= kvBudget) {
                selected = mode;
                break;
            }
        }
        s.append(" selected=").append(selected == null ? "NONE" : selected.name());
        return s.toString();
    }
}
