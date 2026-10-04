package bslsjdk.mcnpu;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * One logical element-wise add over arrays of any length, split into fixed-shape
 * ways that run over parallel sockets and are merged back into a single output.
 *
 * The caller sees one call and one result; the split, the round trips and the
 * merge are internal. The way shape is fixed on purpose: one shape means the
 * service builds its graph once and every later way reuses it, and it means the
 * merge is a plain offset copy rather than a shape negotiation.
 *
 * Two limits worth knowing before choosing wayElements:
 *
 * - The HTP graph is per size, and the accepted sizes top out at 16384. A way of
 *   884736 elements (nine chunks) does not exist as a shape and will be rejected;
 *   that is why the default way is 16384 and why total 7962624 becomes 486 ways.
 * - The service serialises execute behind one lock, so "parallel" here overlaps
 *   encode, transfer and decode with execute; it does not multiply the device.
 */
public final class NpuBigAdd {

    /** Largest element count the ADD graph accepts. */
    public static final int MAX_WAY = 16384;

    public static final class Result {
        public final float[] out;
        public final int ways;
        public final int wayElements;
        public final int okWays;
        public final long totalUs;
        public final String status;
        Result(float[] out, int ways, int wayElements, int okWays, long totalUs, String status) {
            this.out = out; this.ways = ways; this.wayElements = wayElements;
            this.okWays = okWays; this.totalUs = totalUs; this.status = status;
        }
    }

    private NpuBigAdd() {}

    /**
     * @param parallelism socket count. More sockets overlap transfer with execute;
     *                    they do not add execution throughput.
     */
    public static Result add(float[] a, float[] b, int wayElements, int parallelism) {
        long t0 = System.nanoTime();
        if (a == null || b == null || a.length != b.length) {
            return new Result(null, 0, wayElements, 0, 0, "ERR BIGADD_ARGS");
        }
        int n = wayElements <= 0 ? MAX_WAY : Math.min(wayElements, MAX_WAY);
        int p = Math.max(1, Math.min(parallelism <= 0 ? 4 : parallelism, 16));
        int total = a.length;
        int ways = (total + n - 1) / n;
        float[] out = new float[total];
        AtomicInteger okWays = new AtomicInteger();
        AtomicInteger failures = new AtomicInteger();
        StringBuilder firstError = new StringBuilder();

        // Contiguous groups, one socket each, one BINADD per group. Batching a
        // group into a single call with case_count = group size is what keeps the
        // graph cache hit rate high: every case in the call has the same shape.
        int groups = Math.min(p, ways);
        CountDownLatch latch = new CountDownLatch(groups);
        int base = ways / groups;
        int extra = ways % groups;
        int cursor = 0;
        for (int g = 0; g < groups; g++) {
            int groupWays = base + (g < extra ? 1 : 0);
            final int start = cursor;
            cursor += groupWays;
            final int gw = groupWays;
            new Thread(() -> {
                try {
                    float[][] aa = new float[gw][n];
                    float[][] bb = new float[gw][n];
                    for (int w = 0; w < gw; w++) {
                        int off = (start + w) * n;
                        int len = Math.min(n, total - off);
                        if (len <= 0) continue;
                        // Both arrays pad with 0, so x + 0 = x and the tail can
                        // simply be dropped on merge.
                        System.arraycopy(a, off, aa[w], 0, len);
                        System.arraycopy(b, off, bb[w], 0, len);
                    }
                    NpuServiceClient.BinAddResult r = NpuServiceClient.binAdd(aa, bb, n);
                    okWays.addAndGet(r.okCount);
                    if (!r.status.startsWith("OK")) {
                        failures.incrementAndGet();
                        synchronized (firstError) {
                            if (firstError.length() == 0) firstError.append(r.status);
                        }
                    } else if (r.out != null) {
                        for (int w = 0; w < gw; w++) {
                            int off = (start + w) * n;
                            int len = Math.min(n, total - off);
                            if (len <= 0) continue;
                            System.arraycopy(r.out[w], 0, out, off, len);
                        }
                    }
                } catch (Throwable t) {
                    failures.incrementAndGet();
                    synchronized (firstError) {
                        if (firstError.length() == 0) firstError.append(String.valueOf(t.getMessage()));
                    }
                } finally {
                    latch.countDown();
                }
            }, "npu-bigadd-" + g).start();
        }
        try {
            latch.await();
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        }
        long us = (System.nanoTime() - t0) / 1000L;
        String status = failures.get() == 0
                ? "OK BIGADD total=" + total + " ways=" + ways + " n=" + n + " sockets=" + groups
                  + " ok=" + okWays.get() + "/" + ways + " us=" + us
                : "ERR BIGADD failed_groups=" + failures.get() + " first=" + firstError;
        return new Result(out, ways, n, okWays.get(), us, status);
    }

    /** CPU reference used by the self test; the NPU result is compared against it. */
    public static float[] cpuAdd(float[] a, float[] b) {
        float[] out = new float[a.length];
        for (int i = 0; i < a.length; i++) out[i] = a[i] + b[i];
        return out;
    }

    /**
     * Runs one split add and checks the merged output against the CPU element by
     * element. Reports the worst divergence instead of a pass flag, so a merge
     * that is off by one way is visible as a magnitude, not as a boolean.
     */
    public static String selfTest(int total, int wayElements, int parallelism) {
        float[] a = new float[total], b = new float[total];
        for (int i = 0; i < total; i++) {
            a[i] = ((i % 97) - 48) / 32.0f;
            b[i] = ((i % 53) + 1) / 64.0f;
        }
        long c0 = System.nanoTime();
        float[] want = cpuAdd(a, b);
        long cpuUs = (System.nanoTime() - c0) / 1000L;
        Result r = add(a, b, wayElements, parallelism);
        if (r.out == null) return "BIGADD SELFTEST " + r.status;
        double maxAbs = 0;
        int bad = 0;
        for (int i = 0; i < total; i++) {
            double d = Math.abs(r.out[i] - want[i]);
            if (d > 1e-3) bad++;
            if (d > maxAbs) maxAbs = d;
        }
        return "BIGADD SELFTEST total=" + total + " ways=" + r.ways + " n=" + r.wayElements
                + " " + r.status + " cpu_us=" + cpuUs + " max_abs=" + maxAbs + " mismatch=" + bad;
    }
}
