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
 * Three limits worth knowing before choosing wayElements:
 *
 * - The HTP graph is per size, and the accepted sizes top out at 16384. A way of
 *   884736 elements (nine chunks) does not exist as a shape and will be rejected;
 *   that is why the default way is 16384 and why total 7962624 becomes 486 ways.
 * - The service serialises execute behind one lock, so "parallel" here overlaps
 *   encode, transfer and decode with execute; it does not multiply the device.
 * - One IPC call must stay small. 486 ways of 16384 in a single call is a 63 MB
 *   body, which the service would have to buffer in one allocation. Ways are
 *   therefore submitted in sub-batches bounded by TARGET_BODY_BYTES, so the body
 *   size is decided here rather than discovered as an OOM on the device.
 */
public final class NpuBigAdd {

    /** Largest element count the ADD graph accepts. */
    public static final int MAX_WAY = 16384;

    /**
     * Body size one IPC call aims for. At n=16384 this is 16 cases (2 MB); at
     * smaller n the case count rises so the cost per call stays roughly flat
     * instead of being dominated by round trips.
     */
    private static final long TARGET_BODY_BYTES = 2L * 1024 * 1024;

    public static final class Result {
        public final float[] out;
        public final int ways;
        public final int wayElements;
        public final int okWays;
        public final long totalUs;
        public final int calls;
        public final String status;
        Result(float[] out, int ways, int wayElements, int okWays, long totalUs, int calls, String status) {
            this.out = out; this.ways = ways; this.wayElements = wayElements;
            this.okWays = okWays; this.totalUs = totalUs; this.calls = calls;
            this.status = status;
        }
    }

    private NpuBigAdd() {}

    /** Largest case count one call may carry for this way size, body-size bound. */
    public static int maxCasesPerCall(int wayElements) {
        long perCase = 8L * wayElements;          // two float32 arrays
        if (perCase <= 0) return 1;
        long c = TARGET_BODY_BYTES / perCase;
        return (int) Math.max(1, Math.min(64, c));
    }

    /**
     * @param parallelism socket count. More sockets overlap transfer with execute;
     *                    they do not add execution throughput.
     */
    public static Result add(float[] a, float[] b, int wayElements, int parallelism) {
        long t0 = System.nanoTime();
        if (a == null || b == null || a.length != b.length) {
            return new Result(null, 0, wayElements, 0, 0, 0, "ERR BIGADD_ARGS");
        }
        int n = wayElements <= 0 ? MAX_WAY : Math.min(wayElements, MAX_WAY);
        int p = Math.max(1, Math.min(parallelism <= 0 ? 4 : parallelism, 16));
        int total = a.length;
        int ways = (total + n - 1) / n;
        float[] out = new float[total];
        AtomicInteger okWays = new AtomicInteger();
        AtomicInteger failures = new AtomicInteger();
        AtomicInteger callCount = new AtomicInteger();
        StringBuilder firstError = new StringBuilder();

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
                    int sub = maxCasesPerCall(n);
                    int w = 0;
                    while (w < gw) {
                        int count = Math.min(sub, gw - w);
                        float[][] aa = new float[count][n];
                        float[][] bb = new float[count][n];
                        for (int j = 0; j < count; j++) {
                            int idx = start + w + j;
                            int off = idx * n;
                            int len = Math.min(n, total - off);
                            // Fresh arrays are already zero, so a short tail is
                            // correct padding: x + 0 = x and out is untouched.
                            if (len > 0) {
                                System.arraycopy(a, off, aa[j], 0, len);
                                System.arraycopy(b, off, bb[j], 0, len);
                            }
                        }
                        callCount.incrementAndGet();
                        NpuServiceClient.BinAddResult r = NpuServiceClient.binAdd(aa, bb, n);
                        okWays.addAndGet(r.okCount);
                        if (!r.status.startsWith("OK")) {
                            failures.incrementAndGet();
                            synchronized (firstError) {
                                if (firstError.length() == 0) firstError.append(r.status);
                            }
                        } else if (r.out != null) {
                            for (int j = 0; j < count; j++) {
                                int idx = start + w + j;
                                int off = idx * n;
                                int len = Math.min(n, total - off);
                                if (len > 0) System.arraycopy(r.out[j], 0, out, off, len);
                            }
                        }
                        w += count;
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
                  + " calls=" + callCount.get() + " ok=" + okWays.get() + "/" + ways + " us=" + us
                : "ERR BIGADD failed_groups=" + failures.get() + " first=" + firstError;
        return new Result(out, ways, n, okWays.get(), us, callCount.get(), status);
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
