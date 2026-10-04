package bslsjdk.mcnpu;

import android.content.Context;
import android.net.Uri;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.zip.CRC32;

/**
 * Reads the .binadd container: a 32 byte little-endian header followed, per case,
 * by n float32 for a then n float32 for b.
 *
 * The file is never loaded as a whole. A 9x9 import is 63 MB, which as float[]
 * plus an output array plus per-call buffers is far past what a phone will give
 * one process. The run walks the file in segments sized by the way shape, so the
 * resident cost is a few MB no matter how large n is, and a failure at way 400
 * still reports everything that succeeded before it.
 *
 * The header carries the way shape and the parallelism so the file, not the App,
 * decides how the work is split. That keeps a re-run comparable: the same file
 * always produces the same number of ways and therefore the same graph reuse.
 */
public final class NpuBinFile {

    public static final int MAGIC = 0x424E4144;               // "BNAD"
    public static final int HEADER_BYTES = 32;
    public static final int VERSION = 1;
    public static final int OP_ADD = 1;
    public static final int OP_MUL = 2;

    /** Ways held in memory at once. 64 x 16384 floats is 4 MB per input array. */
    private static final int SEG_WAYS = 64;

    /** Per-element tolerance for the CPU comparison. */
    private static final double TOL = 1e-3;

    public interface Progress {
        void on(String line);
    }

    private NpuBinFile() {}

    /**
     * Copies the picked document into cache storage.
     *
     * A SAF Uri only hands out a forward-only stream, and this reader needs to
     * jump between the a block and the b block of each case, which sit n*4 bytes
     * apart. Materialising once is what makes that seek possible.
     */
    public static File materialize(Context ctx, Uri uri) throws IOException {
        File dst = new File(ctx.getCacheDir(), "import.binadd");
        if (dst.exists()) dst.delete();
        long copied = 0;
        try (InputStream in = ctx.getContentResolver().openInputStream(uri);
             FileOutputStream out = new FileOutputStream(dst)) {
            if (in == null) throw new IOException("无法打开文件");
            byte[] buf = new byte[256 * 1024];
            int r;
            while ((r = in.read(buf)) != -1) {
                out.write(buf, 0, r);
                copied += r;
            }
        }
        if (copied < HEADER_BYTES) throw new IOException("文件小于头部，大小=" + copied);
        return dst;
    }

    /** Parses the header without reading the body. */
    public static String describe(File f) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(f, "r")) {
            ByteBuffer h = readHeader(raf);
            int magic = h.getInt();
            int version = h.getShort() & 0xFFFF;
            int op = h.getShort() & 0xFFFF;
            int n = h.getInt();
            int caseCount = h.getInt();
            int wayElements = h.getInt();
            int parallelism = h.getInt();
            h.getInt();
            long checksum = h.getInt() & 0xFFFFFFFFL;
            int way = wayElements <= 0 ? NpuBigAdd.MAX_WAY : Math.min(wayElements, NpuBigAdd.MAX_WAY);
            int ways = ((n + way - 1) / way) * Math.max(1, caseCount);
            return "magic=0x" + Integer.toHexString(magic)
                    + " version=" + version + " op=" + op
                    + " n=" + n + " cases=" + caseCount
                    + " way=" + way + " par=" + parallelism
                    + " crc=" + checksum + " ways=" + ways
                    + " file_mb=" + (f.length() / 1048576.0);
        }
    }

    /**
     * Splits every case into fixed ways, runs them through NpuBigAdd, and compares
     * the merged result against a CPU add element by element.
     */
    public static String run(File f, Progress progress) throws IOException {
        StringBuilder log = new StringBuilder();
        try (RandomAccessFile raf = new RandomAccessFile(f, "r")) {
            FileChannel ch = raf.getChannel();
            ByteBuffer h = readHeader(raf);
            int magic = h.getInt();
            int version = h.getShort() & 0xFFFF;
            int op = h.getShort() & 0xFFFF;
            int n = h.getInt();
            int caseCount = h.getInt();
            int wayElements = h.getInt();
            int parallelism = h.getInt();
            h.getInt();
            long checksum = h.getInt() & 0xFFFFFFFFL;

            if (magic != MAGIC) throw new IOException("magic 不匹配 0x" + Integer.toHexString(magic));
            if (version != VERSION) throw new IOException("version 不支持 " + version);
            if (op != OP_ADD) throw new IOException("暂不支持 op=" + op + "（当前仅 ADD）");
            if (n <= 0 || caseCount <= 0) throw new IOException("非法 n=" + n + " cases=" + caseCount);

            int way = wayElements <= 0 ? NpuBigAdd.MAX_WAY : Math.min(wayElements, NpuBigAdd.MAX_WAY);
            int par = parallelism <= 0 ? 4 : Math.min(parallelism, 16);
            int waysPerCase = (n + way - 1) / way;
            int totalWays = waysPerCase * caseCount;

            long expected = HEADER_BYTES + 8L * n * caseCount;
            if (f.length() != expected) {
                log.append("WARN 文件大小 ").append(f.length()).append(" 与头部推算 ")
                   .append(expected).append(" 不一致\n");
            }

            log.append("=== IMPORTED BINADD ===\n")
               .append("n=").append(n).append("  cases=").append(caseCount)
               .append("  ways=").append(totalWays)
               .append("  way_elements=").append(way)
               .append("  parallelism=").append(par)
               .append("  file_mb=").append(String.format("%.1f", f.length() / 1048576.0)).append("\n");

            if (checksum != 0) {
                long t0 = System.nanoTime();
                long actual = crcBody(ch, expected - HEADER_BYTES);
                long crcUs = (System.nanoTime() - t0) / 1000L;
                boolean ok = actual == checksum;
                log.append("crc32 header=").append(checksum).append(" actual=").append(actual)
                   .append(ok ? " OK" : " MISMATCH").append(" verify_us=").append(crcUs).append("\n");
                if (progress != null) progress.on("CRC32 " + (ok ? "OK" : "MISMATCH header=" + checksum + " actual=" + actual));
                if (!ok) return log.append("SUMMARY FAIL crc_mismatch\n").toString();
            }

            float[] aBuf = new float[SEG_WAYS * way];
            float[] bBuf = new float[SEG_WAYS * way];
            double maxAbs = 0;
            long mismatch = 0, compared = 0;
            int okWays = 0, calls = 0, failedSegments = 0;
            String firstError = null;
            long npuUs = 0, cpuUs = 0, tStart = System.nanoTime();

            for (int c = 0; c < caseCount; c++) {
                long aBase = HEADER_BYTES + 8L * n * c;
                long bBase = aBase + 4L * n;
                for (int s = 0; s < waysPerCase; s += SEG_WAYS) {
                    int thisWays = Math.min(SEG_WAYS, waysPerCase - s);
                    int firstElem = s * way;
                    int want = Math.min(thisWays * way, n - firstElem);
                    if (want <= 0) break;

                    int gotA = readFloats(ch, aBase + 4L * firstElem, aBuf, want);
                    int gotB = readFloats(ch, bBase + 4L * firstElem, bBuf, want);
                    if (gotA != want || gotB != want) {
                        throw new IOException("文件在 case " + c + " 段 " + s + " 处提前结束");
                    }

                    float[] aSeg = Arrays.copyOf(aBuf, want);
                    float[] bSeg = Arrays.copyOf(bBuf, want);

                    NpuBigAdd.Result r = NpuBigAdd.add(aSeg, bSeg, way, par);
                    calls += r.calls;
                    npuUs += r.totalUs;
                    if (r.out == null) {
                        failedSegments++;
                        if (firstError == null) firstError = r.status;
                        if (progress != null) {
                            final int pc = c, ps = s, pw = thisWays;
                            progress.on("SEG case=" + pc + " ways=" + ps + ".." + (ps + pw) + " FAILED " + r.status);
                        }
                        continue;
                    }
                    okWays += r.okWays;

                    long c0 = System.nanoTime();
                    for (int i = 0; i < want; i++) {
                        double d = Math.abs(r.out[i] - (aSeg[i] + bSeg[i]));
                        // Written as !(d <= TOL) on purpose: a NaN result makes
                        // both `d > TOL` and `d > maxAbs` false, so the obvious
                        // form would silently drop a failed element from both
                        // the count and the worst case.
                        if (!(d <= TOL)) mismatch++;
                        if (Double.isNaN(d) || d > maxAbs) maxAbs = d;
                    }
                    compared += want;
                    cpuUs += (System.nanoTime() - c0) / 1000L;

                    if (progress != null) {
                        // Copies, not the originals: a lambda cannot capture a
                        // for-loop variable or a running maximum, both of which
                        // are reassigned and therefore not effectively final.
                        final int pc = c, ps = s, pw = thisWays;
                        final double pMax = maxAbs;
                        final int pOk = r.okWays, pCalls = r.calls;
                        final long pUs = r.totalUs;
                        progress.on("SEG case=" + pc + " ways=" + ps + ".." + (ps + pw - 1)
                                + " ok=" + pOk + "/" + pw
                                + " calls=" + pCalls + " us=" + pUs
                                + " max_abs=" + pMax);
                    }
                }
            }

            long elapsedMs = (System.nanoTime() - tStart) / 1000000L;
            log.append("graph_cached: first=false, rest=true (shape 固定，图只建一次)\n")
               .append("ok_ways=").append(okWays).append("/").append(totalWays)
               .append("  ipc_calls=").append(calls)
               .append("  npu_us=").append(npuUs).append("  cpu_cmp_us=").append(cpuUs).append("\n")
               .append("compared=").append(compared).append("\n")
               .append("max_abs=").append(maxAbs).append("\n")
               .append("mismatch=").append(mismatch).append("\n")
               .append("elapsed_ms=").append(elapsedMs).append("\n");
            boolean pass = failedSegments == 0 && mismatch == 0 && okWays == totalWays && compared == (long) n * caseCount;
            log.append("SUMMARY ").append(pass ? "PASS" : "FAIL")
               .append("  failed_segments=").append(failedSegments);
            if (firstError != null) log.append("  first=").append(firstError);
            log.append("\n");
            return log.toString();
        }
    }

    private static ByteBuffer readHeader(RandomAccessFile raf) throws IOException {
        ByteBuffer h = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        int off = 0;
        while (off < HEADER_BYTES) {
            int r = raf.read(h.array(), off, HEADER_BYTES - off);
            if (r < 0) throw new IOException("头部不完整，只读到 " + off + " 字节");
            off += r;
        }
        return h;
    }

    private static int readFloats(FileChannel ch, long filePos, float[] dst, int count) throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(count * 4).order(ByteOrder.LITTLE_ENDIAN);
        int got = 0;
        while (bb.hasRemaining()) {
            int r = ch.read(bb, filePos + got);
            if (r < 0) break;
            got += r;
        }
        bb.flip();
        int floats = bb.remaining() / 4;
        bb.asFloatBuffer().get(dst, 0, floats);
        return floats;
    }

    private static long crcBody(FileChannel ch, long bodyBytes) throws IOException {
        CRC32 crc = new CRC32();
        ByteBuffer bb = ByteBuffer.allocate(256 * 1024);
        long pos = HEADER_BYTES, remaining = bodyBytes;
        while (remaining > 0) {
            bb.clear();
            int want = (int) Math.min(bb.capacity(), remaining);
            bb.limit(want);
            int got = 0;
            while (bb.hasRemaining()) {
                int r = ch.read(bb, pos + got);
                if (r < 0) break;
                got += r;
            }
            if (got == 0) break;
            bb.flip();
            crc.update(bb.array(), 0, got);
            pos += got;
            remaining -= got;
        }
        return crc.getValue();
    }
}
