package bslsjdk.mcnpu;

import java.io.*;
import java.net.InetAddress;
import java.net.Socket;
import java.net.InetSocketAddress;

public final class NpuServiceClient {
    private static final int IPC_PORT = 38761;
    /** Must match NpuService, which binds explicitly to 127.0.0.1. */
    private static final String IPC_HOST = "127.0.0.1";
    private static final int CONNECT_TIMEOUT_MS = 3000;
    private static final int READ_TIMEOUT_MS = 15000;
    private NpuServiceClient() {}

    public static String request(String command) {
        if (command == null || command.isEmpty()) return "ERR EMPTY_COMMAND";
        long t0 = System.nanoTime();
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(InetAddress.getByName(IPC_HOST), IPC_PORT), CONNECT_TIMEOUT_MS);
            // Long enough for a cold graph build on the largest shape: the first
            // n=16384 ADD pays graph creation, which measured ~150-230 ms, and a 3 s
            // ceiling turned a slow-but-valid first case into SERVICE_UNAVAILABLE.
            socket.setSoTimeout(READ_TIMEOUT_MS);
            BufferedWriter out = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()), 64 * 1024);
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()), 64 * 1024);
            out.write(command);
            out.write("\n");
            out.flush();
            String line = in.readLine();
            return line == null ? "ERR EMPTY_REPLY" : line;
        } catch (Throwable t) {
            // Keep the concrete exception type: it is the only way to tell
            // refused (not listening) from timeout (blocked) from reset.
            return "ERR SERVICE_UNAVAILABLE " + t.getClass().getName()
                    + " msg=" + String.valueOf(t.getMessage())
                    + " us=" + ((System.nanoTime() - t0) / 1000);
        }
    }

    public static boolean isAvailable() {
        return request("PING").startsWith("PONG MCNPU/");
    }

    public static String status() {
        return request("STATUS");
    }

    public static String smoke() {
        return request("SMOKE");
    }

    // Largest ADD length the service measured on the device. Starts at the
    // pre-probe default and is replaced by refreshMaxAddElements().
    private static volatile int cachedMaxAdd = 16384;

    /** Last known ceiling. Cheap: used per case during import validation. */
    public static int maxAddElements() { return cachedMaxAdd; }

    /**
     * Asks the service what its ADD probe measured, and caches the answer.
     *
     * Cached because it is read once per imported case: a round trip per case
     * would cost more than the validation it gates. Safe to call with the
     * service down - the cached value simply stays as it was.
     */
    public static synchronized int refreshMaxAddElements() {
        String r = request("CAPABILITIES");
        if (r != null) {
            int i = r.indexOf("max_elements=");
            if (i >= 0) {
                String tail = r.substring(i + 13).trim();
                int e = 0;
                while (e < tail.length() && Character.isDigit(tail.charAt(e))) e++;
                if (e > 0) {
                    try {
                        int v = Integer.parseInt(tail.substring(0, e));
                        if (v > 0) cachedMaxAdd = v;
                    } catch (NumberFormatException ignored) { /* keep the cached value */ }
                }
            }
        }
        return cachedMaxAdd;
    }

    /**
     * Sends ADD cases as raw little-endian float32 instead of decimal text.
     *
     * The body is written with a blocking loop: TCP is a stream, so a single
     * write is never assumed to carry the whole buffer, especially at 512 KB.
     */
    /** One BINADD call plus the float32 results it returned. */
    public static final class BinAddResult {
        public String status = "ERR NO_REPLY";
        public float[][] out;
        public long clientUs;
        public int okCount;
        public String raw = "";
    }

    /**
     * Sends ADD cases as raw little-endian float32 and reads the results back the
     * same way.
     *
     * The body is written and read with blocking loops: TCP is a stream, so no
     * single operation is assumed to carry the whole buffer, and the reply is a
     * text header followed by a binary body rather than one or the other, so a
     * failure is still readable in a log.
     */
    public static BinAddResult binAdd(float[][] a, float[][] b, int n) {
        BinAddResult res = new BinAddResult();
        if (a == null || b == null || a.length == 0 || a.length != b.length) {
            res.status = "ERR BINADD_ARGS";
            return res;
        }
        int cases = a.length;
        java.nio.ByteBuffer body = java.nio.ByteBuffer.allocate(4 * 2 * n * cases)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (int c = 0; c < cases; c++) {
            for (int i = 0; i < n; i++) body.putFloat(a[c][i]);
            for (int i = 0; i < n; i++) body.putFloat(b[c][i]);
        }
        byte[] payload = body.array();
        long t0 = System.nanoTime();
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(InetAddress.getByName(IPC_HOST), IPC_PORT), CONNECT_TIMEOUT_MS);
            socket.setSoTimeout(READ_TIMEOUT_MS);
            OutputStream os = new BufferedOutputStream(socket.getOutputStream(), 64 * 1024);
            InputStream is = new BufferedInputStream(socket.getInputStream(), 64 * 1024);
            os.write(("BINADD " + n + " " + cases + "\n").getBytes(java.nio.charset.StandardCharsets.UTF_8));
            int off = 0;
            while (off < payload.length) {
                int chunk = Math.min(64 * 1024, payload.length - off);
                os.write(payload, off, chunk);
                off += chunk;
            }
            os.flush();
            String header = readReplyLine(is);
            res.raw = header == null ? "" : header;
            if (header == null) {
                res.status = "ERR EMPTY_REPLY";
                return res;
            }
            res.status = header;
            int at = header.indexOf("ok=");
            if (at >= 0) {
                int slash = header.indexOf('/', at);
                if (slash > at) {
                    try { res.okCount = Integer.parseInt(header.substring(at + 3, slash).trim()); }
                    catch (NumberFormatException ignored) {}
                }
            }
            if (header.startsWith("OK")) {
                byte[] results = new byte[4 * n * cases];
                readFully(is, results, results.length);
                java.nio.ByteBuffer rb = java.nio.ByteBuffer.wrap(results)
                        .order(java.nio.ByteOrder.LITTLE_ENDIAN);
                res.out = new float[cases][n];
                for (int c = 0; c < cases; c++) for (int i = 0; i < n; i++) res.out[c][i] = rb.getFloat();
            }
            return res;
        } catch (Throwable t) {
            res.status = "ERR SERVICE_UNAVAILABLE " + t.getClass().getName()
                    + " msg=" + String.valueOf(t.getMessage());
            return res;
        } finally {
            res.clientUs = (System.nanoTime() - t0) / 1000L;
        }
    }

    private static void readFully(InputStream in, byte[] buf, int len) throws IOException {
        int off = 0;
        while (off < len) {
            int r = in.read(buf, off, len - off);
            if (r < 0) throw new java.io.EOFException("expected " + len + " bytes, got " + off);
            off += r;
        }
    }

    private static String readReplyLine(InputStream in) throws IOException {
        java.io.ByteArrayOutputStream buf = new java.io.ByteArrayOutputStream(256);
        int ch;
        while ((ch = in.read()) >= 0) {
            if (ch == '\n') return new String(buf.toByteArray(), java.nio.charset.StandardCharsets.UTF_8);
            if (ch != '\r') buf.write(ch);
            if (buf.size() > 65536) throw new IOException("reply line too long");
        }
        return buf.size() == 0 ? null : new String(buf.toByteArray(), java.nio.charset.StandardCharsets.UTF_8);
    }
}
