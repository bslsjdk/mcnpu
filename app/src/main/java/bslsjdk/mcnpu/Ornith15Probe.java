package bslsjdk.mcnpu;

import java.io.EOFException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/** Metadata-only Ornith-1.5-9B GGUF probe. Never loads model weights. */
public final class Ornith15Probe {
    private Ornith15Probe() {}

    public static String inspect(String path, long contextTokens) {
        if (path == null || path.isEmpty()) return "ERR ORNITH15 null_path";
        if (contextTokens <= 0 || contextTokens > 262144)
            return "ERR ORNITH15 context=" + contextTokens;

        try (RandomAccessFile f = new RandomAccessFile(path, "r")) {
            byte[] magic = new byte[4];
            f.readFully(magic);
            if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F')
                return "ERR ORNITH15 bad_magic";

            long version = u32(f);
            if (version != 2 && version != 3) return "ERR ORNITH15 version=" + version;
            long tensorCount = u64(f);
            long kvCount = u64(f);
            if (tensorCount <= 0 || tensorCount > 200000 || kvCount > 100000)
                return "ERR ORNITH15 counts";

            Map<String, Object> meta = new HashMap<>();
            long alignment = 32;
            for (long i = 0; i < kvCount; i++) {
                String key = string(f);
                int type = (int) u32(f);
                Object value = readValue(f, type, 0);
                meta.put(key, value);
                if ("general.alignment".equals(key) && value instanceof Number)
                    alignment = ((Number) value).longValue();
            }

            long blockCount = number(meta, "qwen35.block_count", "qwen35.text.block_count");
            long hidden = number(meta, "qwen35.embedding_length", "qwen35.text.embedding_length");
            long vocab = number(meta, "qwen35.vocab_size");
            long context = number(meta, "qwen35.context_length", "qwen35.text.context_length");
            String arch = text(meta, "general.architecture");

            if (blockCount <= 0) blockCount = 32;
            if (hidden <= 0) hidden = 4096;
            if (vocab <= 0) vocab = 248320;
            if (context <= 0) context = 262144;

            long fullAttentionLayers = blockCount / 4;
            if (blockCount % 4 != 0)
                return "ERR ORNITH15 unexpected_layer_interval block_count=" + blockCount;
            long linearLayers = blockCount - fullAttentionLayers;

            Map<Long, Long> types = new HashMap<>();
            for (long i = 0; i < tensorCount; i++) {
                string(f);
                long dims = u32(f);
                if (dims <= 0 || dims > 8) return "ERR ORNITH15 dims";
                for (long d = 0; d < dims; d++) {
                    long n = u64(f);
                    if (n <= 0) return "ERR ORNITH15 shape";
                }
                long type = u32(f);
                u64(f);
                types.put(type, types.getOrDefault(type, 0L) + 1L);
            }

            long kvBytesPerTokenFp16 = fullAttentionLayers * 2L * 4L * 256L * 2L;
            long kvBytesAtContext = safeMul(kvBytesPerTokenFp16, contextTokens);
            long kvBytesAt64k = safeMul(kvBytesPerTokenFp16, 65536L);
            long kvQ8At64k = safeMul(kvBytesPerTokenFp16 / 2L, 65536L);

            StringBuilder out = new StringBuilder();
            out.append("OK ORNITH15_PROBE/1");
            out.append(" arch=").append(arch == null ? "unknown" : arch);
            out.append(" layers=").append(blockCount);
            out.append(" full_attn=").append(fullAttentionLayers);
            out.append(" linear_attn=").append(linearLayers);
            out.append(" hidden=").append(hidden);
            out.append(" vocab=").append(vocab);
            out.append(" native_context=").append(context);
            out.append(" requested_context=").append(contextTokens);
            out.append(" kv_fp16_bytes_per_token=").append(kvBytesPerTokenFp16);
            out.append(" kv_fp16_requested=").append(kvBytesAtContext);
            out.append(" kv_fp16_64k=").append(kvBytesAt64k);
            out.append(" kv_q8_64k_est=").append(kvQ8At64k);
            out.append(" tensor_types=").append(types);
            out.append(" file_bytes=").append(f.length());
            return out.toString();
        } catch (EOFException e) {
            return "ERR ORNITH15 truncated";
        } catch (Throwable t) {
            return "ERR ORNITH15 " + t.getClass().getSimpleName() + ": " + t.getMessage();
        }
    }

    private static long safeMul(long a, long b) {
        if (a <= 0 || b <= 0 || a > Long.MAX_VALUE / b) return Long.MAX_VALUE;
        return a * b;
    }

    private static long number(Map<String, Object> m, String... keys) {
        for (String k : keys) {
            Object v = m.get(k);
            if (v instanceof Number) return ((Number) v).longValue();
        }
        return -1;
    }

    private static String text(Map<String, Object> m, String key) {
        Object v = m.get(key);
        return v instanceof String ? (String) v : null;
    }

    private static Object readValue(RandomAccessFile f, int type, int depth) throws IOException {
        if (depth > 4) throw new IOException("array depth");
        switch (type) {
            case 0: return (long) f.readUnsignedByte();
            case 1: return (long) f.readByte();
            case 2: return readU16(f);
            case 3: return (long) (short) readU16(f);
            case 4: return u32(f);
            case 5: return (long) (int) u32(f);
            case 6: return f.readFloat();
            case 7: return f.readUnsignedByte() != 0;
            case 8: return string(f);
            case 9:
                int elem = (int) u32(f);
                long n = u64(f);
                if (n < 0 || n > 1000000) throw new IOException("array size");
                ArrayList<Object> a = new ArrayList<>((int) n);
                for (long i = 0; i < n; i++) a.add(readValue(f, elem, depth + 1));
                return a;
            case 10: return u64(f);
            case 11: return f.readLong();
            case 12: return f.readDouble();
            default: throw new IOException("unknown type=" + type);
        }
    }

    private static long readU16(RandomAccessFile f) throws IOException {
        return f.readUnsignedByte() | ((long) f.readUnsignedByte() << 8);
    }

    private static long u32(RandomAccessFile f) throws IOException {
        long a = f.readUnsignedByte(), b = f.readUnsignedByte();
        long c = f.readUnsignedByte(), d = f.readUnsignedByte();
        return a | (b << 8) | (c << 16) | (d << 24);
    }

    private static long u64(RandomAccessFile f) throws IOException {
        long v = 0;
        for (int i = 0; i < 8; i++) v |= ((long) f.readUnsignedByte()) << (8 * i);
        return v;
    }

    private static String string(RandomAccessFile f) throws IOException {
        long n = u64(f);
        if (n < 0 || n > (1L << 20)) throw new IOException("string too large");
        byte[] b = new byte[(int) n];
        f.readFully(b);
        return new String(b, StandardCharsets.UTF_8);
    }
}
