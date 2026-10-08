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

    /** Validates GGUF header/metadata/tensor descriptors only; no model weights are loaded. */
    public static String validateGguf(String path) {
        if (path == null || path.isEmpty()) return "ERR BONSAI2_PQ2_VALIDATE null_path";
        try (java.io.RandomAccessFile f = new java.io.RandomAccessFile(path, "r")) {
            byte[] magic = new byte[4];
            f.readFully(magic);
            if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F')
                return "ERR BONSAI2_PQ2_VALIDATE bad_magic";
            long version = u32(f);
            if (version != 2 && version != 3) return "ERR BONSAI2_PQ2_VALIDATE version=" + version;
            long tensorCount = u64(f);
            long kvCount = u64(f);
            if (tensorCount <= 0 || tensorCount > 200000 || kvCount > 100000)
                return "ERR BONSAI2_PQ2_VALIDATE counts";

            String arch = null, transform = null, axis = null, signMode = null;
            long block = 0, hadVersion = 0;
            boolean gdn = false, haveGdn = false;
            java.util.List<String> names = null, inverse = null;
            java.util.List<Long> widths = null, signs = null;

            for (long i = 0; i < kvCount; i++) {
                String key = string(f);
                int type = (int)u32(f);
                switch (key) {
                    case "general.architecture":
                        arch = type == 8 ? string(f) : skipValue(f, type) ? null : null;
                        if (type != 8) return "ERR BONSAI2_PQ2_VALIDATE architecture_type";
                        break;
                    case "prism.hadamard.version":
                        if (type != 4) return "ERR BONSAI2_PQ2_VALIDATE hadamard_version_type";
                        hadVersion = u32(f);
                        break;
                    case "prism.hadamard.block_size":
                        if (type != 4) return "ERR BONSAI2_PQ2_VALIDATE block_type";
                        block = u32(f);
                        break;
                    case "prism.hadamard.transform":
                        if (type != 8) return "ERR BONSAI2_PQ2_VALIDATE transform_type";
                        transform = string(f);
                        break;
                    case "prism.hadamard.axis":
                        if (type != 8) return "ERR BONSAI2_PQ2_VALIDATE axis_type";
                        axis = string(f);
                        break;
                    case "prism.hadamard.sign_mode":
                        if (type != 8) return "ERR BONSAI2_PQ2_VALIDATE sign_mode_type";
                        signMode = string(f);
                        break;
                    case "prism.hadamard.weight_names":
                        names = stringArray(f, type);
                        break;
                    case "prism.hadamard.inverse_weight_names":
                        inverse = stringArray(f, type);
                        break;
                    case "prism.hadamard.sign_widths":
                        widths = integerArray(f, type);
                        break;
                    case "prism.hadamard.sign_values":
                        signs = integerArray(f, type);
                        break;
                    case "prism.hadamard.gdn_v_grouped":
                        if (type != 7) return "ERR BONSAI2_PQ2_VALIDATE gdn_type";
                        gdn = f.readUnsignedByte() != 0;
                        haveGdn = true;
                        break;
                    default:
                        if (!skipValue(f, type)) return "ERR BONSAI2_PQ2_VALIDATE metadata";
                }
            }

            if (!"qwen35".equals(arch) || hadVersion != 1 || block != 1024 ||
                    !"normalized-sylvester-walsh-hadamard".equals(transform) ||
                    !"input-last-dimension".equals(axis) || !"explicit".equals(signMode) ||
                    names == null || names.isEmpty() || inverse == null || inverse.size() != 1 ||
                    !"token_embd.weight".equals(inverse.get(0)) || !haveGdn || !gdn ||
                    widths == null || widths.size() != 3 || signs == null || signs.size() != 28672)
                return "ERR BONSAI2_PQ2_VALIDATE hadamard_metadata";

            long sum = 0;
            for (long w : widths) {
                if (w <= 0 || w % 1024 != 0) return "ERR BONSAI2_PQ2_VALIDATE sign_width";
                sum += w;
            }
            if (sum != signs.size()) return "ERR BONSAI2_PQ2_VALIDATE sign_sum";
            for (long s : signs) if (s != 1 && s != -1) return "ERR BONSAI2_PQ2_VALIDATE sign_value";

            long pq2 = 0, ptq1 = 0;
            for (long i = 0; i < tensorCount; i++) {
                string(f);
                long dims = u32(f);
                if (dims <= 0 || dims > 8) return "ERR BONSAI2_PQ2_VALIDATE dims";
                long first = 0;
                for (long d = 0; d < dims; d++) {
                    long n = u64(f);
                    if (n <= 0) return "ERR BONSAI2_PQ2_VALIDATE shape";
                    if (d == 0) first = n;
                }
                long type = u32(f);
                u64(f); // tensor data offset, not dereferenced
                if (type == 142) {
                    pq2++;
                    if (first % 128 != 0) return "ERR BONSAI2_PQ2_VALIDATE pq2_shape";
                } else if (type == 143) {
                    ptq1++;
                }
            }

            return pq2 > 0
                    ? "OK BONSAI2_PQ2_VALIDATE/1 type=142 group=128 block=34 hadamard=validated tensors=" + pq2
                    : "ERR BONSAI2_PQ2_VALIDATE no_type_142";
        } catch (java.io.EOFException e) {
            return "ERR BONSAI2_PQ2_VALIDATE truncated";
        } catch (Throwable t) {
            return "ERR BONSAI2_PQ2_VALIDATE " + t.getClass().getSimpleName() + ": " + t.getMessage();
        }
    }

    private static long u32(java.io.RandomAccessFile f) throws java.io.IOException {
        long a=f.readUnsignedByte(), b=f.readUnsignedByte(), c=f.readUnsignedByte(), d=f.readUnsignedByte();
        return a | (b<<8) | (c<<16) | (d<<24);
    }

    private static long u64(java.io.RandomAccessFile f) throws java.io.IOException {
        long v=0;
        for (int i=0;i<8;i++) v |= ((long)f.readUnsignedByte()) << (8*i);
        return v;
    }

    private static String string(java.io.RandomAccessFile f) throws java.io.IOException {
        long n=u64(f);
        if (n < 0 || n > (1L<<20)) throw new java.io.IOException("string too large");
        byte[] b=new byte[(int)n];
        f.readFully(b);
        return new String(b, java.nio.charset.StandardCharsets.UTF_8);
    }

    private static java.util.List<String> stringArray(java.io.RandomAccessFile f, int type) throws java.io.IOException {
        if (type != 9) throw new java.io.IOException("array type");
        int elem=(int)u32(f); long n=u64(f);
        if (elem != 8 || n > 10000) throw new java.io.IOException("string array");
        java.util.ArrayList<String> out=new java.util.ArrayList<>((int)n);
        for(long i=0;i<n;i++) out.add(string(f));
        return out;
    }

    private static java.util.List<Long> integerArray(java.io.RandomAccessFile f, int type) throws java.io.IOException {
        if (type != 9) throw new java.io.IOException("array type");
        int elem=(int)u32(f); long n=u64(f);
        if (n > 100000) throw new java.io.IOException("integer array");
        java.util.ArrayList<Long> out=new java.util.ArrayList<>((int)n);
        for(long i=0;i<n;i++) {
            switch(elem) {
                case 0: case 1: out.add((long)f.readByte()); break;
                case 4: case 5: out.add(u32(f)); break;
                case 10: case 11: out.add(u64(f)); break;
                default: throw new java.io.IOException("integer element type");
            }
        }
        return out;
    }

    private static boolean skipValue(java.io.RandomAccessFile f, int type) throws java.io.IOException {
        switch(type) {
            case 0: case 1: f.skipBytes(1); return true;
            case 2: case 3: f.skipBytes(2); return true;
            case 4: case 5: case 6: f.skipBytes(4); return true;
            case 7: f.skipBytes(1); return true;
            case 8: string(f); return true;
            case 9:
                int elem=(int)u32(f); long n=u64(f);
                if(n<0 || n>10000000) return false;
                for(long i=0;i<n;i++) if(!skipValue(f,elem)) return false;
                return true;
            case 10: case 11: case 12: f.skipBytes(8); return true;
            default: return false;
        }
    }

    /** Reads exactly one 34-byte type-142 block from the model file and decodes it. */
    public static String inspectFirstPq2Block(String path) {
        if (path == null || path.isEmpty()) return "ERR BONSAI2_PQ2_BLOCK null_path";
        try (java.io.RandomAccessFile f = new java.io.RandomAccessFile(path, "r")) {
            byte[] magic = new byte[4];
            f.readFully(magic);
            if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F')
                return "ERR BONSAI2_PQ2_BLOCK bad_magic";
            long version = u32(f);
            if (version != 2 && version != 3) return "ERR BONSAI2_PQ2_BLOCK version=" + version;
            long tensorCount = u64(f);
            long kvCount = u64(f);
            if (tensorCount <= 0 || tensorCount > 200000 || kvCount > 100000)
                return "ERR BONSAI2_PQ2_BLOCK counts";

            long alignment = 32;
            for (long i = 0; i < kvCount; i++) {
                String key = string(f);
                int type = (int)u32(f);
                if ("general.alignment".equals(key)) {
                    if (type != 4) return "ERR BONSAI2_PQ2_BLOCK alignment_type";
                    alignment = u32(f);
                    if (alignment <= 0 || alignment > (1L << 20) || (alignment & (alignment - 1)) != 0)
                        return "ERR BONSAI2_PQ2_BLOCK alignment";
                } else if (!skipValue(f, type)) {
                    return "ERR BONSAI2_PQ2_BLOCK metadata";
                }
            }

            long tensorBase = alignUp(f.getFilePointer(), alignment);
            long firstOffset = -1;
            String firstName = null;
            long firstDim = 0;
            long firstType = -1;
            for (long i = 0; i < tensorCount; i++) {
                String name = string(f);
                long dims = u32(f);
                if (dims <= 0 || dims > 8) return "ERR BONSAI2_PQ2_BLOCK dims";
                long dim0 = 0;
                for (long d = 0; d < dims; d++) {
                    long n = u64(f);
                    if (n <= 0) return "ERR BONSAI2_PQ2_BLOCK shape";
                    if (d == 0) dim0 = n;
                }
                long type = u32(f);
                long offset = u64(f);
                if (type == 142 && firstOffset < 0) {
                    firstOffset = offset;
                    firstName = name;
                    firstDim = dim0;
                    firstType = type;
                }
            }

            if (firstOffset < 0) return "ERR BONSAI2_PQ2_BLOCK no_type_142";
            if (firstDim % 128 != 0) return "ERR BONSAI2_PQ2_BLOCK dim0_not_group128";
            if (firstOffset > Long.MAX_VALUE - tensorBase) return "ERR BONSAI2_PQ2_BLOCK offset_overflow";

            long absolute = tensorBase + firstOffset;
            if (absolute < 0 || absolute > f.length() - 34) return "ERR BONSAI2_PQ2_BLOCK out_of_file";
            f.seek(absolute);

            int d0 = f.readUnsignedByte();
            int d1 = f.readUnsignedByte();
            float scale = halfToFloat(d0 | (d1 << 8));
            if (!Float.isFinite(scale)) return "ERR BONSAI2_PQ2_BLOCK bad_scale";

            float min = Float.POSITIVE_INFINITY, max = Float.NEGATIVE_INFINITY;
            double sum = 0.0;
            int[] hist = new int[4];
            for (int i = 0; i < 32; i++) {
                int packed = f.readUnsignedByte();
                for (int slot = 0; slot < 4; slot++) {
                    int q = (packed >>> (slot * 2)) & 3;
                    hist[q]++;
                    float v = (q - 1) * scale;
                    min = Math.min(min, v);
                    max = Math.max(max, v);
                    sum += v;
                }
            }
            return "OK BONSAI2_PQ2_BLOCK/1 type=142 tensor=" + firstName +
                    " absolute_offset=" + absolute + " block_bytes=34 scale=" + scale +
                    " min=" + min + " max=" + max + " sum=" + sum +
                    " codes=" + hist[0] + "," + hist[1] + "," + hist[2] + "," + hist[3];
        } catch (java.io.EOFException e) {
            return "ERR BONSAI2_PQ2_BLOCK truncated";
        } catch (Throwable t) {
            return "ERR BONSAI2_PQ2_BLOCK " + t.getClass().getSimpleName() + ": " + t.getMessage();
        }
    }

    private static long alignUp(long value, long alignment) {
        long mask = alignment - 1;
        if (value > Long.MAX_VALUE - mask) throw new IllegalArgumentException("alignment overflow");
        return (value + mask) & ~mask;
    }

    private static float halfToFloat(int h) {
        int sign = (h & 0x8000) << 16;
        int exp = (h >>> 10) & 0x1f;
        int mant = h & 0x3ff;
        int bits;
        if (exp == 0) {
            if (mant == 0) bits = sign;
            else {
                int e = -14;
                while ((mant & 0x400) == 0) { mant <<= 1; e--; }
                mant &= 0x3ff;
                bits = sign | ((e + 127) << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            bits = sign | 0x7f800000 | (mant << 13);
        } else {
            bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        }
        return Float.intBitsToFloat(bits);
    }

}
