package bslsjdk.mcnpu;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.EOFException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Strict metadata-only probe for the one Android target:
 * Ornith-1.5-9B-MLX-4bit/model.safetensors.
 *
 * This deliberately does not interpret the file as GGUF and never maps or
 * allocates the 4.69+ GiB weight payload. It validates the Safetensors
 * container, Qwen3.5 layer topology, and the MLX 4-bit tensor companion layout.
 */
public final class Ornith15MlxProbe {
    private static final long MAX_HEADER_BYTES = 16L * 1024L * 1024L;
    private static final long MAX_TENSOR_COUNT = 10000L;
    private static final int EXPECTED_LAYERS = 32;
    private static final int EXPECTED_HIDDEN = 4096;
    private static final int EXPECTED_VOCAB = 248320;
    private static final long EXPECTED_CONTEXT = 262144L;
    private static final int EXPECTED_BITS = 4;
    private static final int EXPECTED_GROUP_SIZE = 64;

    private static final Pattern LAYER =
            Pattern.compile("(?:^|\\.)layers\\.(\\d+)(?:\\.|$)");

    private Ornith15MlxProbe() {}

    public static String inspect(String path, long requestedContext) {
        if (path == null || path.isEmpty())
            return "ERR ORNITH15_MLX null_path";
        if (requestedContext <= 0 || requestedContext > EXPECTED_CONTEXT)
            return "ERR ORNITH15_MLX context=" + requestedContext;

        try (RandomAccessFile f = new RandomAccessFile(path, "r")) {
            long fileBytes = f.length();
            if (fileBytes < 8)
                return "ERR ORNITH15_MLX truncated";

            long headerBytes = readU64LE(f);
            if (headerBytes <= 0 || headerBytes > MAX_HEADER_BYTES ||
                    headerBytes > fileBytes - 8)
                return "ERR ORNITH15_MLX bad_header_length=" + headerBytes;

            byte[] raw = new byte[(int) headerBytes];
            f.readFully(raw);
            String json = new String(raw, StandardCharsets.UTF_8);

            JSONObject root = new JSONObject(json);
            long tensorCount = 0;
            long metadataCount = 0;
            Set<Integer> layers = new HashSet<>();
            Set<String> dtypes = new HashSet<>();
            long quantWeights = 0;
            long quantScales = 0;
            long quantBiases = 0;
            long bf16 = 0;
            long f16 = 0;
            long u32 = 0;
            long minOffset = Long.MAX_VALUE;
            long maxOffset = 0;

            Iterator<String> keys = root.keys();
            while (keys.hasNext()) {
                String name = keys.next();
                if ("__metadata__".equals(name)) {
                    Object meta = root.opt(name);
                    if (meta instanceof JSONObject) metadataCount =
                            ((JSONObject) meta).length();
                    continue;
                }

                Object value = root.get(name);
                if (!(value instanceof JSONObject))
                    return "ERR ORNITH15_MLX tensor_entry=" + name;

                JSONObject tensor = (JSONObject) value;
                tensorCount++;
                if (tensorCount > MAX_TENSOR_COUNT)
                    return "ERR ORNITH15_MLX tensor_count";

                String dtype = tensor.optString("dtype", "");
                JSONArray shape = tensor.optJSONArray("shape");
                JSONArray offsets = tensor.optJSONArray("data_offsets");
                if (dtype.isEmpty() || shape == null || offsets == null ||
                        offsets.length() != 2)
                    return "ERR ORNITH15_MLX malformed_tensor=" + name;

                dtypes.add(dtype);
                if ("BF16".equals(dtype)) bf16++;
                else if ("F16".equals(dtype)) f16++;
                else if ("U32".equals(dtype)) u32++;

                long start = offsets.getLong(0);
                long end = offsets.getLong(1);
                if (start < 0 || end < start || end > fileBytes - 8 - headerBytes)
                    return "ERR ORNITH15_MLX bad_offsets=" + name;
                minOffset = Math.min(minOffset, start);
                maxOffset = Math.max(maxOffset, end);

                Matcher m = LAYER.matcher(name);
                if (m.find()) {
                    int layer = Integer.parseInt(m.group(1));
                    if (layer < 0 || layer >= EXPECTED_LAYERS)
                        return "ERR ORNITH15_MLX layer=" + layer;
                    layers.add(layer);
                }

                if (name.endsWith(".weight")) {
                    // MLX quantized linear weights are packed into U32 words.
                    if ("U32".equals(dtype)) quantWeights++;
                } else if (name.endsWith(".scales")) {
                    quantScales++;
                } else if (name.endsWith(".biases")) {
                    quantBiases++;
                }

                // Shape sanity for the model's embedding/logit width. We don't
                // require a particular quantization dtype here because the
                // embedding implementation is allowed to differ from linears.
                if (name.contains("embed_tokens") || name.equals("lm_head.weight")) {
                    if (shape.length() >= 2) {
                        long a = shape.optLong(shape.length() - 2, -1);
                        long b = shape.optLong(shape.length() - 1, -1);
                        if (!((a == EXPECTED_VOCAB && b == EXPECTED_HIDDEN) ||
                              (a == EXPECTED_HIDDEN && b == EXPECTED_VOCAB)))
                            return "ERR ORNITH15_MLX embedding_shape=" + name;
                    }
                }
            }

            if (tensorCount == 0)
                return "ERR ORNITH15_MLX no_tensors";
            if (layers.size() != EXPECTED_LAYERS)
                return "ERR ORNITH15_MLX layers=" + layers.size() +
                        " expected=" + EXPECTED_LAYERS;

            // A real MLX 4-bit export must contain packed U32 weight tensors
            // accompanied by scales and biases. We intentionally require all
            // three classes instead of merely looking for the word "4bit".
            if (quantWeights == 0 || quantScales == 0 || quantBiases == 0)
                return "ERR ORNITH15_MLX quant_layout weights=" + quantWeights +
                        " scales=" + quantScales + " biases=" + quantBiases;

            if (minOffset == Long.MAX_VALUE || maxOffset <= minOffset)
                return "ERR ORNITH15_MLX empty_data";

            long dataBytes = fileBytes - 8L - headerBytes;
            StringBuilder out = new StringBuilder("OK ORNITH15_MLX_PROBE/1");
            out.append(" format=SAFETENSORS");
            out.append(" quant=MLX_4BIT");
            out.append(" bits=").append(EXPECTED_BITS);
            out.append(" group_size=").append(EXPECTED_GROUP_SIZE);
            out.append(" layers=").append(layers.size());
            out.append(" hidden=").append(EXPECTED_HIDDEN);
            out.append(" vocab=").append(EXPECTED_VOCAB);
            out.append(" native_context=").append(EXPECTED_CONTEXT);
            out.append(" requested_context=").append(requestedContext);
            out.append(" tensors=").append(tensorCount);
            out.append(" u32=").append(u32);
            out.append(" bf16=").append(bf16);
            out.append(" f16=").append(f16);
            out.append(" packed_weights=").append(quantWeights);
            out.append(" scales=").append(quantScales);
            out.append(" biases=").append(quantBiases);
            out.append(" dtypes=").append(dtypes);
            out.append(" metadata=").append(metadataCount);
            out.append(" data_bytes=").append(dataBytes);
            out.append(" file_bytes=").append(fileBytes);
            return out.toString();
        } catch (EOFException e) {
            return "ERR ORNITH15_MLX truncated";
        } catch (JSONException e) {
            return "ERR ORNITH15_MLX bad_json=" + e.getMessage();
        } catch (Throwable t) {
            return "ERR ORNITH15_MLX " + t.getClass().getSimpleName() +
                    ": " + t.getMessage();
        }
    }

    private static long readU64LE(RandomAccessFile f) throws IOException {
        long v = 0;
        for (int i = 0; i < 8; i++)
            v |= ((long) f.readUnsignedByte()) << (8 * i);
        return v;
    }
}
