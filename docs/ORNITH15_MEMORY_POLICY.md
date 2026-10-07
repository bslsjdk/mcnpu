# Ornith-1.5-9B memory policy

The runtime ceiling is 4 GiB. The planner deliberately reserves 1 GiB for
non-KV resident memory, leaving 3 GiB as the initial KV budget.

This is a planning guard, not a guarantee. Actual RSS/PSS must still be measured.

## Estimated KV storage

The model has eight full-attention layers. With 4 KV heads and 256 dimensions,
FP16 K+V costs 32768 bytes/token.

| Context | FP16 | Q8/Q8 | Q8/Q5 | Q8/Q4 |
|---|---:|---:|---:|---:|
| 64K | 2.00 GiB | 1.00 GiB | ~0.81 GiB | 0.75 GiB |
| 128K | 4.00 GiB | 2.00 GiB | ~1.63 GiB | 1.50 GiB |
| 262K | 8.00 GiB | 4.00 GiB | ~3.25 GiB | 3.00 GiB |

These are storage-only estimates. Quantization metadata, alignment, recurrent
state, runtime objects, mapped-weight resident pages and NPU buffers are not
included.

The current planner therefore reserves 1 GiB outside KV and selects the least
aggressive *estimated* mode that fits. It does not implement quantization.

## Why Q8/Q5 is the first long-context candidate

Recent llama.cpp experiments report that asymmetric Q8 K / Q5 V can preserve
quality better than more aggressive Q4 KV while still cutting cache storage
substantially. Other measurements show that quantized KV can be close to F16
throughput on some backends but slower on others, so the mobile HTP path must
measure both quality and decode speed rather than assuming compression is free.

For 262K, Q8/Q5 is still too close to the 4 GiB ceiling once non-KV memory is
counted. Therefore the final 262K target requires either:

- a more compact KV representation with acceptable quality, or
- additional state compression/offload that does not violate the RAM ceiling.

The runtime must never silently fall back to a smaller context.

## Next implementation step

Implement the actual KV storage abstraction behind this planner:

1. F16 reference cache.
2. Q8 K/Q8 V cache.
3. Q8 K/Q5 V cache.
4. Q8 K/Q4 V only as an experimental extreme mode.
5. identical long-context replay tests;
6. RSS/PSS + prefill/decode measurements;
7. choose the least aggressive mode that passes the quality and speed gates.

A cache mode is not considered supported merely because its arithmetic budget fits.
