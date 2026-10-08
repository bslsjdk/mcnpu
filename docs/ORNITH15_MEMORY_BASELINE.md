# Ornith-1.5-9B mobile memory baseline

## Goal

Use Ornith-1.5-9B as the first complete local LLM target for MCNPU.

The target is not merely "make it fit":

- 64K real context is the first acceptance gate.
- 262144 native context is the final target.
- Runtime process RAM must remain within the 4 GiB hard ceiling.
- Preserve model quality as much as possible.
- Preserve prefill and decode speed as much as possible.
- No cloud inference and no fake model output.
- MCNPU owns the persistent local QNN/HTP device.
- The model runtime owns weights, tokenizer, recurrent state, KV cache, scheduling and sampling.

## Architecture

The official configuration describes Qwen3.5:

- 32 decoder layers.
- 8 full-attention layers, every fourth layer.
- 24 linear-attention/GDN layers.
- hidden size 4096.
- 4 KV heads for full attention, head dimension 256.
- native context 262144 tokens.

The hybrid layout is the important long-context property: only the eight full-attention layers retain conventional K/V history; the other 24 layers maintain fixed recurrent state.

For an FP16 full-attention KV baseline:

8 layers * 2 (K,V) * 4 KV heads * 256 dim * 2 bytes = 32768 bytes/token.

Therefore the arithmetic baseline is:

- 64K FP16 KV = 2 GiB.
- 262K FP16 KV = 8 GiB.

A Q8 KV representation would be about half that storage before metadata and alignment overhead. This is only a planning estimate until implemented and measured.

## Memory strategy

1. Keep the GGUF weights mmap-backed instead of copying the entire model into anonymous RAM.
2. Keep only hot compute tensors and reusable scratch buffers resident.
3. Reuse NPU and CPU staging buffers across layers.
4. Treat full-attention KV separately from the fixed recurrent state of linear-attention layers.
5. Benchmark FP16 KV, Q8 KV and later more aggressive representations against identical long-context tests.
6. Track RSS and PSS during prefill and decode.
7. Do not use model file size as a proxy for runtime RAM.
8. Do not claim 64K support unless the runtime really preserves 64K tokens without silent truncation or summarization.

## First implementation gate

Ornith15Probe.inspect(path, contextTokens) reads only the GGUF header, metadata and tensor descriptors. It never loads model weights.

It reports:

- architecture
- layer count
- full/linear attention counts
- hidden/vocabulary size
- native/requested context
- tensor type histogram
- full-attention KV bytes/token
- FP16 KV budget
- Q8 KV planning estimate

## Acceptance gates

### Gate A: 64K

- same model weights
- 64K real context
- no history truncation
- runtime RAM <= 4 GiB
- quality compared with the reference runtime
- prefill/decode latency recorded

### Gate B: 128K

Same measurements.

### Gate C: 262K

Only attempt after the previous gates are stable. The FP16 KV arithmetic baseline alone is about 8 GiB, so the final target requires a more compact state representation.

## Current status

MCNPU already provides a persistent QNN/HTP owner and binary INT8 matmul transport. Ornith runtime integration is not complete.

The next implementation stage is the real Ornith GGUF loader/runtime skeleton, followed by a true 64K KV implementation and memory instrumentation. Synthetic assistant output is not an acceptable substitute for inference.
