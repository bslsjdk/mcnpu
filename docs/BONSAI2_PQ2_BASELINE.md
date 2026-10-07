# Bonsai 2 PQ2_0 baseline

## What is verified now

The PrismML fork defines PQ2_0 as GGML type 142, group size 128, with a 34-byte block:

- 2-byte little-endian FP16 scale d
- 32 bytes of packed 2-bit codes
- four values per packed byte
- code mapping: 00 -> -1, 01 -> 0, 10 -> +1, 11 -> +2
- decoded value: (code - 1) * d

The standalone native probe in app/src/main/cpp/bonsai_pq2.cpp constructs one synthetic block, decodes all 128 values, and checks every output against the expected four-code cycle. It does not load model weights and does not allocate a model-sized buffer.

A small Java wrapper is available as Bonsai2Pq2Probe.run(). A second entry point, Bonsai2Pq2Probe.runHadamard(), validates the normalized 1024-point FWHT self-inverse property with an explicit sign vector.

The Java probe also has inspectFirstPq2Block(path). It parses the real GGUF header, metadata and tensor descriptors, locates the first type-142 tensor payload, seeks to its first 34-byte block, and reports its real FP16 scale and 2-bit histogram. It never allocates the model weights.

This is still a correctness/probing layer only. A passing probe does not mean that Bonsai 2 inference is integrated.

## Prism reference

The PrismML fork's current model-format documentation identifies PQ2_0 as the private group-128 format and states that Bonsai 2 also requires the Prism activation/Hadamard transform.

The Hadamard metadata is not a weight-side transform. The Prism loader keys each explicit sign vector by the input width of the affected weight: the current Bonsai 2 packs use widths 5120, 6144 and 17408. The sign vector is applied to the activation before the normalized FWHT; the inverse transform is used for the special token embedding path. Therefore the next implementation must not accidentally multiply decoded PQ2 weights by the sign vector.

## Still not implemented

1. GGUF model loading for type 142 in MCNPU.
2. Bonsai 2 tokenizer/model graph/runtime.
3. Exact Hadamard/sign/permutation activation transform.
   - The normalized 1024-point FWHT algebra is covered by a standalone correctness probe.
   - Real GGUF metadata parsing and the model graph path are not integrated yet.
   - The real sign vectors are validated by the Java GGUF validator, but are not yet fed into an inference graph.
4. Mapping transformer matmuls onto the existing HTP INT8 path.
5. KV-cache management and sampling.
6. End-to-end correctness against the official Prism runtime.

The architecture remains intentionally split: the future Bonsai runtime owns model weights, KV cache, tokenizer, graph scheduling and sampling; MCNPU remains the persistent local QNN/HTP owner. The 4 GiB runtime-memory ceiling remains a hard constraint for the Android process.

## Next gate

Before loading the 27B file into the inference runtime, finish the exact GGUF type-142 reader and Hadamard/sign/permutation path on a tiny synthetic tensor. Then benchmark one real transformer matrix path through HTP using a model-shaped dimension, compare it with a CPU reference, and only after that begin end-to-end runtime integration.
