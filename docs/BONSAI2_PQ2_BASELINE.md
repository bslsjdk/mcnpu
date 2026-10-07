# Bonsai 2 PQ2_0 baseline

## What is verified now

The PrismML fork defines PQ2_0 as GGML type 142, group size 128, with a 34-byte block:

- 2-byte little-endian FP16 scale d
- 32 bytes of packed 2-bit codes
- four values per packed byte
- code mapping: 00 -> -1, 01 -> 0, 10 -> +1, 11 -> +2
- decoded value: (code - 1) * d

The standalone native probe in app/src/main/cpp/bonsai_pq2.cpp constructs one synthetic block, decodes all 128 values, and checks every output against the expected four-code cycle. It does not load model weights and does not allocate a model-sized buffer.

A small Java wrapper is available as Bonsai2Pq2Probe.run().

This is a codec correctness probe only. A passing probe does not mean that Bonsai 2 inference is integrated.

## Prism reference

The implementation baseline is the PrismML prism branch. Their current model format documentation identifies PQ2_0 as the private group-128 format and states that Bonsai 2 also requires the Prism activation/Hadamard transform.

## Still not implemented

1. GGUF model loading for type 142 in MCNPU.
2. Bonsai 2 tokenizer/model graph/runtime.
3. Hadamard/sign/permutation activation transform.
4. Mapping transformer matmuls onto the existing HTP INT8 path.
5. KV-cache management and sampling.
6. End-to-end correctness against the official Prism runtime.

The architecture remains intentionally split: the future Bonsai runtime owns model weights, KV cache, tokenizer, graph scheduling and sampling; MCNPU remains the persistent local QNN/HTP owner. The 4 GiB runtime-memory ceiling remains a hard constraint for the Android process.

## Next gate

Before loading the 27B file, implement and test the GGUF type-142 reader plus the exact Prism Hadamard/sign/permutation metadata path on a tiny synthetic tensor. Only after that should the runtime touch a real PQ2_0 model file.
