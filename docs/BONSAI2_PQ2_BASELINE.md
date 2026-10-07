# Bonsai 2 27B PQ2_0 NPU baseline

This branch is an isolated experiment on top of MCNPU. The `main` branch is not modified.

## Goal

Establish a measurable low-level NPU data path that a PrismML/llama.cpp Bonsai 2 integration can use later.

The MCNPU process remains the persistent QNN/HTP owner. Bonsai/llama.cpp remains responsible for model loading, tokenizer, sampling, KV cache, and the transformer execution schedule.

## Current transport

Loopback TCP:

- address: `127.0.0.1:38761`
- control framing: one UTF-8 command line terminated by LF
- tensor payload: raw binary bytes immediately following the command line
- response: UTF-8 header line followed by raw binary output when specified
- legacy JSON/text paths remain available for diagnostics

The existing binary INT8 matrix path is:

```
SUBMITBIN_MATMUL8 m k n alen blen\n
<A: alen raw int8 bytes>
<B: blen raw int8 bytes>
```

Response:

```
OK BIN_SUBMIT ... scaleC=<float> cbytes=<m*n> ... binary=1\n
<C: m*n raw int8 bytes>
```

The first four bytes returned by the native JNI buffer path are the little-endian IEEE-754 `scaleC` value. The IPC response puts that scale in the text header and returns only the raw result bytes.

## Why this is only the baseline

Bonsai 2 PQ2_0 is not equivalent to ordinary INT8 weights. The existing INT8 HTP path therefore must not be described as "Bonsai PQ2 execution" yet.

The next integration layer must translate the actual PrismML/llama.cpp PQ2 kernels into supported HTP operations and preserve the model's rotated-basis / dequantization semantics. Until that adapter exists, the correct experiment is a kernel-level baseline, not a fake end-to-end Bonsai benchmark.

## Baseline measurements

For every candidate kernel, record:

- prefill latency
- decode latency
- HTP execution time
- IPC round-trip time
- input/output bytes
- CPU time
- process RSS/PSS
- total device RAM pressure
- storage read bytes and read bytes/token

The hard runtime memory ceiling for this project is 4 GB. Any implementation that crosses it is rejected even if its throughput is higher.

## Required comparison order

1. Existing MCNPU INT8 MATMUL path.
2. Persistent binary IPC with a reused connection.
3. HTP shape-bucket/cache behavior for the actual Bonsai matrix shapes.
4. Only after the transport is stable: integrate the PrismML/llama.cpp PQ2 kernel.
5. Then compare dense HTP against any sparse variant with a real device microbenchmark.

## Non-goals

- Do not load the 27B model into MCNPU.
- Do not duplicate the model in the Android service.
- Do not stream transformer layers through IPC one layer at a time.
- Do not claim PQ2 support merely because INT8 MATMUL works.
- Do not remove the existing diagnostic or legacy paths while the experiment is being validated.
