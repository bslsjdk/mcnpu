# ADD ceiling: where 16384 came from, and the fp16 question

## 1. Is 16384 a device limit? No.

Confirmed by reading the code, not by assuming:

| Where | What it is |
|---|---|
| `ADD_LADDER_BASE[]` in `app/src/main/cpp/mcnpu.cpp` | a literal array in **our own .so** |
| `g_addLadderMax` | a mutable global in **our own .so** |
| `MainActivity.maxCaseLen()`, `NpuService` `BINADD` range, `NpuBigAdd.maxWay()` | Java copies of the same number |

Every one of them lives in code we compile. Nothing referenced `/vendor/lib64`
or any QNN symbol to obtain it. So:

- **No root is required.** Recompile + install is enough. This matches the
  conclusion that raising it is a user-space change.
- The number was only *the largest size we had verified*, never a measurement.
  Anything above it was rejected **before it could reach the HTP**, so "the
  device caps ADD at 16384" was a conclusion our own code produced.

## 2. We did not hardcode 65536

Hardcoding a bigger number would be silently wrong on any phone whose real
ceiling is lower, and would still not tell us where the ceiling actually is.

Instead `nativeAddProbe()` walks candidate sizes on the real device, ascending,
stopping at the first failure (if 65536 cannot be built, 131072 will not
either, and continuing only burns graphs). The measured value flows outward:

```
probe -> g_addLadderMax -> nativeAddMax() -> NpuRuntime.maxAddElements()
      -> HELLO max_elements= -> NpuServiceClient.cachedMaxAdd
      -> NpuBigAdd.maxWay(), NpuBinFile, MainActivity.maxCaseLen()
```

No layer hardcodes the ceiling any more; they all read the measured one.

Two details that matter:

- The probe deliberately **does not** hold `gRuntimeMutex` across the run.
  `runAddEx` takes that lock itself and `std::mutex` is not recursive, so a
  probe holding it would deadlock on its own first candidate.
- The probe builds one graph per candidate, which is exactly the cache pressure
  the ladder exists to avoid, so it calls `resetContextLocked()` afterwards.

## 3. fp16

ADD ran on `QNN_DATATYPE_FLOAT_32` even though fp16 is the datatype HTP treats
as native. `runAddEx` now takes an `fp16` flag, and the graph cache key became
`(n, fp16)` — keying on `n` alone would hand an fp32 graph to an fp16 call and
the buffer would be read at half the element size.

The probe now walks the sizes twice and reports:

```
OK ADD_PROBE max_fp32=... max_fp16=... ladder=...
 fp32 16 ok ... create_us=... execute_us=...
 fp16 16 ok ... create_us=... execute_us=...
```

**The data path stays fp32 on purpose.** Changing the wire format is a separate
decision and should be made with these numbers in hand, not before.

Why measure it at all if ADD is only a test op: bytes per element is what sets
the way count (7,962,624 / 16384 = 486 ways for a 9x9). Halving the element size
halves both the IPC payload and, potentially, the way count. The same reasoning
will apply to the future Perlin kernel, whose output is low-precision anyway.

## 4. Conversion correctness

`f32ToF16` / `f16ToF32` are hand-written (no libm, no `_Float16`, since the .so
builds against a different ABI than the QNN samples). Verified offline against
zero, subnormals, normals, inf/nan bounds and 65504: max relative error 4.1e-4,
and every fp16 subnormal round-trips bit-exactly. The verify tolerance for fp16
is 5e-3, because the fp32 tolerance (1e-3) is tighter than the format itself and
would report a perfectly good execution as a failure.

## 5. What to look for in the log

```
ADD_PROBE elapsed_ms=... 
OK ADD_PROBE max_fp32=<N> max_fp16=<M> ladder=16,64,256,1024,4096,16384,<N>
 fp32 16384 ok elapsed_us=... create_us=... execute_us=...
 fp32 32768 FAIL ...
 fp16 32768 ok ...
```

- `max_fp32` much larger than 16384 -> way count for a 9x9 drops from 486
  proportionally, and the per-call overhead drops with it.
- `max_fp16 > max_fp32` -> fp16 buys size, and halving the element size buys
  bandwidth on top. That is the case for changing the wire format.
- `max_fp16 == max_fp32` -> fp16 buys only bandwidth. Weigh it against the
  host-side conversion pass, which is O(n) CPU work per call.

## 6. Still open

- The probe measures *"can it build"*, not *"is it fast"*. Bigger is still
  right in general (total execute work is the same; only the per-call overhead
  shrinks), but `execute_us` is logged per size so a size that is correct but
  pathologically slow would be visible rather than silently adopted.
- `service_queue_us=2298` on **every** request, byte-identical, is still the
  highest-value unexplained number. Real queueing varies. It is likely a stale
  timestamp, and if so, fixing it is worth more than any kernel work.
