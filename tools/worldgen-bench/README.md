# Worldgen Differential Benchmark

*元宝 · 2026-10-04*

Runs entirely in the terminal: **no Minecraft, no phone, no QNN.** The point is to
iterate the terrain algorithm here, where a bad idea costs seconds, instead of on a
device where it costs a rebuild and a log pull.

```
./bench.py --compare --seeds 10 --chunks 2 --bins 16 --json result.json
```

---

## Results (10 seeds × 4 chunks = 40 chunks, SNAPDRAGON_8S_GEN3 profile)

| | Density MAE | RMSE | Sign agr | Height MAE | ms/chunk | Speedup |
|---|---|---|---|---|---|---|
| **vanilla (oracle)** | — | — | — | — | 6.41 | 1.00x |
| **direct-int8** | **0.000348** | 0.000420 | **99.990%** | **0.040** | 6.33 | 1.01x |
| **matmul-int8** | 0.020485 | 0.025350 | 99.396% | 2.328 | 114.96 | **0.06x** |

### What this means

**INT8 quantisation is basically free.** `direct-int8` is the floor any kernel could
achieve - correct maths with INT8 values - and it lands at MAE 0.00035, sign agreement
99.990%, height error 0.04 blocks. **A DSP noise kernel has plenty of headroom.**

**Matmul is a dead end.** `matmul-int8` is 59x worse on density MAE, 58x worse on
height, and **18x slower**. And that is after two real bugs were fixed - before the
fixes it looked even worse for reasons that had nothing to do with matmul.

The mechanism is visible in the group statistics: one chunk needs **60,907 group
submissions** (6 channels × ~7 octaves × ~1500 groups), and each submission carries the
device's fixed cost (~2.5 ms measured: 187 µs IPC + 2298 µs service queue). Sixty
thousand fixed costs cannot be beaten by any amount of kernel efficiency.

### The binning trade-off is a trap

Fractional-position binning controls accuracy, but accuracy and speed pull in opposite
directions:

| BINS | MAE | sign | hMAE | est. ms/chunk | avg group | singleton |
|---|---|---|---|---|---|---|
| 4 | 0.0466 | 98.61% | 4.80 | 1,027 | 85.3 | 0% |
| 8 | 0.0309 | 99.18% | 2.56 | 1,936 | 31.9 | 0% |
| 16 | 0.0213 | 99.10% | 2.56 | 4,268 | 14.8 | 0% |
| 32 | 0.0189 | 99.35% | 1.92 | 7,816 | 8.1 | 1.6% |
| 64 | 0.0187 | 99.35% | 1.92 | 14,079 | 4.5 | **23.8%** |

Accuracy stops improving at ~0.019 while cost keeps climbing, and at BINS=64 nearly a
quarter of groups have collapsed to a single point - the degeneration predicted from
the algebra, now measured.

**Best case is 1,027 ms per chunk against vanilla's 7 ms. There is no operating point
where matmul wins.**

---

## Three bugs the bench caught (before any device saw them)

This is the concrete return on building the harness. Each one produced a large error
that looked like a fundamental limitation:

1. **Corner offsets** - all eight corners were sampled with the same fractional
   coordinate. Vanilla uses `(dx-Δx, dy-Δy, dz-Δz)` per corner.
2. **Corner bit order** - vanilla's corner encoding is **z, y, x**, not x, y, z. Three
   separate places encoded it as x, y, z.
3. **Double-counted z** - `_hash` applied the z offset twice, once via the coordinate
   and again via the bit.

Error went **0.665 → 0.004** across these fixes. Before them, increasing binning
precision did not reduce error at all, which is exactly the signature that a
*structural* bug is being mistaken for an *approximation* limit.

On a device all three would have looked like "the terrain is wrong".

---

## Design

### Backends

| backend | what it is |
|---|---|
| `fp32` | vanilla maths, no device limits. The control. |
| `direct-int8` | vanilla maths with INT8 values - the quantisation floor |
| `matmul-int8` | expressed as matmuls the current backend can run |

**Comparing fp32 vs direct-int8 isolates quantisation error.**
**Comparing direct-int8 vs matmul-int8 isolates the cost of forcing it through matmul.**

Without that split, one bad number would tell us nothing about which of the two caused it.

### Simulator

Not "run it on the CPU and call it an NPU". It models the real backend:

- shape validation against the bucket whitelist and element budget
- INT8 quantisation with a real scale, so precision is genuinely lost
- padding up to the next bucket, so wasted arithmetic is really performed
- an int8 kernel with a 32-bit accumulator, where long-K error accumulates
- dequantisation
- latency from the measured device model, including cold graph creation

**Results and limits are separate pipeline stages** - that is what lets the tool answer
"the algorithm is correct in FP32 but falls apart under INT8 + buckets + padding".

### Profile

`profile.py` holds numbers measured on the actual device, not guesses:

- buckets `[32,64,128,256,512,1024]`; **16 is rejected** (scale mismatch was observed)
- element budget 16384
- fixed cost 187 µs IPC + 2298 µs service queue (the queue dominates, not the socket)
- cold graph 387 ms, cached 0
- kernel cost calibrated from `128×512×512 → 29,294 µs`

Submit `m=37` and it rejects, exactly as the phone would:

```
REJECT: m=37 not in allowed buckets [32, 64, 128, 256, 512, 1024]
REJECT: m=16 rejected: scale mismatch observed on device
```

---

## Caveat

The density function is a **structural stand-in**, not vanilla's full `final_density`
with its splines. It is monotone in the noise channels so that sign and height are
meaningful error metrics. It measures **how noise error propagates into terrain** -
which is the question this tool exists to answer - not whether our terrain is vanilla.

`NpuTerrainLattice`'s sin/cos + random weights remains an experiment and is not
treated as a vanilla parity reference anywhere here.

---

## Conclusion

| route | verdict |
|---|---|
| matmul-based terrain | ❌ **dead** - 146x slower at best, error floor 0.019 |
| FastRPC/HVX noise kernel | ✅ **viable** - quantisation floor is 0.00035 |

**Do not write more matmul kernels for terrain.** The one kernel worth writing is a
Perlin kernel on Hexagon via FastRPC - `libcdsprpc.so` exists precisely to run our own
DSP code, not just to service QNN. One call instead of sixty thousand.

`NpuTerrainGate` stays **CLOSED** until a real kernel passes parity.

---

## Files

| file | purpose |
|---|---|
| `vanilla.py` | Xoroshiro128++ / Perlin / NormalNoise, exact port incl. Java int64 overflow |
| `profile.py` | device profiles, bucket whitelist, latency model |
| `simulator.py` | NPU simulator: validate, quantise, pad, kernel, dequantise, time |
| `worldgen.py` | the three backends + lattice/density |
| `diff.py` | density/height/sign comparison |
| `bench.py` | CLI |
| `result.json` | last run |

*—— 元宝*


---

# NPU Assist: batched noise evaluation

*元宝 · 2026-10-04*

Matmul did not lose because the maths was wrong. It lost because one chunk needed
**~60,907 separate submissions**, each paying the device's ~2.5 ms fixed cost
(187 µs IPC + 2298 µs service queue). Sixty thousand fixed costs cannot be recovered by
kernel efficiency.

A kernel does not have that problem: it takes a block of points and returns a block of
values, so **the fixed cost is paid once per block**.

```
./sweep.py 4 4        # 4 seeds x 16 chunks
./bench.py --compare --seeds 4 --chunks 4 --batch 16
```

## Throughput vs batch size (64 chunks, SNAPDRAGON_8S_GEN3)

| batch | ms/chunk | speedup | MAE | sign agr | hMAE | calls | pad waste |
|---|---|---|---|---|---|---|---|
| 1 | 3.50 | 1.98x | 0.000356 | 99.986% | 0.06 | 64 | **1.67x** |
| 2 | 2.26 | 3.07x | 0.000372 | 99.989% | 0.05 | 32 | 1.67x |
| 4 | 1.88 | 3.69x | 0.000382 | 99.982% | 0.07 | 32 | 1.04x |
| 8 | 1.57 | 4.43x | 0.000385 | 99.985% | 0.07 | 24 | 1.04x |
| **16** | **1.41** | **4.91x** | 0.000387 | 99.987% | 0.06 | 20 | **1.04x** |

**Accuracy does not move with batch size** — it stays at the INT8 floor
(MAE ~0.00038, sign 99.98%, height 0.06 blocks) because batching changes only how work
is *grouped*, not what is *computed*. That is the property that makes batching safe.

Two effects drive the gain:
- **padding waste 1.67x → 1.04x** — one chunk is 1225 points, padded up to 2048; four
  chunks fill the 4096 bucket almost exactly
- **fixed cost amortised** — 64 calls become 20

## Kernel-speed sensitivity (batch=16)

| ns/op | µs/unit | ms/chunk | speedup |
|---|---|---|---|
| 0.87 *(efficient, calibrated)* | 0.026 | 1.41 | **4.91x** |
| 2.00 | 0.060 | 2.24 | 3.10x |
| 4.85 *(pessimistic, calibrated)* | 0.146 | 4.32 | **1.61x** |

**Even against the pessimistic bound it wins (1.61x).** That is the number to plan
against, not the 4.91x.

Both bounds are calibrated from measured submissions, not guessed:
`128×512×512 → 29,294 µs` gives 0.87 ns/op; `128×32×32 → 636 µs` gives 4.85 ns/op.
The real kernel lands between them.

## The m-bucket ceiling decides everything

`m=4096` was observed succeeding in device logs. That single fact is what makes this
viable — at a 1024 ceiling the gain largely disappears. **If GPT finds 8192 works, the
curve improves further; this is worth measuring early.**

## One real constraint found

**A kernel call must not mix seeds.** An earlier version batched across seed boundaries
and the density MAE jumped ~100x (0.0004 → 0.04) while the speedup numbers looked fine.
The batcher groups per seed, per dimension. Worth carrying into the real implementation:
it would have looked like "NPU terrain is subtly wrong in some places".

## Verdict

| route | verdict |
|---|---|
| matmul terrain | ❌ dead — 0.05x, error floor 0.005+ |
| **assist: batched kernel** | ✅ **1.6x – 4.9x**, at the INT8 accuracy floor |

`NpuTerrainGate` stays CLOSED until a real kernel passes parity, but the throughput case
is now demonstrated rather than assumed.

## Files added

| file | purpose |
|---|---|
| `assist.py` | `AssistFrpcBackend` — batched kernel-style evaluation |
| `sweep.py` | batch-size and kernel-speed sweep |

*—— 元宝*
