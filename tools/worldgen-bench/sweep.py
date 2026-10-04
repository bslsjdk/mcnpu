#!/usr/bin/env python3
"""
Throughput sweep for the assist path.

Two questions decide whether this is worth building:
  1. how many chunks should be aggregated into one kernel call?
  2. how sensitive is the answer to kernel speed, which we have not measured?

Both are swept. The kernel-speed row matters because a single optimistic number would
make this look decided when it is not - the pessimistic bound is what we should plan against.
"""

import sys
import time
import numpy as np

from profile import (SNAPDRAGON_8S_GEN3 as P, NS_PER_OP_EFFICIENT,
                     NS_PER_OP_PESSIMISTIC)
from assist import AssistFrpcBackend
from worldgen import (FP32Backend, CHANNEL_ORDER, lattice_coords, density_field)
import diff


def main(seeds=8, chunks_per_axis=2):
    seeds_l = [1000 + i * 7919 for i in range(seeds)]

    # ---- vanilla oracle ---------------------------------------------------
    ref = FP32Backend()
    ref_rows, ref_ms = [], 0.0
    for s in seeds_l:
        for cx in range(chunks_per_axis):
            for cz in range(chunks_per_axis):
                c, _ = lattice_coords(cx * 16, -64, cz * 16)
                t = time.perf_counter()
                d = density_field(ref.channels(c, s, CHANNEL_ORDER), c)
                ref_ms += (time.perf_counter() - t) * 1000
                ref_rows.append(d)
    n = len(ref_rows)
    print(f"Vanilla oracle: {n} chunks, {ref_ms:.0f}ms, {ref_ms/n:.2f} ms/chunk\n")
    print(f"{'batch':>6} {'ms/chunk':>9} {'speedup':>8} {'MAE':>10} {'sign':>9} {'hMAE':>7} {'calls':>7} {'pad':>6}")
    print("-" * 72)

    # Chunks are grouped per seed, never across seeds: a kernel call carries one
    # noise state, so mixing worlds in one batch would silently evaluate the wrong
    # noise. This is a real constraint on the batcher, not just a test artefact -
    # an earlier version batched across seed boundaries and the error jumped ~100x,
    # which is how it was caught.
    def batches(size):
        out = []
        for s in seeds_l:
            coords = [lattice_coords(cx * 16, -64, cz * 16)[0]
                      for cx in range(chunks_per_axis)
                      for cz in range(chunks_per_axis)]
            for i in range(0, len(coords), size):
                out.append((s, coords[i:i + size]))
        return out

    for batch in [1, 2, 4, 8, 16, 32]:
        b = AssistFrpcBackend(P, ns_per_op=NS_PER_OP_EFFICIENT)
        rows = []
        for sd, cs in batches(batch):
            res = b.eval_batch(cs, sd, CHANNEL_ORDER)
            if res is None:
                continue
            for k, c in enumerate(cs):
                rows.append(density_field(res[k], c))

        stats = [diff.compare(ref_rows[i], rows[i], 5, 49, 5)
                 for i in range(len(rows))]
        agg = diff.merge(stats)
        # Device-side time is the modelled figure; host time is not what we are
        # trying to beat.
        dev_ms = b.latency_us / 1000.0
        per = dev_ms / max(1, len(rows))
        print(f"{batch:>6} {per:>9.2f} {ref_ms/max(1,len(rows))/per:>7.2f}x "
              f"{agg['density_mae']:>10.7f} {agg['sign_agreement']*100:>8.3f}% "
              f"{agg['height_mae']:>7.2f} {b.calls:>7} "
              f"{b.padded_units/b.units:>5.2f}x")

    print("\n--- kernel speed sensitivity (batch=16) ---")
    print(f"{'ns/op':>7} {'us/unit':>8} {'ms/chunk':>9} {'speedup':>8}")
    for ns in [NS_PER_OP_EFFICIENT, 2.0, NS_PER_OP_PESSIMISTIC]:
        b = AssistFrpcBackend(P, ns_per_op=ns)
        cnt = 0
        for sd, cs in batches(16):
            cnt += len(cs)
            b.eval_batch(cs, sd, CHANNEL_ORDER)
        per = b.latency_us / 1000.0 / max(1, cnt)
        print(f"{ns:>7.2f} {b.per_unit_us:>8.4f} {per:>9.2f} "
              f"{ref_ms/n/per:>7.2f}x")


if __name__ == "__main__":
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    c = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    main(a, c)
