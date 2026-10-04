#!/usr/bin/env python3
"""
Worldgen Differential Benchmark.

Runs entirely in the terminal: no Minecraft, no phone, no QNN. The point is to iterate
the algorithm here, where a bad idea costs seconds, instead of on a device where it
costs a rebuild and a log pull.

    ./bench.py --backend vanilla --seeds 20 --chunks 4
    ./bench.py --backend npu-sim --seeds 20 --chunks 4
    ./bench.py --compare --seeds 20 --chunks 4 --json out.json
"""

import argparse
import csv
import json
import sys
import time

import numpy as np

from profile import PROFILES
from simulator import NpuSimulator
from worldgen import (BACKENDS, CHANNEL_ORDER, MatmulInt8Backend,
                      lattice_coords, density_field, SIZE_Y, STEP_Y)
import diff

REGIONS = ["ocean", "plains", "hills", "mountains"]


def classify(h):
    m = float(np.mean(h[h >= 0])) if np.any(h >= 0) else -64.0
    if m < 55:
        return "ocean"
    if m < 70:
        return "plains"
    if m < 90:
        return "hills"
    return "mountains"


BINS_OVERRIDE = 16

def run_backend(name, seeds, chunks, profile_name="SNAPDRAGON_8S_GEN3"):
    profile = PROFILES[profile_name]
    try:
        from worldgen import MatmulInt8Backend
        MatmulInt8Backend.BINS = BINS_OVERRIDE
    except Exception:
        pass
    sim = NpuSimulator(profile)
    B = BACKENDS[name](profile, sim)
    rows = []
    t_total = time.perf_counter()
    for seed in seeds:
        for cx in range(chunks):
            for cz in range(chunks):
                coords, dims = lattice_coords(cx * 16, -64, cz * 16)
                t0 = time.perf_counter()
                ch = B.channels(coords, seed, CHANNEL_ORDER)
                d = density_field(ch, coords)
                el = (time.perf_counter() - t0) * 1000
                lx, ly, lz = dims
                h = diff.heights(d, lx, ly, lz)
                rows.append({
                    "seed": seed, "chunk": (cx, cz), "region": classify(h),
                    "ms": el, "density": d, "height": h,
                })
    return rows, (time.perf_counter() - t_total) * 1000, B, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="compare",
                    choices=["vanilla", "fp32", "direct-int8", "matmul-int8", "compare"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--profile", default="SNAPDRAGON_8S_GEN3")
    ap.add_argument("--compare", action="store_true",
                    help="run every backend against the vanilla oracle")
    ap.add_argument("--json")
    ap.add_argument("--csv")
    ap.add_argument("--bins", type=int, default=16,
                    help="fractional-position bins per axis for matmul-int8")
    a = ap.parse_args()

    global BINS_OVERRIDE
    BINS_OVERRIDE = a.bins
    seeds = [1000 + i * 7919 for i in range(a.seeds)]

    if a.compare or a.backend == "compare":
        names = ["direct-int8", "matmul-int8"]
        ref_rows, ref_ms, _, _ = run_backend("fp32", seeds, a.chunks, a.profile)
        print(f"Vanilla oracle: {len(ref_rows)} chunks in {ref_ms:.0f}ms "
              f"({ref_ms/max(1,len(ref_rows)):.2f} ms/chunk)")
        print()
        results = {}
        for n in names:
            rows, ms, B, sim = run_backend(n, seeds, a.chunks, a.profile)
            stats = []
            for r, rr in zip(rows, ref_rows):
                lx, ly, lz = 5, 49, 5
                s = diff.compare(rr["density"], r["density"], lx, ly, lz)
                s["chunk"] = r["chunk"]
                s["region"] = r["region"]
                stats.append(s)
            agg = diff.merge(stats)
            agg["backend"] = n
            agg["total_ms"] = ms
            agg["ms_per_chunk"] = ms / max(1, len(rows))
            agg["speedup"] = ref_ms / ms if ms > 0 else 0.0
            agg["backend_stats"] = B.stats()
            agg["sim"] = sim.summary()
            results[n] = agg
            print(f"--- {n} ---")
            print(f"  Density MAE      {agg['density_mae']:.6f}   RMSE {agg['density_rmse']:.6f}   MAX {agg['density_max_worst']:.6f}")
            print(f"  Sign agreement   {agg['sign_agreement']*100:.3f}%   mismatch {agg['sign_mismatch']*100:.3f}%")
            print(f"  Height MAE       {agg['height_mae']:.3f} blocks   MAX {agg['height_max_worst']:.1f}")
            print(f"  Vanilla          {ref_ms/max(1,len(rows)):.2f} ms/chunk")
            print(f"  {n:<15} {agg['ms_per_chunk']:.2f} ms/chunk")
            print(f"  Speedup          {agg['speedup']:.2f}x")
            print(f"  Backend          {agg['backend_stats']}")
            print()

        if a.json:
            slim = {k: {kk: vv for kk, vv in v.items()
                        if not isinstance(vv, (np.ndarray,))}
                    for k, v in results.items()}
            with open(a.json, "w") as f:
                json.dump(slim, f, indent=2)
            print("wrote", a.json)
        return

    rows, ms, B, sim = run_backend(
        "fp32" if a.backend == "vanilla" else a.backend, seeds, a.chunks, a.profile)
    print(f"{a.backend}: {len(rows)} chunks, {ms:.0f}ms total, "
          f"{ms/max(1,len(rows)):.2f} ms/chunk")
    print(f"  {B.stats()}")
    print(f"  {sim.summary()}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed", "cx", "cz", "region", "ms", "height_mean"])
            for r in rows:
                w.writerow([r["seed"], r["chunk"][0], r["chunk"][1], r["region"],
                            f"{r['ms']:.3f}", f"{np.mean(r['height']):.2f}"])
        print("wrote", a.csv)


if __name__ == "__main__":
    sys.exit(main())
