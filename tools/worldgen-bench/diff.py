"""
Differential comparison.

Optimisation priority, in order:
  1. sign agreement   - a density sign flip changes terrain topology (a cave that is not
                        there, or solid rock where air should be). Nothing else matters
                        if this is wrong.
  2. height agreement - what the player actually sees.
  3. density MAE/RMSE - finer-grained agreement.
  4. performance      - never buy tiny float accuracy with a lot of speed.
"""

import numpy as np
from worldgen import STEP_Y


def surface_height(density_lat, ly):
    """Highest solid y per (x,z) column. density_lat is (lx*ly*lz,) as (lx,ly,lz)."""
    d = density_lat.reshape(-1, ly, density_lat.size // (ly * (density_lat.size // ly // ly)) if False else -1)
    return None


def heights(density_flat, lx, ly, lz):
    d = density_flat.reshape(lx, ly, lz)
    solid = d > 0
    out = np.full((lx, lz), -1, dtype=np.float64)
    for ix in range(lx):
        for iz in range(lz):
            col = np.nonzero(solid[ix, :, iz])[0]
            if col.size:
                out[ix, iz] = col[-1] * STEP_Y - 64
    return out


def compare(d_ref, d_test, lx, ly, lz):
    err = d_test - d_ref
    abs_err = np.abs(err)
    sign_ref = np.sign(d_ref)
    sign_test = np.sign(d_test)
    agree = (sign_ref == sign_test)
    # Zero is a boundary case with no topology meaning; exclude it from the rate.
    nonzero = d_ref != 0

    h_ref = heights(d_ref, lx, ly, lz)
    h_test = heights(d_test, lx, ly, lz)
    h_err = np.abs(h_test - h_ref)

    worst = int(np.argmax(abs_err))
    return {
        "density_mae": float(abs_err.mean()),
        "density_rmse": float(np.sqrt((err ** 2).mean())),
        "density_max": float(abs_err.max()),
        "sign_agreement": float(agree[nonzero].mean()) if nonzero.any() else 1.0,
        "sign_mismatch": float((~agree[nonzero]).mean()) if nonzero.any() else 0.0,
        "height_mae": float(h_err.mean()),
        "height_max": float(h_err.max()),
        "worst_idx": worst,
        "worst_err": float(abs_err[worst]),
    }


def merge(rows):
    """Aggregate per-chunk results into one report."""
    if not rows:
        return {}
    keys = ["density_mae", "density_rmse", "density_max", "sign_agreement",
            "sign_mismatch", "height_mae", "height_max", "worst_err"]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows])
        out[k] = float(v.mean())
        out[k + "_p95"] = float(np.percentile(v, 95))
    out["density_max_worst"] = float(max(r["density_max"] for r in rows))
    out["height_max_worst"] = float(max(r["height_max"] for r in rows))
    out["chunks"] = len(rows)
    wi = int(np.argmax([r["worst_err"] for r in rows]))
    out["worst_chunk"] = rows[wi].get("chunk")
    out["worst_sample_err"] = rows[wi]["worst_err"]
    return out
