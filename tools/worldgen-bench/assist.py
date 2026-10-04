"""
NPU assist: batched noise evaluation through a kernel-style call.

This is the throughput answer to the matmul result. Matmul lost not because the maths
was wrong but because one chunk needed ~60,907 tiny submissions, each paying the
device's ~2.5 ms fixed cost. A kernel does not have that problem: it takes a block of
points and returns a block of values, so the fixed cost is paid once per block rather
than once per group.

The two things that decide throughput here are:
  - how many points fit in one call (the m bucket ceiling)
  - how many chunks are aggregated into one call (amortises the fixed cost)

Both are swept in sweep.py.
"""

import numpy as np
from vanilla import NormalNoise, CHANNELS
from simulator import quantize_int8
import profile as P


class AssistFrpcBackend:
    """
    Models a FastRPC noise kernel on Hexagon.

    Correct maths (that is what a kernel would implement), real INT8 output
    quantisation, real bucket padding, and latency from the measured device model.
    """

    name = "assist-frpc"

    def __init__(self, prof, sim=None, int8_out=True,
                 ns_per_op=P.NS_PER_OP_EFFICIENT):
        self.profile = prof
        self.sim = sim
        self.int8_out = int8_out
        self.per_unit_us = P.OPS_PER_NOISE_UNIT * ns_per_op / 1000.0
        self.cache = {}
        self.calls = 0
        self.rejected = 0
        self.last_reject = "none"
        self.latency_us = 0.0
        self.units = 0
        self.padded_units = 0

    # ---- noise ------------------------------------------------------------
    def _noise(self, seed, ch):
        key = (seed, ch)
        if key not in self.cache:
            fo, mods, amp = CHANNELS[ch]
            self.cache[key] = NormalNoise(seed, fo, mods, amp)
        return self.cache[key]

    @staticmethod
    def effective_octaves(names):
        """Octaves that actually evaluate; a zero amplitude modifier consumes none."""
        return sum(1 for n in names for m in CHANNELS[n][1] if m != 0)

    # ---- planning ---------------------------------------------------------
    def _plan_calls(self, m):
        maxm = max(self.profile.buckets)
        calls, rem = [], m
        while rem > 0:
            take = min(rem, maxm)
            pm = self.profile.bucketize(take)
            calls.append((take, pm))
            rem -= take
        return calls

    # ---- the call ---------------------------------------------------------
    def eval_batch(self, coords_list, seed, names):
        """One kernel call covering several chunks. Returns a list of channel dicts."""
        allc = np.vstack(coords_list)
        m = len(allc)
        oct_ = self.effective_octaves(names)
        pn = self.profile.bucketize(oct_)
        pk = 32

        x, y, z = allc[:, 0], allc[:, 1], allc[:, 2]
        raw = {n: self._noise(seed, n).value(x, y, z) for n in names}

        out = {n: np.empty(m) for n in names}
        off = 0
        for take, pm in self._plan_calls(m):
            ok, reason = self.profile.check_shape(pm, pk, pn)
            if not ok:
                self.rejected += 1
                self.last_reject = reason
                return None
            self.calls += 1
            self.units += take * oct_
            self.padded_units += pm * oct_
            # Fixed cost once per call, kernel cost on the padded size.
            self.latency_us += (self.profile.latency.FIXED_US
                                + pm * oct_ * self.per_unit_us)
            for n in names:
                sl = raw[n][off:off + take]
                if self.int8_out:
                    q, spec = quantize_int8(sl)
                    out[n][off:off + take] = q * spec.scale
                else:
                    out[n][off:off + take] = sl
            off += take

        # Split back per chunk.
        res, off = [], 0
        for c in coords_list:
            n_pts = len(c)
            res.append({n: out[n][off:off + n_pts] for n in names})
            off += n_pts
        return res

    def stats(self):
        waste = self.padded_units / self.units if self.units else 0.0
        return (f"calls={self.calls} rejected={self.rejected} "
                f"latency={self.latency_us/1000:.1f}ms "
                f"padding_waste={waste:.2f}x")


class AssistFrpcF32Backend(AssistFrpcBackend):
    """Same kernel, FP32 output - the optimistic bound if HVX runs float."""
    name = "assist-frpc-f32"

    def __init__(self, prof, sim=None, ns_per_op=P.NS_PER_OP_EFFICIENT):
        super().__init__(prof, sim, int8_out=False, ns_per_op=ns_per_op)
