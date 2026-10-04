"""
End-to-end model of the assist path, including transport.

assist.py modelled the kernel in isolation. That is not enough to pick a protocol:
the wire cost of a 16-chunk batch is large enough to matter, and it depends entirely
on how the request is encoded. This models the whole round trip so the encoding can be
chosen on numbers instead of preference.
"""

import numpy as np
from vanilla import NormalNoise, CHANNELS
from simulator import quantize_int8
from worldgen import lattice_coords, density_field, CHANNEL_ORDER
import profile as P

# Loopback throughput. Two bounds because we have not measured it and the answer
# changes the recommendation: at 100 MB/s the wire dominates, at 2 GB/s it does not.
BPS_PESSIMISTIC = 100e6
BPS_OPTIMISTIC = 2e9

# A device-side buffer ceiling. Real value unknown; 1 MB is a conservative guess and
# the sweep reports when it binds.
MAX_PAYLOAD_BYTES = 1 << 20


class FastRpcSimBackend:
    """
    Models: encode request -> send -> kernel -> encode result -> receive.

    The kernel maths is the same as assist.py (correct noise). What is modelled here
    on top of that is the wire.
    """

    def __init__(self, prof, encoding="params", out_dtype="int8",
                 bps=BPS_OPTIMISTIC, ns_per_op=P.NS_PER_OP_EFFICIENT,
                 max_payload=MAX_PAYLOAD_BYTES):
        self.profile = prof
        self.encoding = encoding
        self.out_dtype = out_dtype
        self.bps = bps
        self.max_payload = max_payload
        self.per_unit_us = P.OPS_PER_NOISE_UNIT * ns_per_op / 1000.0
        self.cache = {}
        self.calls = 0
        self.rejected = 0
        self.last_reject = "none"
        self.latency_us = 0.0
        self.wire_us = 0.0
        self.kernel_us = 0.0
        self.bytes_in = 0
        self.bytes_out = 0
        self.units = 0
        self.padded_units = 0

    def _noise(self, seed, ch):
        key = (seed, ch)
        if key not in self.cache:
            fo, mods, amp = CHANNELS[ch]
            self.cache[key] = NormalNoise(seed, fo, mods, amp)
        return self.cache[key]

    @staticmethod
    def effective_octaves(names):
        return sum(1 for n in names for m in CHANNELS[n][1] if m != 0)

    # ---- wire encodings ---------------------------------------------------
    def request_bytes(self, n_chunks, n_pts):
        """
        Two encodings, and the gap between them is the whole point.

        'coords' ships every lattice point as 3 floats.
        'params' ships the lattice definition - chunk coords, minY, step - and lets the
        kernel generate the points itself. The lattice is perfectly regular (step 4/8/4),
        so nothing is lost by describing it instead of listing it.
        """
        if self.encoding == "coords":
            return n_chunks * n_pts * 3 * 4
        return 32 + n_chunks * 8           # global header + (chunkX, chunkZ) per chunk

    def result_bytes(self, n_chunks, n_pts, n_ch):
        if self.out_dtype == "fp32":
            return n_chunks * n_pts * n_ch * 4
        # int8 payload + one scale per channel per chunk
        return n_chunks * n_pts * n_ch + n_chunks * n_ch * 4

    # ---- planning ---------------------------------------------------------
    def _plan_calls(self, m):
        maxm = max(self.profile.buckets)
        calls, rem = [], m
        while rem > 0:
            take = min(rem, maxm)
            calls.append((take, self.profile.bucketize(take)))
            rem -= take
        return calls

    # ---- the round trip ---------------------------------------------------
    def eval_batch(self, coords_list, seed, names):
        allc = np.vstack(coords_list)
        m = len(allc)
        n_pts = len(coords_list[0])
        oct_ = self.effective_octaves(names)
        pn = self.profile.bucketize(oct_)
        pk = 32

        # 1. encode + send request
        b_in = self.request_bytes(len(coords_list), n_pts)
        if b_in > self.max_payload:
            self.rejected += 1
            self.last_reject = f"REQ_TOO_LARGE ({b_in})"
            return None
        self.bytes_in += b_in
        t_in = b_in / self.bps * 1e6

        # 2. kernel
        x, y, z = allc[:, 0], allc[:, 1], allc[:, 2]
        raw = {n: self._noise(seed, n).value(x, y, z) for n in names}
        out = {n: np.empty(m) for n in names}
        off = 0
        k_us = 0.0
        fixed_us = 0.0
        for take, pm in self._plan_calls(m):
            ok, reason = self.profile.check_shape(pm, pk, pn)
            if not ok:
                self.rejected += 1
                self.last_reject = reason
                return None
            self.calls += 1
            self.units += take * oct_
            self.padded_units += pm * oct_
            k_us += pm * oct_ * self.per_unit_us
            # One fixed cost per call, accumulated per call. Using the running total
            # instead (self.calls) multiplies every later batch by its ordinal and
            # roughly triples modelled latency - it made the pessimistic case look
            # like a loss when it was not.
            fixed_us += self.profile.latency.FIXED_US
            for n in names:
                sl = raw[n][off:off + take]
                if self.out_dtype == "int8":
                    q, spec = quantize_int8(sl)
                    out[n][off:off + take] = q * spec.scale
                else:
                    out[n][off:off + take] = sl
            off += take

        # 3. encode + receive result
        b_out = self.result_bytes(len(coords_list), n_pts, len(names))
        if b_out > self.max_payload:
            self.rejected += 1
            self.last_reject = f"RESULT_TOO_LARGE ({b_out})"
            return None
        self.bytes_out += b_out
        t_out = b_out / self.bps * 1e6

        self.kernel_us += k_us
        self.wire_us += t_in + t_out
        self.latency_us += fixed_us + t_in + k_us + t_out

        res, off = [], 0
        for c in coords_list:
            n = len(c)
            res.append({k: out[k][off:off + n] for k in names})
            off += n
        return res

    def stats(self):
        return (f"calls={self.calls} total={self.latency_us/1000:.1f}ms "
                f"(kernel {self.kernel_us/1000:.1f} / wire {self.wire_us/1000:.1f}) "
                f"out={self.bytes_out//1024}KB pad={self.padded_units/max(1,self.units):.2f}x")
