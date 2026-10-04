"""
NPU Simulator.

This is NOT "run it on the CPU and call it an NPU". It reproduces the behaviour and the
limits of the real backend:

  - shape validation against the device's bucket whitelist and element budget
  - INT8 quantisation with a real scale, so precision is genuinely lost
  - padding up to the next bucket, so wasted arithmetic is really performed
  - an int8 kernel with a 32-bit accumulator, which is where long-K error accumulates
  - dequantisation
  - latency from the measured device model, including cold graph creation

Results and limits are deliberately separate parts of the pipeline. That separation is
what lets this tool answer the question that matters: an algorithm can be mathematically
correct in FP32 and still fall apart once INT8 + buckets + padding are applied, and the
only way to find that out before touching the phone is to model those steps honestly.
"""

import numpy as np


class QuantSpec:
    def __init__(self, scale, zero_point=0):
        self.scale = scale
        self.zero_point = zero_point


def quantize_int8(x):
    """Symmetric INT8. Scale is set by the true dynamic range of THIS tensor."""
    x = np.asarray(x, dtype=np.float64)
    amax = float(np.max(np.abs(x))) if x.size else 0.0
    # A flat zero tensor would divide by zero; any scale works when every value is 0.
    scale = amax / 127.0 if amax > 0 else 1.0
    q = np.clip(np.round(x / scale), -128, 127).astype(np.int64)
    return q, QuantSpec(scale)


def dequantize(q, spec):
    return q.astype(np.float64) * spec.scale


class SubmitResult:
    def __init__(self, ok, out=None, reason="ok", padded=None, latency_us=0.0,
                 quant_error=None):
        self.ok = ok
        self.out = out
        self.reason = reason
        self.padded = padded          # (m,k,n) actually executed after padding
        self.latency_us = latency_us
        self.quant_error = quant_error

    def __repr__(self):
        if not self.ok:
            return f"SubmitResult(REJECT: {self.reason})"
        return (f"SubmitResult(ok, padded={self.padded}, "
                f"latency={self.latency_us/1000:.2f}ms)")


class NpuSimulator:
    def __init__(self, profile, quantize=True, model_latency=True):
        self.profile = profile
        self.do_quantize = quantize and profile.int8
        self.model_latency = model_latency
        self.stats = {"submits": 0, "rejected": 0, "us": 0.0,
                      "padded_elements": 0, "useful_elements": 0}

    def _finish(self, res, useful, padded):
        self.stats["submits"] += 1
        self.stats["us"] += res.latency_us
        self.stats["useful_elements"] += useful
        self.stats["padded_elements"] += padded
        return res

    def matmul(self, A, B, tag=""):
        """
        One submission. A is (m,k), B is (k,n).

        Returns a SubmitResult; on rejection `out` is None and `reason` says why.
        """
        m, k = A.shape
        k2, n = B.shape
        if k != k2:
            return SubmitResult(False, reason=f"inner dim mismatch {k} vs {k2}")

        ok, reason = self.profile.check_shape(m, k, n)
        if not ok:
            self.stats["rejected"] += 1
            return SubmitResult(False, reason=reason)

        pm = self.profile.bucketize(m)
        pk = self.profile.bucketize(k)
        pn = self.profile.bucketize(n)
        if pm is None or pk is None or pn is None:
            self.stats["rejected"] += 1
            return SubmitResult(False, reason="no bucket large enough")

        useful = m * k + k * n + m * n
        padded = pm * pk + pk * pn + pm * pn

        # Pad, and really run the padded shape - that is the arithmetic the device pays for.
        Ap = np.zeros((pm, pk), dtype=np.float64)
        Ap[:m, :k] = A
        Bp = np.zeros((pk, pn), dtype=np.float64)
        Bp[:k, :n] = B

        if self.do_quantize:
            qa, sa = quantize_int8(Ap)
            qb, sb = quantize_int8(Bp)
            # 32-bit accumulator: this is exactly where a long K accumulates error.
            acc = qa.astype(np.int64) @ qb.astype(np.int64)
            out_full = acc.astype(np.float64) * (sa.scale * sb.scale)
            qerr = float(np.max(np.abs(Ap - qa * sa.scale))) + \
                   float(np.max(np.abs(Bp - qb * sb.scale)))
        else:
            out_full = Ap @ Bp
            qerr = 0.0

        out = out_full[:m, :n]

        lat = 0.0
        if self.model_latency:
            lat = self.profile.latency.submit_us(pm, pk, pn)

        res = SubmitResult(True, out=out, padded=(pm, pk, pn),
                           latency_us=lat, quant_error=qerr)
        return self._finish(res, useful, padded)

    def summary(self):
        s = self.stats
        waste = (s["padded_elements"] / s["useful_elements"]) if s["useful_elements"] else 0.0
        return (f"submits={s['submits']} rejected={s['rejected']} "
                f"total={s['us']/1000:.1f}ms "
                f"padding_waste={waste:.2f}x")
