"""
Worldgen on top of the backends.

Three strategies exist so that two different kinds of error can be told apart:

  fp32        - vanilla maths, no device limits at all. The control.
  direct-int8 - vanilla maths, but the channel values pass through INT8, as they would
                inside a real DSP noise kernel. Measures the quantisation floor: the best
                ANY kernel could do on this device.
  matmul-int8 - expressed as matmuls the current backend can actually run. Measures what
                is achievable today, and - via group size - whether the grouping
                degenerates as octaves get higher.

Comparing fp32 vs direct-int8 isolates quantisation error.
Comparing direct-int8 vs matmul-int8 isolates the cost of forcing it through matmul.

That split is the point: without it, a bad result would tell us nothing about which of
the two caused it.

NOTE: the density function here is a structural stand-in, not vanilla's full final_density
with its splines. It is deliberately monotone in the noise channels so that sign and
height are meaningful error metrics. It measures how noise error propagates to terrain -
which is the question this tool exists to answer - not whether our terrain is vanilla.
"""

import numpy as np
from vanilla import NormalNoise, CHANNELS
from simulator import quantize_int8

STEP_X, STEP_Y, STEP_Z = 4, 8, 4
SIZE_X, SIZE_Y, SIZE_Z = 16, 384, 16
SEA_LEVEL = 63


def lattice_coords(ox, oy, oz):
    lx = (SIZE_X - 1) // STEP_X + 2
    ly = (SIZE_Y - 1) // STEP_Y + 2
    lz = (SIZE_Z - 1) // STEP_Z + 2
    xs = ox + np.arange(lx) * STEP_X
    ys = oy + np.arange(ly) * STEP_Y
    zs = oz + np.arange(lz) * STEP_Z
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return (np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1),
            (lx, ly, lz))


class FP32Backend:
    """Vanilla maths, unquantised. The reference the other two are measured against."""

    name = "fp32"

    def __init__(self, profile=None, sim=None):
        self.cache = {}

    def _noise(self, seed, ch):
        key = (seed, ch)
        if key not in self.cache:
            fo, mods, amp = CHANNELS[ch]
            self.cache[key] = NormalNoise(seed, fo, mods, amp)
        return self.cache[key]

    def channels(self, coords, seed, names):
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        return {n: self._noise(seed, n).value(x, y, z) for n in names}

    def stats(self):
        return "control: no device limits"


class DirectInt8Backend(FP32Backend):
    """
    As a real DSP noise kernel would behave: correct maths, INT8 in and out.

    This is the quantisation floor - if an algorithm cannot beat this, no amount of
    kernel work will save it.
    """

    name = "direct-int8"

    def channels(self, coords, seed, names):
        raw = super().channels(coords, seed, names)
        out = {}
        for n, v in raw.items():
            q, spec = quantize_int8(v)
            out[n] = q.astype(np.float64) * spec.scale
        return out

    def stats(self):
        return "quantisation floor (correct maths, int8 values)"


class MatmulInt8Backend(FP32Backend):
    """
    Noise expressed as matmuls the current backend can run.

    Perlin is v = sum_k w_k(f) * d_k where d_k = g_k . (p - c_k). Only the second step
    is a matmul, and only when w_k is shared - so points are grouped by their fractional
    position f (binned) and, within a bin, v = D[g,8] @ W[8,1].

    Two costs are measured honestly:
      - binning f is an approximation, and its error is reported
      - grouping degenerates when the cell size is smaller than the lattice spacing,
        which happens at high octaves; average group size is reported so the collapse
        is visible rather than hidden in a total
    """

    name = "matmul-int8"
    BINS = 4          # fractional-position bins per axis
    MIN_BUCKET = 32

    def __init__(self, profile=None, sim=None):
        super().__init__()
        self.groups = []
        self.max_group = 0
        self.profile = profile
        self.sim = sim
        self.rejected = 0
        self.last_reject = "none"

    def channels(self, coords, seed, names):
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        out = {}
        for n in names:
            noise = self._noise(seed, n)
            total = np.zeros(len(coords))
            for o, lvl in enumerate(noise.levels):
                if lvl is None or noise.amps[o] == 0.0:
                    continue
                freq = 2.0 ** (noise.first_octave + o)
                px, py, pz = x * freq, y * freq, z * freq
                total += self._octave_matmul(lvl, px, py, pz) * noise.amps[o]
            out[n] = total / noise.normalization
        return out

    def _octave_matmul(self, lvl, px, py, pz):
        fx = px + lvl.xo
        fy = py + lvl.yo
        fz = pz + lvl.zo
        ix = np.floor(fx).astype(np.int64)
        iy = np.floor(fy).astype(np.int64)
        iz = np.floor(fz).astype(np.int64)
        dx, dy, dz = fx - ix, fy - iy, fz - iz

        # d_k = g_k . (p - c_k) for the 8 corners
        D = np.empty((len(px), 8))
        for k in range(8):
            # _hash takes the cell base, not the corner: the corner offset is applied
            # inside via the k bits. Passing ix+(k>>2) here applied the z offset twice,
            # once through hz and again through az.
            idx = self._hash(lvl, ix, iy, iz)
            # Each corner gets its own offset: corner (cx,cy,cz) is sampled at
            # (dx-cx, dy-cy, dz-cz). Using the raw fractional coordinate for all eight
            # was wrong and showed up as a large error that no amount of binning
            # precision could fix - exactly the kind of bug this bench exists to catch
            # before it reaches a device.
            oz, oy, ox = (k & 1), ((k >> 1) & 1), ((k >> 2) & 1)
            px_, py_, pz_ = dx - ox, dy - oy, dz - oz
            D[:, k] = (self._grad(idx[:, k], 0, px_, py_, pz_)
                       + self._grad(idx[:, k], 1, px_, py_, pz_))

        # Bin the fractional position so w becomes shared within a bin.
        bx = np.minimum((dx * self.BINS).astype(np.int64), self.BINS - 1)
        by = np.minimum((dy * self.BINS).astype(np.int64), self.BINS - 1)
        bz = np.minimum((dz * self.BINS).astype(np.int64), self.BINS - 1)
        bid = (bx * self.BINS + by) * self.BINS + bz

        vals = np.zeros(len(px))
        for b in np.unique(bid):
            sel = np.nonzero(bid == b)[0]
            self.groups.append(len(sel))
            self.max_group = max(self.max_group, len(sel))
            g = len(sel)
            pm = self.profile.bucketize(g) if self.profile else max(g, self.MIN_BUCKET)
            pk = self.profile.bucketize(8) if self.profile else 8
            pn = self.profile.bucketize(1) if self.profile else 1
            if pm is None or pk is None or pn is None:
                self.rejected += 1
                continue
            # Bin centre, not the first point in the bin: the first point sits at the
            # bin's low edge, which biases every weight in the same direction.
            bcx = (b % (self.BINS * self.BINS * self.BINS)) // (self.BINS * self.BINS)
            bcy = (b // self.BINS) % self.BINS
            bcz = b % self.BINS
            f = np.array([(bcx + 0.5) / self.BINS,
                          (bcy + 0.5) / self.BINS,
                          (bcz + 0.5) / self.BINS])
            w = self._weights(f)                      # (8,)
            # Pad to legal buckets and run through the simulator, so the submission is
            # checked, quantised and timed exactly as the device would do it. Padding
            # 8 -> pk and 1 -> pn is real waste the hardware pays for; hiding it would
            # make this backend look far cheaper than it can ever be.
            Dpad = np.zeros((pm, pk))
            Dpad[:g, :8] = D[sel]
            wpad = np.zeros((pk, pn))
            wpad[:8, :1] = w.reshape(8, 1)
            if self.sim is not None:
                r = self.sim.matmul(Dpad, wpad)
                if not r.ok:
                    self.rejected += 1
                    self.last_reject = r.reason
                    continue
                vals[sel] = r.out[:g, 0]
            else:
                vals[sel] = Dpad[:g, :8] @ wpad[:8, :1]
        return vals

    def _hash(self, lvl, hx, hy, hz):
        p = lvl.p
        X, Y, Z = hx & 255, hy & 255, hz & 255
        corners = np.empty((len(hx), 8), dtype=np.int64)
        for k in range(8):
            # Bit order matters and is easy to get backwards. Reading vanilla's naming:
            #   x bit -> p[X]   vs p[X+1]        (a0 vs a1)
            #   z bit -> p[a]   vs p[a+1]        (b0/b2 vs b1/b3)
            #   y bit -> +Z     vs +Z+1          (b0 vs b0+1)
            # An earlier version swapped the x and z roles; the resulting error looked
            # like a quantisation problem and did not move when binning precision was
            # increased, which is how it was found.
            az = (k & 1)
            ay = (k >> 1) & 1
            ax = (k >> 2) & 1
            a = p[X + ax] + Y + az
            corners[:, k] = p[p[a] + Z + ay]
        return corners

    def _grad(self, idx, axis, dx, dy, dz):
        """Gradient dot product component. idx is (M,8)."""
        h = idx & 15
        if axis == 0:
            u = np.where(h < 8, dx, dy)
            return np.where((h & 1) == 0, u, -u)
        if axis == 1:
            v = np.where(h < 4, dy, np.where((h == 12) | (h == 14), dx, dz))
            return np.where((h & 2) == 0, v, -v)
        return np.zeros_like(dx)

    def _weights(self, f):
        fx, fy, fz = f
        ux, uy, uz = fade(fx), fade(fy), fade(fz)
        w = np.empty(8)
        for k in range(8):
            wz = uz if (k & 1) else 1 - uz
            wy = uy if ((k >> 1) & 1) else 1 - uy
            wx = ux if ((k >> 2) & 1) else 1 - ux
            w[k] = wx * wy * wz
        return w

    def stats(self):
        if not self.groups:
            return "no groups"
        g = np.array(self.groups)
        return (f"groups={len(g)} avg_size={g.mean():.1f} "
                f"singleton_frac={(g == 1).mean():.1%} "
                f"rejected={self.rejected} last_reject={self.last_reject}")


def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


CHANNEL_ORDER = ["continentalness", "erosion", "temperature", "weirdness"]

BACKENDS = {
    "fp32": FP32Backend,
    "direct-int8": DirectInt8Backend,
    "matmul-int8": MatmulInt8Backend,
}


def density_field(ch, coords):
    """
    Structural stand-in for final_density: monotone in the noise channels with a
    height gradient, so that density sign and surface height are both meaningful.
    """
    cont = ch["continentalness"]
    ero = ch["erosion"]
    temp = ch["temperature"]
    weird = ch["weirdness"]
    # Evaluated on the lattice itself: one density per lattice point, so the result is
    # (M,) and can be reshaped to (lx, ly, lz) for surface extraction.
    ys = coords[:, 1]
    grad = -(ys - SEA_LEVEL) / 128.0
    lat = (0.55 * cont + 0.25 * ero + 0.10 * temp + 0.10 * weird)
    return lat + grad
