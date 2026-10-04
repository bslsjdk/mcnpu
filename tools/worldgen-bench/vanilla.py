"""
Vanilla Oracle - Minecraft 26.3 noise, reproduced exactly.

This is the reference the whole benchmark measures against. It is a faithful port of
NpuNoise.java (which itself was read off the vanilla bytecode), down to Java's 64-bit
overflow semantics - because a "close enough" oracle would make every later error number
meaningless.

Deliberately FP64 with no quantisation anywhere: the oracle must be the most accurate
answer available, so that every difference reported later is attributable to the NPU
path and not to the reference being sloppy.
"""

import numpy as np

MASK64 = (1 << 64) - 1


def i64(x):
    x &= MASK64
    return x - (1 << 64) if x >= (1 << 63) else x


def rotl(x, n):
    x &= MASK64
    return ((x << n) | (x >> (64 - n))) & MASK64


class Rng:
    """Xoroshiro128++ exactly as Minecraft's XoroshiroRandomSource drives it."""

    def __init__(self, seed):
        self.set_seed(seed)

    def set_seed(self, seed):
        if seed == 0:
            seed = -1
        self.lo = i64(seed ^ 0x6A09E667F3BCC909)
        self.hi = i64(seed + 0x9E3779B97F4A7C15)

    def next_long(self):
        l, h = self.lo & MASK64, self.hi & MASK64
        result = i64(rotl(i64(l + h), 17) + l)
        h ^= l
        self.lo = i64(rotl(l, 49) ^ h ^ ((h << 21) & MASK64))
        self.hi = i64(rotl(h, 28))
        return result

    def next_int(self, n):
        if n <= 0:
            raise ValueError
        return (self.next_long() & MASK64) >> 1 % n if False else int(((self.next_long() & MASK64) >> 1) % n)

    def next_double(self):
        return ((self.next_long() & MASK64) >> 11) * (2.0 ** -53)


class PerlinNoise:
    def __init__(self, rng):
        self.xo = rng.next_double() * 256.0
        self.yo = rng.next_double() * 256.0
        self.zo = rng.next_double() * 256.0
        perm = np.arange(256, dtype=np.int64)
        for i in range(256):
            j = rng.next_int(256 - i)
            perm[i], perm[j + i] = perm[j + i], perm[i]
        self.p = np.empty(512, dtype=np.int64)
        self.p[:256] = perm & 255
        self.p[256:] = perm & 255

    def value(self, x, y, z):
        """Vectorised: accepts arrays, returns array."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        p = self.p
        fx = x + self.xo
        fy = y + self.yo
        fz = z + self.zo
        ix = np.floor(fx).astype(np.int64)
        iy = np.floor(fy).astype(np.int64)
        iz = np.floor(fz).astype(np.int64)
        dx = fx - ix
        dy = fy - iy
        dz = fz - iz
        X = ix & 255
        Y = iy & 255
        Z = iz & 255
        u = fade(dx)
        v = fade(dy)
        w = fade(dz)

        a0 = p[X] + Y
        a1 = p[X + 1] + Y
        b0 = p[a0] + Z
        b1 = p[a0 + 1] + Z
        b2 = p[a1] + Z
        b3 = p[a1 + 1] + Z

        g000 = grad(p[b0], dx, dy, dz)
        g001 = grad(p[b1], dx, dy, dz - 1)
        g010 = grad(p[b0 + 1], dx, dy - 1, dz)
        g011 = grad(p[b1 + 1], dx, dy - 1, dz - 1)
        g100 = grad(p[b2], dx - 1, dy, dz)
        g101 = grad(p[b3], dx - 1, dy, dz - 1)
        g110 = grad(p[b2 + 1], dx - 1, dy - 1, dz)
        g111 = grad(p[b3 + 1], dx - 1, dy - 1, dz - 1)

        x00 = lerp(u, g000, g100)
        x10 = lerp(u, g010, g110)
        x01 = lerp(u, g001, g101)
        x11 = lerp(u, g011, g111)
        y0 = lerp(v, x00, x10)
        y1 = lerp(v, x01, x11)
        return lerp(w, y0, y1)


def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def lerp(t, a, b):
    return a + t * (b - a)


def grad(h, x, y, z):
    h = h & 15
    u = np.where(h < 8, x, y)
    v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
    return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)


class NormalNoise:
    """A stack of octaves, exactly vanilla's NormalNoise."""

    def __init__(self, seed, first_octave, amplitudes_spec, base_amplitude):
        n = len(amplitudes_spec)
        parity = (2.0 ** (n - 1)) / (2.0 ** n - 1.0)
        self.first_octave = first_octave
        self.amps = np.zeros(n)
        self.levels = [None] * n
        rng = Rng(seed)
        for o in range(n):
            mod = amplitudes_spec[o]
            # A zero modifier consumes no Perlin and does not advance the shared Rng -
            # getting this wrong shifts every later octave for that channel.
            if mod == 0.0:
                self.amps[o] = 0.0
                continue
            self.amps[o] = base_amplitude * parity * mod
            self.levels[o] = PerlinNoise(rng)
        total = float(self.amps.sum())
        self.normalization = 1.0 if total == 0.0 else total

    def value(self, x, y, z):
        total = np.zeros_like(np.asarray(x, dtype=np.float64))
        for o, lvl in enumerate(self.levels):
            if lvl is None or self.amps[o] == 0.0:
                continue
            freq = 2.0 ** (self.first_octave + o)
            total += lvl.value(x * freq, y * freq, z * freq) * self.amps[o]
        return total / self.normalization


# Vanilla channel parameters, read from data/minecraft/worldgen/noise/*.json
CHANNELS = {
    "continentalness": (-9, [1, 1, 2, 2, 2, 1, 1, 1, 1], 0.8880832896205223),
    "erosion":         (-9, [1, 1, 0, 1, 1],             1.063180125160734),
    "temperature":     (-10, [1.5, 0, 1, 0, 0, 0],       1.2453007926713473),
    "humidity":        (-8, [1, 1, 0, 1, 1],             0.9815664666103303),
    "weirdness":       (-7, [1, 1, 2, 1],                0.8008025445443292),
    "jaggedness":      (-7, [1, 2, 1, 0, 0],             0.9387857679500631),
}
