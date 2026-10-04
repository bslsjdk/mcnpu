"""
NPU device profiles.

These numbers are not invented - they are the constraints measured on the actual
Snapdragon 8s Gen 3 QNN HTP backend, written down so the simulator behaves like the
real thing instead of like an idealised accelerator.

The point of a profile is that a shape which would be rejected on the phone is rejected
here too, and a shape which is slow there is slow here. Otherwise an algorithm tuned
against the simulator would fall over the moment it met real hardware.
"""

BUCKETS_8S_GEN3 = [32, 64, 128, 256, 512, 1024]

# 16 was measured and does NOT behave like a normal bucket: it produced a scale
# mismatch. Listing it as allowed would let an algorithm pass here and fail on device.
BROKEN_BUCKETS = {16: "scale mismatch observed on device"}


class LatencyModel:
    """
    Timing behaviour measured on device, split into the parts that actually vary.

    Fixed cost is dominated by the queue wait, not by the socket: a loopback ping
    measured p50 187us while service_queue_us sat at 2298 on every single request.
    """

    FIXED_US = 187.0 + 2298.0      # ipc round trip + service queue
    GRAPH_CREATE_US = 387_000.0    # cold: 387 ms measured
    GRAPH_CACHED_US = 0.0

    # Kernel cost calibrated from two measured points:
    #   128 x 512 x 512  -> 29294 us
    #   128 x 32 x 32    -> ~600 us
    PER_ELEMENT_US = 29294.0 / (128 * 512 + 512 * 512 + 128 * 512)
    MIN_KERNEL_US = 250.0

    def __init__(self, graph_cached=True):
        self.graph_cached = graph_cached
        self.graph_created = False

    def kernel_us(self, m, k, n):
        elements = m * k + k * n + m * n
        return max(self.MIN_KERNEL_US, elements * self.PER_ELEMENT_US)

    def submit_us(self, m, k, n):
        total = self.FIXED_US + self.kernel_us(m, k, n)
        if not self.graph_created:
            total += self.GRAPH_CREATE_US
            if self.graph_cached:
                self.graph_created = True
        return total


class NpuProfile:
    def __init__(self, name, buckets, int8, max_elements, latency):
        self.name = name
        self.buckets = buckets
        self.int8 = int8
        self.max_elements = max_elements
        self.latency = latency

    def bucketize(self, v):
        """Smallest allowed bucket >= v, or None if none is large enough."""
        for b in self.buckets:
            if b >= v:
                return b
        return None

    def check_shape(self, m, k, n):
        """
        Validate a submission exactly as the device would.

        Returns (ok, reason). Rejection here is the whole reason the simulator exists:
        an algorithm that only works with unconstrained shapes is not an NPU algorithm.
        """
        for name, v in (("m", m), ("k", k), ("n", n)):
            if v in BROKEN_BUCKETS:
                return False, f"{name}={v} rejected: {BROKEN_BUCKETS[v]}"
            if v not in self.buckets:
                return False, f"{name}={v} not in allowed buckets {self.buckets}"
        elements = m * k + k * n + m * n
        if elements > self.max_elements:
            return False, (
                f"element budget exceeded: {elements} > {self.max_elements} "
                f"(m={m} k={k} n={n})"
            )
        return True, "ok"


SNAPDRAGON_8S_GEN3 = NpuProfile(
    name="SNAPDRAGON_8S_GEN3",
    buckets=BUCKETS_8S_GEN3,
    int8=True,
    # The lattice path's own batching choice, not a device limit. Kept here so the
    # simulator enforces the same budget the worldgen pipeline plans around.
    max_elements=16384,
    latency=LatencyModel(),
)

PROFILES = {"SNAPDRAGON_8S_GEN3": SNAPDRAGON_8S_GEN3}
