#include <jni.h>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cmath>
#include <string>

namespace {

constexpr size_t PQ2_BLOCK_BYTES = 34;
constexpr size_t PQ2_VALUES = 128;

static uint16_t load_u16_le(const uint8_t * p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

static float half_to_float(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x3ffu;
            bits = sign | (static_cast<uint32_t>(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Exact Prism PQ2_0 block codec.
// Layout: fp16 scale + 32 bytes carrying four 2-bit values per byte.
// Codes: 00=-1, 01=0, 10=+1, 11=+2; value=(code-1)*scale.
static void decode_block(const uint8_t * block, float * out) {
    const float d = half_to_float(load_u16_le(block));
    const uint8_t * qs = block + 2;
    for (size_t j = 0; j < PQ2_VALUES; ++j) {
        const uint8_t q = static_cast<uint8_t>((qs[j >> 2] >> ((j & 3u) * 2u)) & 0x03u);
        out[j] = static_cast<float>(static_cast<int>(q) - 1) * d;
    }
}

static void fwht_normalized(float * x, size_t n) {
    for (size_t h = 1; h < n; h <<= 1) {
        for (size_t i = 0; i < n; i += h << 1) {
            for (size_t j = 0; j < h; ++j) {
                const float a = x[i + j];
                const float b = x[i + j + h];
                x[i + j] = a + b;
                x[i + j + h] = a - b;
            }
        }
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(n));
    for (size_t i = 0; i < n; ++i) x[i] *= scale;
}

static std::string run_hadamard_probe() {
    constexpr size_t N = 1024;
    float x[N];
    float original[N];
    for (size_t i = 0; i < N; ++i) {
        const float sign = (i & 1u) ? -1.0f : 1.0f;
        x[i] = sign * (0.125f + static_cast<float>(i % 17) * 0.03125f);
        original[i] = x[i];
    }

    for (size_t i = 0; i < N; ++i) {
        const float s = (i % 7u == 0u || i % 11u == 0u) ? -1.0f : 1.0f;
        x[i] *= s;
    }

    fwht_normalized(x, N);
    fwht_normalized(x, N);

    for (size_t i = 0; i < N; ++i) {
        const float s = (i % 7u == 0u || i % 11u == 0u) ? -1.0f : 1.0f;
        x[i] *= s;
    }

    float max_abs = 0.0f;
    double energy_before = 0.0;
    double energy_after = 0.0;
    int bad = 0;
    for (size_t i = 0; i < N; ++i) {
        const float err = std::fabs(x[i] - original[i]);
        max_abs = std::fmax(max_abs, err);
        if (err > 1e-5f) ++bad;
        energy_before += static_cast<double>(original[i]) * original[i];
        energy_after += static_cast<double>(x[i]) * x[i];
    }

    return "OK BONSAI2_HADAMARD_PROBE/1 block=1024 transform=normalized-sylvester-walsh-hadamard "
           "sign_mode=explicit self_inverse=1 bad=" + std::to_string(bad) +
           " max_abs=" + std::to_string(max_abs) +
           " energy_err=" + std::to_string(std::fabs(energy_after - energy_before));
}

static std::string run_probe() {
    alignas(16) uint8_t block[PQ2_BLOCK_BYTES] = {};
    block[0] = 0x00;
    block[1] = 0x3c;

    for (size_t i = 0; i < 32; ++i) {
        block[2 + i] = 0xE4;
    }

    float out[PQ2_VALUES];
    const auto t0 = std::chrono::steady_clock::now();
    decode_block(block, out);
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count();

    int bad = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < PQ2_VALUES; ++i) {
        const int slot = static_cast<int>(i & 3u);
        const float expected = static_cast<float>(slot - 1);
        const float err = std::fabs(out[i] - expected);
        max_abs = std::fmax(max_abs, err);
        if (err > 1e-6f) ++bad;
    }

    double sum = 0.0;
    for (float v : out) sum += v;

    return "OK BONSAI2_PQ2_DECODE/1 type=142 group=128 block_bytes=34 "
           "scale=fp16 codec=Q2 codes=00:-1,01:0,10:+1,11:+2 "
           "bad=" + std::to_string(bad) +
           " max_abs=" + std::to_string(max_abs) +
           " sum=" + std::to_string(sum) +
           " decode_us=" + std::to_string(static_cast<long long>(us));
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_bslsjdk_mcnpu_NpuRuntime_nativeBonsai2Pq2DecodeProbe(JNIEnv * env, jclass) {
    const std::string r = run_probe();
    return env->NewStringUTF(r.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_bslsjdk_mcnpu_Bonsai2Pq2Probe_nativeRun(JNIEnv * env, jclass) {
    const std::string r = run_probe();
    return env->NewStringUTF(r.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_bslsjdk_mcnpu_Bonsai2Pq2Probe_nativeHadamardRun(JNIEnv * env, jclass) {
    const std::string r = run_hadamard_probe();
    return env->NewStringUTF(r.c_str());
}
