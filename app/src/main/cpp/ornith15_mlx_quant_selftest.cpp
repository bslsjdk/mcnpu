#include "ornith15_mlx_quant.h"
#include <cmath>
#include <sstream>

namespace ornith15 {

bool MlxQuantSelfTest(std::string* report) {
    MlxQuantSpec s;
    s.bits = 4;
    s.group_size = 8;
    s.affine = true;

    // Packed low-to-high nibbles: q = 0..7, then 15..8.
    const uint32_t packed[2] = {
        0x76543210u,
        0x89abcdefu
    };
    const float scales[2] = {2.0f, 0.5f};
    const float biases[2] = {-1.0f, 3.0f};
    std::vector<float> out;
    std::string e;
    if (!MlxDequantizeRow(packed, scales, biases, 16, s, &out, &e)) {
        if (report) *report = "ERR MLX_QUANT_SELFTEST decode=" + e;
        return false;
    }
    const float expected[16] = {
        -1.f, 1.f, 3.f, 5.f, 7.f, 9.f, 11.f, 13.f,
        10.5f, 10.0f, 9.5f, 9.0f, 8.5f, 8.0f, 7.5f, 7.0f
    };
    for (size_t i = 0; i < 16; ++i) {
        if (!std::isfinite(out[i]) || std::fabs(out[i] - expected[i]) > 1e-6f) {
            if (report) {
                std::ostringstream os;
                os << "ERR MLX_QUANT_SELFTEST mismatch i=" << i
                   << " got=" << out[i] << " expected=" << expected[i];
                *report = os.str();
            }
            return false;
        }
    }

    std::vector<float> rejected;
    if (MlxDequantizeRow(packed, nullptr, biases, 16, s, &rejected, &e)) {
        if (report) *report = "ERR MLX_QUANT_SELFTEST accepted_null_scales";
        return false;
    }
    if (MlxDequantizeRow(packed, scales, nullptr, 16, s, &rejected, &e)) {
        if (report) *report = "ERR MLX_QUANT_SELFTEST accepted_null_biases";
        return false;
    }
    MlxQuantSpec bad = s;
    bad.group_size = 7;
    if (MlxDequantizeRow(packed, scales, biases, 16, bad, &rejected, &e)) {
        if (report) *report = "ERR MLX_QUANT_SELFTEST accepted_bad_group";
        return false;
    }

    if (report) *report = "OK MLX_QUANT_SELFTEST/1 bits=4 group=8 pack=LSB affine=scale*q+bias cases=4";
    return true;
}

} // namespace ornith15
