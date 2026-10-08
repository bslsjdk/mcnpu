#include "ornith15_mlx_quant.h"
#include <limits>

namespace ornith15 {
bool MlxDequantizeRow(const uint32_t* packed, const float* scales,
                      const float* biases, size_t n,
                      const MlxQuantSpec& s, std::vector<float>* out,
                      std::string* e) {
    if (!packed || !scales || !out || s.bits <= 0 || s.bits > 8 ||
        (32 % s.bits) != 0 || s.group_size <= 0 ||
        (s.group_size % s.bits) != 0 || n % (32 / s.bits) != 0 ||
        n % static_cast<size_t>(s.group_size) != 0) {
        if (e) *e = "invalid MLX quant row";
        return false;
    }
    const size_t per_word = 32u / static_cast<size_t>(s.bits);
    const uint32_t mask = (1u << s.bits) - 1u;
    out->resize(n);
    const size_t groups = n / static_cast<size_t>(s.group_size);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t q = (packed[i / per_word] >> ((i % per_word) * s.bits)) & mask;
        const size_t g = i / static_cast<size_t>(s.group_size);
        const float b = s.affine && biases ? biases[g] : 0.0f;
        (*out)[i] = static_cast<float>(q) * scales[g] + b;
    }
    return true;
}
} // namespace ornith15
