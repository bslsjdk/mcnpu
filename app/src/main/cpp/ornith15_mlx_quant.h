#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "ornith15_safetensors.h"

namespace ornith15 {

// MLX affine quantization descriptor. The on-disk weight is uint32 packed
// values; one scale/bias pair applies to group_size logical input elements.
struct MlxQuantSpec {
    int bits = 4;
    int group_size = 64;
    bool affine = true;
};

// Dequantize one packed row without allocating the full model weight.
// This is deliberately the CPU golden path used to validate tensor layout
// before any HTP implementation is enabled.
bool MlxDequantizeRow(const uint32_t* packed, const float* scales,
                      const float* biases, size_t logical_elements,
                      const MlxQuantSpec& spec, std::vector<float>* out,
                      std::string* error);

// Deterministic CPU-only regression for the MLX affine decoder. No model data is read.
bool MlxQuantSelfTest(std::string* report);

} // namespace ornith15
