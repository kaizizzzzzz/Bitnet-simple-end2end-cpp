#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "bitnet.h"
#include "linear_kernel_no_mul.h"
#include "float_kernel.h"
#include "mlp.h"
#include "rmsnorm.h"

#include <iostream>
#include <cmath>
#include <vector>

// Define the sigmoid function
float sigmoid(float x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Define the SiLU (Swish) function
float silu(float x) {
    return x * sigmoid(x);
}

// Apply SiLU to a vector of values
std::vector<float> apply_silu(const std::vector<std::vector<float>>& input) {
    std::vector<std::vector<float>> output;
    for (const auto& row : input) {
        std::vector<float> output_row;
        for (const auto& elem : row) {
            output_row.push_back(silu(elem));
        }
        output.push_back(output_row);
    }
    return output;
}

// Function for Bitnet MLP equivalent in C++
std::vector<std::vector<float>> bitnet_mlp(
    const std::vector<std::vector<float>> &hidden_states,
    const std::vector<std::vector<uint8_t>> &gate_weights,
    const std::vector<std::vector<uint8_t>> &up_weights,
    const std::vector<std::vector<uint8_t>> &down_weights,
    const float gate_scale,  // Single scaling factor for gate weights
    const float up_scale,  // Single scaling factor for up weights
    const float down_scale,  // Single scaling factor for down weights
    const std::vector<float> &ln_weight_in, // New: weights for RMSNorm
    const std::vector<float> &ln_weight, // New: weights for RMSNorm
    size_t hidden_size, size_t intermediate_size, size_t seq_len
    ) {
    
    // Step 1: Apply post_attention_layernorm
    for (auto &row : hidden_states) {
        row = rms_norm(row, ln_weight_in);
    }

    // Step 2: Quantize the input activations for Q, K, V projections
    auto quantized_result = quantize_activation(hidden_states, 8);
    std::vector<std::vector<int8_t>> quantized_hidden_states = quantized_result.first;
    std::vector<float> scales = quantized_result.second;

    // Step 3: Linear projections for gate, up, using quantized GEMM (forward_no_mul)
    std::vector<std::vector<float>> gate_proj_re = linear_forward_no_mul(quantized_hidden_states, scales, gate_weights, gate_scale, intermediate_size);
    std::vector<std::vector<float>> up_proj_re = linear_forward_no_mul(quantized_hidden_states, scales, up_weights, up_scale, intermediate_size);

    //Step 4: Hadmard product between gate and up
    std::vector<std::vector<float>> gate_up_mul = element_mul_2D_float(gate_proj_re, up_proj_re);

    //Step 5: Apply SiLU activation
    std::vector<std::vector<float>> gate_up_mul_silu = apply_silu(gate_up_mul);

    // Step 6: Apply RMS normalization before down projection
    for (auto &row : gate_up_mul_silu) {
        row = rms_norm(row, ln_weight);
    }

    //Step 7: projection for down using quantized GEMM (forward_no_mul)
    // Quantize the activation for down projection Linear
    auto quantized_result = quantize_activation(gate_up_mul_silu, 8);
    std::vector<std::vector<int8_t>> quantized_hidden_states = quantized_result.first;
    std::vector<float> scales = quantized_result.second;

    std::vector<std::vector<float>> down_proj_re = linear_forward_no_mul(quantized_hidden_states, scales, down_weights, down_scale, hidden_size);

    return down_proj_re;
}
