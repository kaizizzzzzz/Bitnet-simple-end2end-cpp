#ifndef MLP_H
#define MLP_H

#include <vector>
#include <cstdint>

// Function for Bitnet MLP equivalent in C++
std::vector<std::vector<float>> bitnet_mlp(
    const std::vector<std::vector<float>> &hidden_states,
    const std::vector<std::vector<uint8_t>> &gate_weights,
    const std::vector<std::vector<uint8_t>> &up_weights,
    const std::vector<std::vector<uint8_t>> &down_weights,
    const float gate_scale,  // Single scaling factor for mlp1 weights
    const float up_scale,  // Single scaling factor for mlp2 weights
    const float down_scale,  // Single scaling factor for mlp3 weights
    const std::vector<float> &ln_weight_in, // New: weights for RMSNorm
    const std::vector<float> &ln_weight, // New: weights for RMSNorm
    size_t hidden_size, size_t intermediate_size, size_t seq_len
    );

#endif // MLP_H