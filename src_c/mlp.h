#ifndef MLP_H
#define MLP_H

#include <vector>
#include <cstdint>

// Function for Bitnet MLP equivalent in C++
std::vector<std::vector<float>> bitnet_mlp(
    std::vector<std::vector<float>> &hidden_states,
    const QuantizedData &gate_weights,
    const QuantizedData &up_weights,
    const QuantizedData &down_weights,
    const std::vector<float> &ln_weight_in, // New: weights for RMSNorm
    const std::vector<float> &ln_weight, // New: weights for RMSNorm
    size_t hidden_size, size_t intermediate_size
    );

#endif // MLP_H