#ifndef ATTENTION_H
#define ATTENTION_H

#include <vector>
#include <cstdint>

// Function for Bitnet Attention equivalent in C++
std::vector<std::vector<float>> bitnet_attention(
    const std::vector<std::vector<float>> &hidden_states,
    const std::vector<std::vector<uint8_t>> &q_weights,
    const std::vector<std::vector<uint8_t>> &k_weights,
    const std::vector<std::vector<uint8_t>> &v_weights,
    const std::vector<std::vector<uint8_t>> &o_weights,
    const float q_scale,  // Single scaling factor for Q weights
    const float k_scale,  // Single scaling factor for K weights
    const float v_scale,  // Single scaling factor for V weights
    const float o_scale,  // Single scaling factor for O weights
    const std::vector<float> &inv_freq,  // inv_freq for rotary embeddings
    const std::vector<float> &ln_weight, // weights for RMSNorm
    size_t hidden_size, size_t num_heads, size_t head_dim, size_t seq_len
);

#endif // ATTENTION_H
