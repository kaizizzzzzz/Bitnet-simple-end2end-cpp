#ifndef ATTENTION_H
#define ATTENTION_H

#include <vector>
#include <cstdint>

// Function for Bitnet Attention equivalent in C++
std::vector<std::vector<float>> bitnet_attention(
    std::vector<std::vector<float>> &hidden_states,
    const QuantizedData &q_weights,
    const QuantizedData &k_weights,
    const QuantizedData &v_weights,
    const QuantizedData &o_weights,
    const std::vector<float> &inv_freq,  // New: inv_freq for rotary embeddings
    const std::vector<float> &ln_weight_in, // New: weights for RMSNorm
    const std::vector<float> &ln_weight, // New: weights for RMSNorm
    size_t hidden_size, size_t num_heads, size_t head_dim, size_t seq_len,
    std::vector<std::vector<std::vector<float>>> &k_cache,
    std::vector<std::vector<std::vector<float>>> &v_cache,
    size_t p_id,
    std::string mode = "prefill"
    );

#endif // ATTENTION_H
