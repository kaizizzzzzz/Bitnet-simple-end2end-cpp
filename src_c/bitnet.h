#ifndef BITNET_H
#define BITNET_H

#include <vector>
#include <cstdint>


// Define the type for a 3D tensor for simplicity
using Tensor3D = std::vector<std::vector<std::vector<float>>>;
using Tensor2D = std::vector<std::vector<float>>;


// Function for Bitnet decoder equivalent in C++
std::vector<std::vector<float>> bitnet_decoder(
    std::vector<std::vector<float>> &hidden_states,
    const Tensor3D &q_weights_all_layers,
    const Tensor3D &k_weights_all_layers,
    const Tensor3D &v_weights_all_layers,
    const Tensor3D &o_weights_all_layers,
    const Tensor3D &gate_weights_all_layers,
    const Tensor3D &up_weights_all_layers,
    const Tensor3D &down_weights_all_layers,
    const std::vector<float> &q_scales_all_layers,  // Single scaling factor for Q weights
    const std::vector<float> &k_scales_all_layers,  // Single scaling factor for K weights
    const std::vector<float> &v_scales_all_layers,  // Single scaling factor for V weights
    const std::vector<float> &o_scales_all_layers,  // Single scaling factor for O weights
    const std::vector<float> &gate_scales_all_layers,  // Single scaling factor for gate weights
    const std::vector<float> &up_scales_all_layers,  // Single scaling factor for up weights
    const std::vector<float> &down_scales_all_layers,  // Single scaling factor for down weights
    const Tensor2D &inv_freq_all_layers,  // New: inv_freq for rotary embeddings
    const Tensor2D &ln_weight_in_attn_all_layers, // New: weights for RMSNorm, attn
    const Tensor2D &ln_weight_attn_all_layers, // New: weights for RMSNorm, attn
    const Tensor2D &ln_weight_in_mlp_all_layers, // New: weights for RMSNorm, mlp
    const Tensor2D &ln_weight_mlp_all_layers, // New: weights for RMSNorm, mlp
    size_t hidden_size, size_t intermediate_size, size_t num_heads, size_t head_dim, size_t seq_len, size_t num_layers,

    const std::vector<float> &ln_weight_in_final, // New: weights for RMSNorm, final
    const Tensor2D &lm_head_weights, // New: weights for LM head
    );
    
#endif // BITNET_H