#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "bitnet.h"
#include "attention.h"
#include "mlp.h"
#include "float_kernel.h"


// Function for Bitnet decoder layer equivalent in C++
std::vector<std::vector<float>> bitnet_decoder_layer(
    std::vector<std::vector<float>> &hidden_states,
    const std::vector<std::vector<uint8_t>> &q_weights,
    const std::vector<std::vector<uint8_t>> &k_weights,
    const std::vector<std::vector<uint8_t>> &v_weights,
    const std::vector<std::vector<uint8_t>> &o_weights,
    const std::vector<std::vector<uint8_t>> &gate_weights,
    const std::vector<std::vector<uint8_t>> &up_weights,
    const std::vector<std::vector<uint8_t>> &down_weights,
    const float q_scale,  // Single scaling factor for Q weights
    const float k_scale,  // Single scaling factor for K weights
    const float v_scale,  // Single scaling factor for V weights
    const float o_scale,  // Single scaling factor for O weights
    const float gate_scale,  // Single scaling factor for gate weights
    const float up_scale,  // Single scaling factor for up weights
    const float down_scale,  // Single scaling factor for down weights
    const std::vector<float> &inv_freq,  // New: inv_freq for rotary embeddings
    const std::vector<float> &ln_weight_in_attn, // New: weights for RMSNorm, attn
    const std::vector<float> &ln_weight_attn, // New: weights for RMSNorm, attn
    const std::vector<float> &ln_weight_in_mlp, // New: weights for RMSNorm, mlp
    const std::vector<float> &ln_weight_mlp, // New: weights for RMSNorm, mlp
    size_t hidden_size, size_t intermediate_size, size_t num_heads, size_t head_dim, size_t seq_len
    ){
    std::vector<std::vector<float>> residual = hidden_states;

    //attention layer
    hidden_states = bitnet_attention(hidden_states, q_weights, k_weights, v_weights, o_weights, q_scale, k_scale, v_scale, o_scale, inv_freq, ln_weight_in_attn, ln_weight_attn, hidden_size, num_heads, head_dim, seq_len); 

    hidden_states = residual + hidden_states;
    residual = hidden_states;

    //mlp layer
    hidden_states = bitnet_mlp(hidden_states, gate_weights, up_weights, down_weights, gate_scale, up_scale, down_scale, ln_weight_in_mlp, ln_weight_mlp, hidden_size, intermediate_size, seq_len);

    hidden_states = residual + hidden_states;    

    return hidden_states;   
}

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
    ){

    for (size_t l = 0; l < num_layers; ++l) {
        hidden_states = bitnet_decoder_layer(hidden_states, q_weights_all_layers[l], k_weights_all_layers[l], v_weights_all_layers[l], o_weights_all_layers[l], gate_weights_all_layers[l], up_weights_all_layers[l], down_weights_all_layers[l], q_scales_all_layers[l], k_scales_all_layers[l], v_scales_all_layers[l], o_scales_all_layers[l], gate_scales_all_layers[l], up_scales_all_layers[l], down_scales_all_layers[l], inv_freq_all_layers[l], ln_weight_in_attn_all_layers[l], ln_weight_attn_all_layers[l], ln_weight_in_mlp_all_layers[l], ln_weight_mlp_all_layers[l], hidden_size, intermediate_size, num_heads, head_dim, seq_len);
    }

    // Apply final_layernorm
    for (auto &row : hidden_states) {
        row = rms_norm(row, ln_weight_in_final);
    }

    // Through LM head for casual inference
    std::vector<std::vector<float>> logits = GEMM_2D_float(hidden_states, lm_head_weights);
    
    return logits;
}
