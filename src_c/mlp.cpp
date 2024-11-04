#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "linear_kernel.h"
#include "mlp.h"
#include "rmsnorm.h"


// Function for Bitnet Attention equivalent in C++
std::vector<std::vector<float>> bitnet_mlp(
    const std::vector<std::vector<float>> &hidden_states,
    const std::vector<std::vector<uint8_t>> &mlp1_weights,
    const std::vector<std::vector<uint8_t>> &mlp2_weights,
    const std::vector<std::vector<uint8_t>> &mlp3_weights,
    const float mlp1_scale,  // Single scaling factor for mlp1 weights
    const float mlp2_scale,  // Single scaling factor for mlp2 weights
    const float mlp3_scale,  // Single scaling factor for mlp3 weights
    const std::vector<float> &ln_weight, // New: weights for RMSNorm
    size_t hidden_size, size_t num_heads, size_t head_dim, size_t seq_len
    ) {

    // Step 1: Quantize the input activations for Q, K, V projections
    auto quantized_result = quantize_activation(hidden_states, 8);
    std::vector<std::vector<int8_t>> quantized_hidden_states = quantized_result.first;
    std::vector<float> scales = quantized_result.second;

    // Step 2: Linear projections for Q, K, V using quantized GEMM (forward_no_mul)
    std::vector<std::vector<float>> q_proj_re = linear_forward_no_mul(quantized_hidden_states, scales, q_weights, q_scale, hidden_size);
    std::vector<std::vector<float>> k_proj_re = linear_forward_no_mul(quantized_hidden_states, scales, k_weights, k_scale, hidden_size);
    std::vector<std::vector<float>> v_proj_re = linear_forward_no_mul(quantized_hidden_states, scales, v_weights, v_scale, hidden_size);

    // Reshape Q, K, V for attention calculation
    Tensor3D q_proj = reshape_2D_to_3D(q_proj_re, num_heads, seq_len, head_dim);
    Tensor3D k_proj = reshape_2D_to_3D(k_proj_re, num_heads, seq_len, head_dim);
    Tensor3D v_proj = reshape_2D_to_3D(v_proj_re, num_heads, seq_len, head_dim);
            
    // Step 3: Apply rotary embedding
    Tensor3D cos(num_heads, std::vector<std::vector<float>>(seq_len, std::vector<float>(head_dim)));
    Tensor3D sin(num_heads, std::vector<std::vector<float>>(seq_len, std::vector<float>(head_dim)));
    rotary_embedding(q_proj, inv_freq, seq_len, cos, sin);
    auto q_k_embed_pair = apply_rotary_pos_emb(q_proj, k_proj, cos, sin);
    Tensor3D q_embed = q_k_embed_pair.first;
    Tensor3D k_embed = q_k_embed_pair.second;

    // Step 4: Transpose K for correct multiplication
    Tensor3D k_proj_transposed = transpose_last_two_dims(k_embed);

    // Step 5: Calculate attention scores (QK^T) / sqrt(d)
    auto attn_weights = matmul_3D(q_embed, k_proj_transposed);
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (auto &head : attn_weights) {
        for (auto &row : head) {
            for (auto &elem : row) {
                elem *= scale_factor;
            }
        }
    }

    // Create a causal mask and apply it to the attention weights
    auto causal_mask = create_causal_mask(seq_len);
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < seq_len; ++j) {
                attn_weights[h][i][j] += causal_mask[i][j];
            }
        }
    }

    // Step 6: Apply softmax
    softmax(attn_weights);

    // Step 7: Multiply with V
    auto attn_output = matmul_3D(attn_weights, v_proj);

    // Step 8: Reshape the attention output from 3D to 2D
    std::vector<std::vector<float>> attn_output_2D(seq_len, std::vector<float>(hidden_size, 0.0f));
    for (size_t s = 0; s < seq_len; ++s) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t d = 0; d < head_dim; ++d) {
                attn_output_2D[s][h * head_dim + d] = attn_output[h][s][d];
            }
        }
    }

    // Step 9: Apply RMS normalization before projection
    for (auto &row : attn_output_2D) {
        row = rms_norm(row, ln_weight);
    }

    // Step 10: Final output projection using quantized GEMM (forward_no_mul)
    // Quantize the attention output before final projection
    auto quantized_final_result = quantize_activation(attn_output_2D, 8);
    std::vector<std::vector<int8_t>> quantized_final_output = quantized_final_result.first;
    std::vector<float> final_scales = quantized_final_result.second;


    std::vector<std::vector<float>> final_output = linear_forward_no_mul(quantized_final_output, final_scales, o_weights, o_scale, hidden_size);

    return final_output;
}
