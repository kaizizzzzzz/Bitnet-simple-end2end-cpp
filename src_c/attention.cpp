#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "linear_kernel.h"
#include "attention.h"

// Define the type for a 3D tensor for simplicity
using Tensor3D = std::vector<std::vector<std::vector<float>>>;

// Function to reshape a 2D matrix into a 3D tensor
// Converts [seq_len, hidden_size] -> [num_heads, seq_len, head_dim]
Tensor3D reshape_2D_to_3D(const std::vector<std::vector<float>> &input, size_t num_heads, size_t seq_len, size_t head_dim) {
    Tensor3D output(num_heads, std::vector<std::vector<float>>(seq_len, std::vector<float>(head_dim)));

    // Ensure the input matrix has the expected dimensions
    if (input.size() != seq_len || input[0].size() != num_heads * head_dim) {
        throw std::runtime_error("Input dimensions do not match the expected [seq_len, num_heads * head_dim]");
    }

    // Correctly iterate through the input matrix to fill the 3D tensor
    for (size_t s = 0; s < seq_len; ++s) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t d = 0; d < head_dim; ++d) {
                output[h][s][d] = input[s][h * head_dim + d];
            }
        }
    }

    return output;
}

// Transpose last two dimensions for the K matrix
// Converts [num_heads, seq_len, head_dim] -> [num_heads, head_dim, seq_len]
Tensor3D transpose_last_two_dims(const Tensor3D &input) {
    size_t num_heads = input.size();
    size_t seq_len = input[0].size();
    size_t head_dim = input[0][0].size();

    Tensor3D output(num_heads, std::vector<std::vector<float>>(head_dim, std::vector<float>(seq_len)));

    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t d = 0; d < head_dim; ++d) {
                output[h][d][s] = input[h][s][d];
            }
        }
    }

    return output;
}

// Matrix multiplication for 3D tensors
// This function multiplies tensors of shape [num_heads, seq_len, head_dim] with [num_heads, head_dim, seq_len]
Tensor3D matmul_3D(const Tensor3D &tensor1, const Tensor3D &tensor2) {
    // Validate dimensions
    size_t num_heads1 = tensor1.size();
    size_t seq_len1 = tensor1[0].size();
    size_t head_dim1 = tensor1[0][0].size();

    size_t num_heads2 = tensor2.size();
    size_t head_dim2 = tensor2[0].size();  // This should match the last dimension of tensor1
    size_t seq_len2 = tensor2[0][0].size();

    if (num_heads1 != num_heads2) {
        throw std::runtime_error("Number of heads in tensor1 and tensor2 must match.");
    }
    if (head_dim1 != head_dim2) {
        throw std::runtime_error("Head dimension of tensor1 and head dimension of tensor2 must match.");
    }

    size_t num_heads = num_heads1;
    size_t seq_len = seq_len1;
    size_t head_dim = head_dim1;

    // Initialize result tensor with the correct dimensions
    Tensor3D result(num_heads, std::vector<std::vector<float>>(seq_len, std::vector<float>(seq_len2, 0.0f)));

    // Perform matrix multiplication
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < seq_len2; ++j) {
                for (size_t k = 0; k < head_dim; ++k) {
                    result[h][i][j] += tensor1[h][i][k] * tensor2[h][k][j];
                }
            }
        }
    }

    return result;
}

// Softmax function (assumes input is of shape [num_heads, seq_len, seq_len])
void softmax(Tensor3D &tensor) {
    for (auto &head : tensor) {
        for (auto &row : head) {
            float max_val = *std::max_element(row.begin(), row.end());
            float sum_exp = 0.0f;

            for (auto &elem : row) {
                elem = std::exp(elem - max_val);  // Prevent overflow
                sum_exp += elem;
            }
            for (auto &elem : row) {
                elem /= sum_exp;
            }
        }
    }
}

// Function to rotate half of the tensor values (used in rotary embeddings)
Tensor3D rotate_half(const Tensor3D &x) {
    size_t num_heads = x.size();
    size_t seq_len = x[0].size();
    size_t head_dim = x[0][0].size();

    Tensor3D rotated_x(num_heads, std::vector<std::vector<float>>(seq_len, std::vector<float>(head_dim)));

    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t d = 0; d < head_dim / 2; ++d) {
                rotated_x[h][s][d] = x[h][s][d + head_dim / 2];        // Assign x2 to x1 position
                rotated_x[h][s][d + head_dim / 2] = -x[h][s][d];       // Assign -x1 to x2 position
            }
        }
    }

    return rotated_x;
}

// Rotary embedding function (cosine and sine calculations)
void rotary_embedding(const Tensor3D &x, const std::vector<float> &inv_freq, size_t seq_len, Tensor3D &cos, Tensor3D &sin) {
    size_t num_heads = x.size();
    size_t head_dim = x[0][0].size();
    
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t d = 0; d < head_dim; ++d) {
                float angle = inv_freq[d / 2] * s;
                cos[h][s][d] = std::cos(angle);
                sin[h][s][d] = std::sin(angle);
            }
        }
    }
}

// Apply rotary position embedding to Q and K tensors
std::pair<Tensor3D, Tensor3D> apply_rotary_pos_emb(const Tensor3D &q, const Tensor3D &k, const Tensor3D &cos, const Tensor3D &sin) {
    Tensor3D q_embed = q;
    Tensor3D k_embed = k;

    size_t num_heads = q.size();
    size_t seq_len = q[0].size();
    size_t head_dim = q[0][0].size();

    Tensor3D rotated_q = rotate_half(q);
    Tensor3D rotated_k = rotate_half(k);

    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t d = 0; d < head_dim; ++d) {
                q_embed[h][s][d] = q[h][s][d] * cos[h][s][d] + rotated_q[h][s][d] * sin[h][s][d];
                k_embed[h][s][d] = k[h][s][d] * cos[h][s][d] + rotated_k[h][s][d] * sin[h][s][d];
            }
        }
    }

    return {q_embed, k_embed};
}


// RMS Normalization function
std::vector<float> rms_norm(const std::vector<float> &hidden_states, const std::vector<float> &weight, float epsilon = 1e-6) {
    size_t hidden_size = hidden_states.size();
    float variance = 0.0f;

    for (float val : hidden_states) {
        variance += val * val;
    }

    variance = std::sqrt(variance / hidden_size + epsilon);
    std::vector<float> normalized_states(hidden_size);

    for (size_t i = 0; i < hidden_size; ++i) {
        normalized_states[i] = (hidden_states[i] / variance) * weight[i];
    }

    return normalized_states;
}

// Helper function to create a causal mask (lower triangular matrix with negative infinity above the diagonal)
std::vector<std::vector<float>> create_causal_mask(size_t seq_len) {
    std::vector<std::vector<float>> mask(seq_len, std::vector<float>(seq_len, -std::numeric_limits<float>::infinity()));
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            mask[i][j] = 0.0f;  // Keep positions on and below the diagonal as zero
        }
    }
    return mask;
}

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
    const std::vector<float> &inv_freq,  // New: inv_freq for rotary embeddings
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
