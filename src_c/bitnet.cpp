#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "bitnet.h"
#include "attention.h"
#include "mlp.h"
#include "float_kernel.h"
#include "embedding.h"
#include "rmsnorm.h"
#include <stdexcept>

std::vector<std::vector<float>> add_2d_vectors(
    const std::vector<std::vector<float>>& a, 
    const std::vector<std::vector<float>>& b) 
{
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        throw std::invalid_argument("Vectors must be the same dimensions for element-wise addition");
    }

    std::vector<std::vector<float>> result = a;
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[i].size(); ++j) {
            result[i][j] += b[i][j];
        }
    }
    return result;
}

// Function for Bitnet decoder layer equivalent in C++
std::vector<std::vector<float>> bitnet_decoder_layer(
    std::vector<std::vector<float>> &hidden_states,
    const QuantizedData &q_weights,
    const QuantizedData &k_weights,
    const QuantizedData &v_weights,
    const QuantizedData &o_weights,
    const QuantizedData &gate_weights,
    const QuantizedData &up_weights,
    const QuantizedData &down_weights,
    const std::vector<float> &inv_freq,  // New: inv_freq for rotary embeddings
    const std::vector<float> &ln_weight_in_attn, // New: weights for RMSNorm, attn
    const std::vector<float> &ln_weight_attn, // New: weights for RMSNorm, attn
    const std::vector<float> &ln_weight_in_mlp, // New: weights for RMSNorm, mlp
    const std::vector<float> &ln_weight_mlp, // New: weights for RMSNorm, mlp
    size_t hidden_size, size_t intermediate_size, size_t num_heads, size_t head_dim, size_t seq_len,
    std::vector<std::vector<std::vector<float>>> &k_cache,
    std::vector<std::vector<std::vector<float>>> &v_cache,
    size_t p_id,
    std::string mode = "prefill"
    ){
    std::vector<std::vector<float>> residual = hidden_states;

    //attention layer
    hidden_states = bitnet_attention(hidden_states, q_weights, k_weights, v_weights, o_weights, inv_freq, ln_weight_in_attn, ln_weight_attn, hidden_size, num_heads, head_dim, seq_len, k_cache, v_cache, p_id, mode); 

    hidden_states = add_2d_vectors(residual, hidden_states);
    residual = hidden_states;

    //mlp layer
    hidden_states = bitnet_mlp(hidden_states, gate_weights, up_weights, down_weights, ln_weight_in_mlp, ln_weight_mlp, hidden_size, intermediate_size);

    hidden_states = add_2d_vectors(residual, hidden_states);  

    return hidden_states;   
}

// Function for Bitnet decoder equivalent in C++
std::vector<std::vector<float>> bitnet_decoder(
    Embedding &embedding_table,
    ModelData &bitnet_model_data,
    std::vector<size_t> &input_ids, size_t hidden_size, size_t intermediate_size, 
    size_t num_heads, size_t head_dim, size_t seq_len, size_t num_layers
    ){
    std::vector<std::vector<std::vector<float>>> k_cache;
    std::vector<std::vector<std::vector<float>>> v_cache;
    if (input_ids.size() != seq_len) {
        throw std::invalid_argument("Input sequence length does not match the expected sequence length");
    }
    std::vector<std::vector<float>> hidden_states = embedding_table.forward(input_ids);

    for (size_t l = 0; l < num_layers; ++l) {
        std::unordered_map<std::string, QuantizedData> quantized_params = bitnet_model_data.layers[l].quantized_params;
        std::unordered_map<std::string, std::vector<float>> float_params_1D = bitnet_model_data.layers[l].float_params_1D;
        std::unordered_map<std::string, std::vector<std::vector<float>>> float_params_2D = bitnet_model_data.layers[l].float_params_2D;
        hidden_states = bitnet_decoder_layer(hidden_states, quantized_params["q_proj"], quantized_params["k_proj"], quantized_params["v_proj"], quantized_params["o_proj"], quantized_params["mlp.gate_proj"], quantized_params["mlp.up_proj"], quantized_params["mlp.down_proj"], float_params_1D["rotary_emb.inv_freq"], float_params_1D["input_layernorm"], float_params_1D["inner_attn_ln"], float_params_1D["post_attention_layernorm"], float_params_1D["mlp.ffn_layernorm"], hidden_size, intermediate_size, num_heads, head_dim, seq_len, k_cache, v_cache, 0, "prefill");
    }

    // Apply final_layernorm
    for (auto &row : hidden_states) {
        row = rms_norm(row, bitnet_model_data.non_layer_params_1D["norm"], RMS_NORM_EPS);
    }

    // Through LM head for casual inference
    std::vector<std::vector<float>> logits = GEMM_2D_float(hidden_states, bitnet_model_data.non_layer_params_2D["lm_head"]);

    return logits;
}


std::vector<std::vector<float>> bitnet_prefill_decoding(
    Embedding &embedding_table,
    ModelData &bitnet_model_data,
    std::vector<size_t> &input_ids, size_t hidden_size, size_t intermediate_size, 
    size_t num_heads, size_t head_dim, size_t seq_len, size_t num_layers,
    std::vector<std::vector<std::vector<std::vector<float>>>> &k_cache,
    std::vector<std::vector<std::vector<std::vector<float>>>> &v_cache,
    size_t p_id,
    std::string mode
    ){
    if (input_ids.size() != 1 && mode == "decoding") {
        throw std::invalid_argument("Input sequence length must be 1 for decoding");
    }
    std::vector<std::vector<float>> hidden_states = embedding_table.forward(input_ids);

    for (size_t l = 0; l < num_layers; ++l) {
        std::unordered_map<std::string, QuantizedData> quantized_params = bitnet_model_data.layers[l].quantized_params;
        std::unordered_map<std::string, std::vector<float>> float_params_1D = bitnet_model_data.layers[l].float_params_1D;
        std::unordered_map<std::string, std::vector<std::vector<float>>> float_params_2D = bitnet_model_data.layers[l].float_params_2D;
        hidden_states = bitnet_decoder_layer(hidden_states, quantized_params["q_proj"], quantized_params["k_proj"], quantized_params["v_proj"], quantized_params["o_proj"], quantized_params["mlp.gate_proj"], quantized_params["mlp.up_proj"], quantized_params["mlp.down_proj"], float_params_1D["rotary_emb.inv_freq"], float_params_1D["input_layernorm"], float_params_1D["inner_attn_ln"], float_params_1D["post_attention_layernorm"], float_params_1D["mlp.ffn_layernorm"], hidden_size, intermediate_size, num_heads, head_dim, seq_len, k_cache[l], v_cache[l], p_id, mode);
    }

    // Apply final_layernorm
    for (auto &row : hidden_states) {
        row = rms_norm(row, bitnet_model_data.non_layer_params_1D["norm"], RMS_NORM_EPS);
    }

    // Through LM head for casual inference
    std::vector<std::vector<float>> logits = GEMM_2D_float(hidden_states, bitnet_model_data.non_layer_params_2D["lm_head"]);

    return logits;
}