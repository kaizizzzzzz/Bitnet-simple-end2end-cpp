#ifndef BITNET_H
#define BITNET_H

#include <vector>
#include <cstdint>
#include "embedding.h"
#include "../inference/load_model.h"

// Define the type for a 3D tensor for simplicity
using Tensor3D = std::vector<std::vector<std::vector<float>>>;
using Tensor2D = std::vector<std::vector<float>>;


// Function for Bitnet decoder equivalent in C++
std::vector<std::vector<float>> bitnet_decoder(
    Embedding &embedding_table,
    ModelData &bitnet_model_data,
    std::vector<size_t> &input_ids, size_t hidden_size, size_t intermediate_size, 
    size_t num_heads, size_t head_dim, size_t seq_len, size_t num_layers
    );

std::vector<std::vector<float>> bitnet_prefill_decoding(
    Embedding &embedding_table,
    ModelData &bitnet_model_data,
    std::vector<size_t> &input_ids, size_t hidden_size, size_t intermediate_size, 
    size_t num_heads, size_t head_dim, size_t seq_len, size_t num_layers,
    std::vector<std::vector<std::vector<std::vector<float>>>> &k_cache,
    std::vector<std::vector<std::vector<std::vector<float>>>> &v_cache,
    size_t p_id,
    std::string mode
);

#endif // BITNET_H
