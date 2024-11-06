#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <vector>
#include <cstdint>
#include <unordered_map>
#include <string>
#include <utility>
#include <stdexcept>

//Bitnet_config_1dot58_large
#define hidden_size 1536
#define intermediate_size 4096
#define num_heads 16
#define head_dim hidden_size / num_heads
#define max_seq_len 2048
#define num_layers 16
#define pack_factor 4
#define vocab_size 32002


struct Shape2D {
    const std::unordered_map<std::string, std::pair<int, int>> shapes = {
        {"q_proj", {hidden_size / pack_factor, hidden_size}},
        {"k_proj", {hidden_size / pack_factor, hidden_size}},
        {"v_proj", {hidden_size / pack_factor, hidden_size}},
        {"o_proj", {hidden_size / pack_factor, hidden_size}},
        {"mlp.up_proj", {hidden_size / pack_factor, intermediate_size}},
        {"mlp.gate_proj", {hidden_size / pack_factor, intermediate_size}},
        {"mlp.down_proj", {intermediate_size / pack_factor, hidden_size}},
        {"embed_tokens", {vocab_size, hidden_size}},
        {"lm_head", {hidden_size, vocab_size}}
    };

    std::pair<size_t, size_t> get_shape(const std::string& name) const {
        auto it = shapes.find(name);
        if (it != shapes.end()) {
            return it->second;
        } else {
            throw std::invalid_argument("Shape not found for name: " + name);
        }
    }
};


#endif // MODEL_CONFIG_H