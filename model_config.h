#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <vector>
#include <cstdint>
#include <unordered_map>
#include <string>
#include <utility>
#include <stdexcept>

//Bitnet_config_1dot58_large
#define HIDDEN_SIZE 1536
#define INTERMEDIATE_SIZE 4096
#define NUM_HEADS 16
#define HEAD_NUM HIDDEN_SIZE / NUM_HEADS
#define MAX_SEQ_LEN 2048
#define NUM_LAYERS 24
#define PACK_FACTOR 4
#define VOCAB_SIZE 32002
#define RMS_NORM_EPS 1e-05


struct Shape2D {
    const std::unordered_map<std::string, std::pair<int, int>> shapes = {
        {"q_proj", {HIDDEN_SIZE / PACK_FACTOR, HIDDEN_SIZE}},
        {"k_proj", {HIDDEN_SIZE / PACK_FACTOR, HIDDEN_SIZE}},
        {"v_proj", {HIDDEN_SIZE / PACK_FACTOR, HIDDEN_SIZE}},
        {"o_proj", {HIDDEN_SIZE / PACK_FACTOR, HIDDEN_SIZE}},
        {"mlp.up_proj", {HIDDEN_SIZE / PACK_FACTOR, INTERMEDIATE_SIZE}},
        {"mlp.gate_proj", {HIDDEN_SIZE / PACK_FACTOR, INTERMEDIATE_SIZE}},
        {"mlp.down_proj", {INTERMEDIATE_SIZE / PACK_FACTOR, HIDDEN_SIZE}},
        {"embed_tokens", {VOCAB_SIZE, HIDDEN_SIZE}},
        {"lm_head", {HIDDEN_SIZE, VOCAB_SIZE}}
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