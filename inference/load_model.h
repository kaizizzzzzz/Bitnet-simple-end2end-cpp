#ifndef LOAD_MODEL_H
#define LOAD_MODEL_H

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <cstdint>
#include <stdexcept>

struct QuantizedData {
    float scale;
    std::vector<uint8_t> packed_data;
};

struct LayerData {
    std::unordered_map<std::string, std::vector<float>> float_params;
    std::unordered_map<std::string, QuantizedData> quantized_params;
};

struct ModelData {
    std::unordered_map<int, LayerData> layers;
    std::unordered_map<std::string, std::vector<float>> non_layer_params;
};


std::pair<std::optional<int>, std::string> parse_name(const std::string& name) {
    // Remove "model" and "weight" or "self_attn" and extract layer index and weight name
    std::string cleaned_name = name;
    size_t pos = cleaned_name.find("model.");
    if (pos != std::string::npos) cleaned_name.erase(pos, 6);

    pos = cleaned_name.find(".weight");
    if (pos != std::string::npos) cleaned_name.erase(pos, 7);

    pos = cleaned_name.find(".self_attn");
    if (pos != std::string::npos) cleaned_name.erase(pos, 10);

    // Extract layer index
    // Check if there is a layer index by looking for ".layers."
    size_t layer_start = cleaned_name.find("layers.");
    if (layer_start == std::string::npos) {
        // No layer index found, return without layer index
        return {std::nullopt, cleaned_name};
    }

    // Layer index exists, parse it
    layer_start += 7;  // Move past "layers."
    // std::cout << "layer_start: " << layer_start << std::endl;
    if (layer_start == std::string::npos || layer_start >= cleaned_name.size()) {
        throw std::invalid_argument("Layer index not found in name: " + name);
    }

    size_t layer_end = cleaned_name.find('.', layer_start);
    if (layer_end == std::string::npos) {
        throw std::invalid_argument("Invalid format for layer index in name: " + name);
    }

    std::string layer_index_str = cleaned_name.substr(layer_start, layer_end - layer_start);
    int layer_index;

    try {
        layer_index = std::stoi(layer_index_str);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument("Failed to convert layer index to integer: " + layer_index_str);
    }

    // Extract the weight name after the layer index
    std::string weight_name = cleaned_name.substr(layer_end + 1);

    return {layer_index, weight_name};
}


ModelData load_model_from_bin(const std::string& input_bin_path) {
    ModelData model_data;
    std::ifstream file(input_bin_path, std::ios::binary);

    if (!file) {
        std::cerr << "Error opening file: " << input_bin_path << std::endl;
        return model_data;
    }

    while (file) {
        // Read name length
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));
        if (!file) break;

        // Read the parameter name
        std::string name(name_len, '\0');
        file.read(&name[0], name_len);

        // Parse the name to get the layer index and weight name
        auto [layer_index, weight_name] = parse_name(name);

        // Read data size
        uint32_t data_size;

        // Check if the parameter is quantized
        bool is_quantized = (weight_name.find("down_proj") != std::string::npos ||
                             weight_name.find("gate_proj") != std::string::npos ||
                             weight_name.find("up_proj") != std::string::npos ||
                             weight_name.find("q_proj") != std::string::npos ||
                             weight_name.find("k_proj") != std::string::npos ||
                             weight_name.find("v_proj") != std::string::npos ||
                             weight_name.find("o_proj") != std::string::npos);

        if (is_quantized) {
            // Read scale
            float scale;
            file.read(reinterpret_cast<char*>(&scale), sizeof(float));
            file.read(reinterpret_cast<char*>(&data_size), sizeof(uint32_t));
            // Read packed data
            std::vector<uint8_t> packed_data(data_size);
            file.read(reinterpret_cast<char*>(packed_data.data()), data_size);

            if (layer_index.has_value()) {
                model_data.layers[layer_index.value()].quantized_params[weight_name] = {scale, std::move(packed_data)};
            } else {
                throw std::runtime_error("Quantized data must have a layer index. Missing layer index for parameter: " + weight_name);
            }
        } else {
            // Read non-quantized data as float32
            file.read(reinterpret_cast<char*>(&data_size), sizeof(uint32_t));
            std::vector<float> param_data(data_size);
            file.read(reinterpret_cast<char*>(param_data.data()), data_size * sizeof(float));
            if (layer_index.has_value()) {
                model_data.layers[layer_index.value()].float_params[weight_name] = std::move(param_data);
            } else {
                model_data.non_layer_params[weight_name] = std::move(param_data);
            }
        }
    }

    file.close();
    return model_data;
}


#endif