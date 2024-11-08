#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <cstdint>
#include <stdexcept>
#include "load_model.h"

template <typename T>
std::vector<std::vector<T>> reshape_to_2d(const std::vector<T>& data, size_t rows, size_t cols) {
    if (data.size() != rows * cols) {
        throw std::invalid_argument("The size of data does not match the specified dimensions.");
    }

    std::vector<std::vector<T>> reshaped(rows, std::vector<T>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            reshaped[i][j] = data[i * cols + j];
        }
    }
    return reshaped;
}

template <typename T>
std::vector<std::vector<T>> transpose_2d(const std::vector<std::vector<T>>& data) {
    size_t rows = data.size();
    size_t cols = data[0].size();
    std::vector<std::vector<T>> transposed(cols, std::vector<T>(rows));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j){
            transposed[j][i] = data[i][j];
        }
    }
    return transposed;
}


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
    Shape2D reshape_size;
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
            std::pair<size_t, size_t> shape = reshape_size.get_shape(weight_name);
            std::vector<std::vector<uint8_t>> packed_data_2d = reshape_to_2d(packed_data, shape.first, shape.second);
            if (layer_index.has_value()) {
                model_data.layers[layer_index.value()].quantized_params[weight_name] = {scale, std::move(packed_data_2d)};
            } else {
                throw std::runtime_error("Quantized data must have a layer index. Missing layer index for parameter: " + weight_name);
            }
        } else {
            // Read non-quantized data as float32
            file.read(reinterpret_cast<char*>(&data_size), sizeof(uint32_t));
            std::vector<float> param_data(data_size);
            file.read(reinterpret_cast<char*>(param_data.data()), data_size * sizeof(float));
            try {
                std::pair<size_t, size_t> shape = reshape_size.get_shape(weight_name);
                std::vector<std::vector<float>> param_data_2d = reshape_to_2d(param_data, shape.first, shape.second);
                if (layer_index.has_value()) {
                    model_data.layers[layer_index.value()].float_params_2D[weight_name] = std::move(param_data_2d);
                } else {
                    model_data.non_layer_params_2D[weight_name] = std::move(param_data_2d);
                }
            } catch (const std::invalid_argument& e) {
                // std::cerr << weight_name << " is 1D "<< std::endl;
                if (layer_index.has_value()) {
                    model_data.layers[layer_index.value()].float_params_1D[weight_name] = std::move(param_data);
                } else {
                    model_data.non_layer_params_1D[weight_name] = std::move(param_data);
                }
            }
        }
    }
    //pay attention to add lm_head
    std::vector<std::vector<float>> lm_head = transpose_2d(model_data.non_layer_params_2D["embed_tokens"]);
    model_data.non_layer_params_2D["lm_head"] = std::move(lm_head);
    file.close();
    return model_data;
}