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
#include "../model_config.h"

struct QuantizedData {
    float scale;
    std::vector<std::vector<uint8_t>> packed_data;
};

struct LayerData {
    std::unordered_map<std::string, std::vector<float>> float_params_1D;
    std::unordered_map<std::string, std::vector<std::vector<float>>> float_params_2D;
    std::unordered_map<std::string, QuantizedData> quantized_params;
};

struct ModelData {
    std::unordered_map<int, LayerData> layers;
    std::unordered_map<std::string, std::vector<float>> non_layer_params_1D;
    std::unordered_map<std::string, std::vector<std::vector<float>>> non_layer_params_2D;
};


ModelData load_model_from_bin(const std::string& input_bin_path); 

#endif