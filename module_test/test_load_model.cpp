#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include "../inference/load_model.h"

//verify
void print_model_data(const ModelData& model_data) {
    // Print non-layer parameters
    std::cout << "Non-layer parameters:\n";
    for (const auto& [param_name, param_data] : model_data.non_layer_params) {
        std::cout << "  Param: " << param_name << " - Size: " << param_data.size() << "\n";
        if (!param_data.empty()) {
            std::cout << "    First value: " << param_data[0] << "\n"; // Example of accessing data
        }
    }

    // Print layer-specific parameters
    for (const auto& [layer_index, layer_data] : model_data.layers) {
        std::cout << "\nLayer " << layer_index << ":\n";

        // Print float parameters
        for (const auto& [param_name, param_data] : layer_data.float_params) {
            std::cout << "  Float Param: " << param_name << " - Size: " << param_data.size() << "\n";
            if (!param_data.empty()) {
                std::cout << "    First value: " << param_data[0] << "\n";
            }
        }

        // Print quantized parameters
        for (const auto& [param_name, quantized_data] : layer_data.quantized_params) {
            std::cout << "  Quantized Param: " << param_name << "\n";
            std::cout << "    Scale: " << quantized_data.scale << "\n";
            std::cout << "    Packed Data Size: " << quantized_data.packed_data.size() << "\n";
            if (!quantized_data.packed_data.empty()) {
                std::cout << "    First byte of packed data: "
                          << static_cast<int>(quantized_data.packed_data[0]) << "\n"; // Display first byte
            }
        }
    }
}

int main() {
    // Path to your binary model file
    std::string input_bin_path = "model.bin";

    // Load model data from binary file
    ModelData model_data = load_model_from_bin(input_bin_path);

    // Print out the loaded model data for verification
    print_model_data(model_data);

    return 0;
}
