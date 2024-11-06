#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include "../inference/load_model.h"

// Verify and print model data
void print_model_data(const ModelData& model_data) {
    // Print non-layer 1D parameters
    std::cout << "Non-layer 1D parameters:\n";
    for (const auto& [param_name, param_data] : model_data.non_layer_params_1D) {
        std::cout << "  Param: " << param_name << " - Size: " << param_data.size() << "\n";
        if (!param_data.empty()) {
            std::cout << "    First value: " << param_data[0] << "\n"; // Example of accessing data
        }
    }

    // Print non-layer 2D parameters
    std::cout << "\nNon-layer 2D parameters:\n";
    for (const auto& [param_name, param_data] : model_data.non_layer_params_2D) {
        std::cout << "  Param: " << param_name << " - Size: [" << param_data.size() << "]["
                  << (param_data[0].empty() ? 0 : param_data[0].size()) << "]\n";
        if (!param_data.empty() && !param_data[0].empty()) {
            std::cout << "    First value: " << param_data[0][0] << "\n"; // Example of accessing data
        }
    }

    // Print layer-specific parameters
    for (const auto& [layer_index, layer_data] : model_data.layers) {
        std::cout << "\nLayer " << layer_index << ":\n";

        // Print 1D float parameters
        for (const auto& [param_name, param_data] : layer_data.float_params_1D) {
            std::cout << "  Float 1D Param: " << param_name << " - Size: " << param_data.size() << "\n";
            if (!param_data.empty()) {
                std::cout << "    First value: " << param_data[0] << "\n";
            }
        }

        // Print 2D float parameters
        for (const auto& [param_name, param_data] : layer_data.float_params_2D) {
            std::cout << "  Param: " << param_name << " - Size: [" << param_data.size() << "]["
                    << (param_data[0].empty() ? 0 : param_data[0].size()) << "]\n";
            if (!param_data.empty() && !param_data[0].empty()) {
                std::cout << "    First value: " << param_data[0][0] << "\n";
            }
        }

        // Print quantized parameters
        for (const auto& [param_name, quantized_data] : layer_data.quantized_params) {
            std::cout << "  Quantized Param: " << param_name << "\n";
            std::cout << "    Scale: " << quantized_data.scale << "\n";
            
            // Print the dimensions of the packed_data 2D vector
            std::cout << "    Packed Data Dimensions: [" << quantized_data.packed_data.size() << "]["
                    << (quantized_data.packed_data[0].empty() ? 0 : quantized_data.packed_data[0].size()) << "]\n";
            
            // Safely print the first byte of the packed data if available
            if (!quantized_data.packed_data.empty() && !quantized_data.packed_data[0].empty()) {
                std::cout << "    First byte of packed data: "
                        << static_cast<int>(quantized_data.packed_data[0][0]) << "\n"; // Display first byte
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
