#include "linear_kernel.h"
#include "read_data.h"

int main() {
    // Dimensions for activation and weight matrices
    size_t activation_rows = 100;
    size_t activation_cols = 1536;
    size_t weight_rows = 1536;
    size_t weight_cols = 1536;
    
    // Explicitly declare the activation matrix
    std::vector<std::vector<float>> activation = readActivation("activation.bin", activation_rows, activation_cols);
    
    // Explicitly declare the packed weight matrix
    std::vector<std::vector<uint8_t>> packed_weight = readPackedWeight("packed_weight.bin", weight_cols, weight_rows / 4);

    // Quantize the input activations
    auto quantized_result = quantize_activation(activation, 8);
    std::vector<std::vector<int8_t>> quantized_input = quantized_result.first;
    std::vector<float> scales = quantized_result.second;

    // Call the forward function without bias (or bias=0)
    std::vector<std::vector<float>> result = linear_forward_no_mul(quantized_input, scales, packed_weight, weight_rows, weight_cols);

    // Print a small portion of the result for verification
    for (size_t i = 0; i < std::min<size_t>(result.size(), 5); ++i) {
        for (size_t j = 0; j < std::min<size_t>(result[0].size(), 5); ++j) {
            std::cout << result[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}