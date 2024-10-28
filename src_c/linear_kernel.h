#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstdint>

// Function to quantize activation to 8 bits
std::pair<std::vector<std::vector<int8_t>>, std::vector<float>> quantize_activation(const std::vector<std::vector<float>> &input, int num_bits) {
    int Qn = -(1 << (num_bits - 1));
    int Qp = (1 << (num_bits - 1)) - 1;
    size_t rows = input.size();
    size_t cols = input[0].size();

    std::vector<std::vector<int8_t>> quantized_input(rows, std::vector<int8_t>(cols, 0));
    std::vector<float> scales(rows, 1.0f);

    for (size_t i = 0; i < rows; ++i) {
        // Calculate the scale for each row
        float max_val = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            max_val = std::max(max_val, std::abs(input[i][j]));
        }
        float scale = Qp / std::max(max_val, 1e-5f);
        scales[i] = scale;

        // Quantize each element in the row
        for (size_t j = 0; j < cols; ++j) {
            int32_t quantized_value = static_cast<int32_t>(std::round(input[i][j] * scale));
            int8_t quantized_value_clamped = (quantized_value < Qn) ? Qn : ((quantized_value > Qp) ? Qp : quantized_value);
            quantized_input[i][j] = static_cast<int8_t>(quantized_value_clamped);
        }
    }

    return {quantized_input, scales};
}

// Function to extract 2-bit weight from packed data (column-based)
int8_t extract_weight_col(uint8_t packed_weight, int index) {
    int shift = 2 * index;
    int8_t value = (packed_weight >> shift) & 0b11;  // Extract the 2-bit segment

    // Decode the 2-bit value
    switch (value) {
        case 0b00: return 0;   // Encoded as 00
        case 0b01: return 1;   // Encoded as 01
        case 0b10: return -1;  // Encoded as 10
        default: return 0;     // Undefined values default to 0
    }
}

// Forward function using column-based packing of weights with activation quantization
std::vector<std::vector<float>> linear_forward_no_mul(const std::vector<std::vector<int8_t>> &input,
                                                      const std::vector<float> scales,
                                                      const std::vector<std::vector<uint8_t>> &packed_weight_col,
                                                      size_t weight_rows, size_t weight_cols
                                                      ) {
    size_t input_rows = input.size();
    size_t input_cols = input[0].size();
    size_t output_cols = weight_cols;

    // Initialize output matrix
    std::vector<std::vector<float>> output(input_rows, std::vector<float>(output_cols, 0.0f));

    // Perform matrix multiplication using column-based packing
    for (size_t i = 0; i < input_rows; ++i) {
        for (size_t j = 0; j < output_cols; ++j) {
            for (size_t k = 0; k < input_cols; k += 4) {
                // Load packed weights for the current column j
                uint8_t packed_value = packed_weight_col[j][k / 4];

                // Extract and decode the weights in this column block
                for (int l = 0; l < 4 && (k + l) < input_cols; ++l) {
                    int8_t weight_value = extract_weight_col(packed_value, l);

                    // Add or subtract based on weight_value, using integer GEMM
                    output[i][j] += input[i][k + l] * weight_value;
                }
            }

            // Dequantize the output by dividing by the scale
            output[i][j] /= scales[i];
        }
    }

    return output;
}


