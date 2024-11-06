#include "read_data.h"
#include "../src_c/attention.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

std::vector<std::vector<float>> readResult(const std::string &file_path, size_t rows, size_t cols) {
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols, 0.0f));
    std::ifstream in_file(file_path, std::ios::binary);
    if (!in_file) {
        std::cerr << "Error opening file for reading: " << file_path << std::endl;
        return result;
    }
    for (auto &row : result) {
        in_file.read(reinterpret_cast<char *>(row.data()), row.size() * sizeof(float));
    }
    in_file.close();
    return result;
}

bool compareResults(const std::vector<std::vector<float>> &result1, const std::vector<std::vector<float>> &result2, float tolerance = 1e-5) {
    if (result1.size() != result2.size() || result1[0].size() != result2[0].size()) {
        std::cerr << "Error: Results have different sizes." << std::endl;
        return false;
    }

    for (size_t i = 0; i < result1.size(); ++i) {
        for (size_t j = 0; j < result1[0].size(); ++j) {
            if (std::abs(result1[i][j] - result2[i][j]) > tolerance) {
                std::cerr << "Mismatch at (" << i << ", " << j << "): " << result1[i][j] << " vs " << result2[i][j] << std::endl;
                return false;
            }
        }
    }
    return true;
}

int main() {
    // Model parameter sizes
    size_t hidden_size = 1536;
    size_t num_heads = 24;
    size_t head_dim = hidden_size / num_heads;
    size_t seq_len = 100;

    // Loop over 5 input sets
    for (int set_idx = 1; set_idx <= 5; ++set_idx) {
        std::string dir_name = "data/input_set_" + std::to_string(set_idx);
        
        // Read activation and weights from binary files
        auto hidden_states = readActivation(dir_name + "/activation.bin", seq_len, hidden_size);
        auto q_weights = readPackedWeights(dir_name + "/q_weight.bin", hidden_size, hidden_size / 4);
        auto k_weights = readPackedWeights(dir_name + "/k_weight.bin", hidden_size, hidden_size / 4);
        auto v_weights = readPackedWeights(dir_name + "/v_weight.bin", hidden_size, hidden_size / 4);
        auto o_weights = readPackedWeights(dir_name + "/o_weight.bin", hidden_size, hidden_size / 4);

        // Read scaling factor
        float q_scale = readScale(dir_name + "/q_weight_scale.bin");
        float k_scale = readScale(dir_name + "/k_weight_scale.bin");
        float v_scale = readScale(dir_name + "/v_weight_scale.bin");
        float o_scale = readScale(dir_name + "/o_weight_scale.bin");

        // Read inv_freq and ln_weight
        auto inv_freq = read1DArray(dir_name + "/inv_freq.bin", head_dim / 2);
        auto ln_weight = read1DArray(dir_name + "/ln_weight.bin", hidden_size);

        // Call the attention function
        auto result = bitnet_attention(hidden_states, q_weights, k_weights, v_weights, o_weights, 
                                       q_scale, k_scale, v_scale, o_scale, inv_freq, ln_weight,
                                       hidden_size, num_heads, head_dim, seq_len);

        //Read the golden result from the golden_result folder
        std::string result_file = "data/golden_results/result_" + std::to_string(set_idx) + ".bin";
        auto golden_result = readResult(result_file, seq_len, hidden_size);

        // Compare results
        if (compareResults(result, golden_result)) {
            std::cout << "Result for set " << set_idx << " matches the golden result." << std::endl;
        } else {
            std::cerr << "Result for set " << set_idx << " does not match the golden result." << std::endl;
        }
    }

    return 0;
}
