#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>

struct QuantizedData {
    float scale;
    std::vector<int8_t> packed_data;
};

struct ModelData {
    std::unordered_map<std::string, std::vector<float>> float_params;
    std::unordered_map<std::string, QuantizedData> quantized_params;
};

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

        // Read data size
        uint32_t data_size;
        file.read(reinterpret_cast<char*>(&data_size), sizeof(uint32_t));

        // Check if the parameter is quantized
        bool is_quantized = (name.find("down_proj") != std::string::npos ||
                             name.find("gate_proj") != std::string::npos ||
                             name.find("up_proj") != std::string::npos ||
                             name.find("q_proj") != std::string::npos ||
                             name.find("k_proj") != std::string::npos ||
                             name.find("v_proj") != std::string::npos ||
                             name.find("o_proj") != std::string::npos);

        if (is_quantized) {
            // Read scale
            float scale;
            file.read(reinterpret_cast<char*>(&scale), sizeof(float));

            // Read packed data
            std::vector<int8_t> packed_data(data_size);
            file.read(reinterpret_cast<char*>(packed_data.data()), data_size);

            model_data.quantized_params[name] = {scale, std::move(packed_data)};
        } else {
            // Read non-quantized data as float32
            std::vector<float> param_data(data_size);
            file.read(reinterpret_cast<char*>(param_data.data()), data_size * sizeof(float));
            model_data.float_params[name] = std::move(param_data);
        }
    }

    file.close();
    return model_data;
}

int main() {
    ModelData model_data = load_model_from_bin("model.bin");

    // Example: accessing the loaded data
    for (const auto& [name, data] : model_data.float_params) {
        std::cout << "Float param name: " << name << " Size: " << data.size() << std::endl;
    }

    for (const auto& [name, quant_data] : model_data.quantized_params) {
        std::cout << "Quantized param name: " << name << " Scale: " << quant_data.scale 
                  << " Packed data size: " << quant_data.packed_data.size() << std::endl;
    }

    return 0;
}
