#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include "../src_c/bitnet.h"
#include "../src_c/embedding.h"
#include "../model_config.h"
#include "load_model.h"

std::vector<float> get_encoded_id(const std::string &file_name){
    std::vector<float> encoded_id;
    std::ifstream file(file_name, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Error opening file: " + file_name);
    }

    // Get the file size
    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);


    // Calculate the number of float elements in the file
    size_t num_elements = file_size / sizeof(float);
    encoded_id.resize(num_elements);

    // Read the data into the vector
    if (!file.read(reinterpret_cast<char*>(encoded_id.data()), file_size)) {
        throw std::runtime_error("Error reading file: " + file_name);
    }

    if (encoded_id.empty()) {
        throw std::runtime_error("No data read from file: " + file_name);
    }
    if (encoded_id[0] != 1) {
        throw std::runtime_error("Should start with encode 1 for </s> " + file_name);
    }

    file.close();

    std::cout << "Encoded_ID: ";
    for (size_t i = 0; i < encoded_id.size(); ++i) {
        std::cout << encoded_id[i] << " ";
    }
    std::cout << std::endl;

    return encoded_id;
}

void inference(){

}