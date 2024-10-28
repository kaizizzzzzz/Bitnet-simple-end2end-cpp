#include <fstream>
#include <iostream>
#include <vector>
#include <string>

// Helper function to read binary files into a 2D vector
std::vector<std::vector<float>> readActivation(const std::string &filepath, size_t rows, size_t cols) {
    std::ifstream file(filepath, std::ios::binary);
    std::vector<std::vector<float>> data(rows, std::vector<float>(cols));
    for (size_t i = 0; i < rows; ++i) {
        file.read(reinterpret_cast<char*>(data[i].data()), cols * sizeof(float));
    }
    return data;
}

// Helper function to read packed weights
std::vector<std::vector<uint8_t>> readPackedWeights(const std::string &filepath, size_t rows, size_t cols) {
    std::ifstream file(filepath, std::ios::binary);
    std::vector<std::vector<uint8_t>> data(rows, std::vector<uint8_t>(cols));
    for (size_t i = 0; i < rows; ++i) {
        file.read(reinterpret_cast<char*>(data[i].data()), cols * sizeof(uint8_t));
    }
    return data;
}

// Helper function to read 1D binary data
std::vector<float> read1DArray(const std::string &filepath, size_t size) {
    std::ifstream file(filepath, std::ios::binary);
    std::vector<float> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size * sizeof(float));
    return data;
}

// Helper function to read a single float scale from a binary file
float readScale(const std::string &filepath) {
    std::ifstream file(filepath, std::ios::binary);
    float scale;
    file.read(reinterpret_cast<char*>(&scale), sizeof(float));
    return scale;
}
