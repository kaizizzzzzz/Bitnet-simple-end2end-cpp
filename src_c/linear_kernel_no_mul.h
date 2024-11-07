#ifndef LINEAR_KERNEL_H
#define LINEAR_KERNEL_H

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstdint>
#include "bitnet.h"

// Function to quantize activation to 8 bits
std::pair<std::vector<std::vector<int8_t>>, std::vector<float>> quantize_activation(const std::vector<std::vector<float>> &input, int num_bits); 

// Forward function using column-based packing of weights with activation quantization
std::vector<std::vector<float>> linear_forward_no_mul(const std::vector<std::vector<int8_t>> &input,
                                                      const std::vector<float> scales,
                                                      const QuantizedData &weights, 
                                                      size_t weight_cols
                                                      ); 
#endif // LINEAR_KERNEL_H