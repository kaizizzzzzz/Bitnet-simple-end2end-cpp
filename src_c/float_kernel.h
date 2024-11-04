#ifndef FLOAT_KERNEL_H
#define FLOAT_KERNEL_H

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstdint>
#include "bitnet.h"

// Matrix multiplication for 3D tensors
// This function multiplies tensors of shape [num_heads, seq_len, head_dim] with [num_heads, head_dim, seq_len]
Tensor3D GEMM_3D_float(const Tensor3D &tensor1, const Tensor3D &tensor2) {
    // Validate dimensions
    size_t num_heads1 = tensor1.size();
    size_t seq_len1 = tensor1[0].size();
    size_t head_dim1 = tensor1[0][0].size();

    size_t num_heads2 = tensor2.size();
    size_t head_dim2 = tensor2[0].size();  // This should match the last dimension of tensor1
    size_t seq_len2 = tensor2[0][0].size();

    if (num_heads1 != num_heads2) {
        throw std::runtime_error("Number of heads in tensor1 and tensor2 must match.");
    }
    if (head_dim1 != head_dim2) {
        throw std::runtime_error("Head dimension of tensor1 and head dimension of tensor2 must match.");
    }

    size_t num_heads = num_heads1;
    size_t seq_len = seq_len1;
    size_t head_dim = head_dim1;

    // Initialize result tensor with the correct dimensions
    Tensor3D result(num_heads, std::vector<std::vector<float>>(seq_len, std::vector<float>(seq_len2, 0.0f)));

    // Perform matrix multiplication
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < seq_len2; ++j) {
                for (size_t k = 0; k < head_dim; ++k) {
                    result[h][i][j] += tensor1[h][i][k] * tensor2[h][k][j];
                }
            }
        }
    }

    return result;
}

// Element multiplication for 2D tensors
// This function multiplies tensors in element of shape [seq_len, intermediate_dim] with [seq_len, intermediate_dim]
Tensor2D element_mul_2D_float(const Tensor2D &tensor1, const Tensor2D &tensor2) {
    // Validate dimensions
    size_t seq_len1 = tensor1.size();
    size_t intermediate_dim1 = tensor1[0].size();

    size_t seq_len2 = tensor2.size();
    size_t intermediate_dim2 = tensor2[0].size();  // This should match the last dimension of tensor1

    if (seq_len1 != seq_len2) {
        throw std::runtime_error("sequence length in tensor1 and tensor2 must match.");
    }
    if (intermediate_dim1 != intermediate_dim2) {
        throw std::runtime_error("intermediate_dim of tensor1 and head dimension of tensor2 must match.");
    }

    size_t seq_len = seq_len1;
    size_t intermediate_dim = intermediate_dim1;

    // Initialize result tensor with the correct dimensions
    Tensor2D result(seq_len, std::vector<float>(intermediate_dim, 0.0f));

    // Perform element multiplication
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < intermediate_dim; ++j) {
            result[i][j] = tensor1[i][j] * tensor2[i][j];
        }
    }

    return result;
}


#endif // FLOAT_KERNEL_H