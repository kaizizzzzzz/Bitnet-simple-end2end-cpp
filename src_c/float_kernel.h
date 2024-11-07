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
Tensor3D GEMM_3D_float(const Tensor3D &tensor1, const Tensor3D &tensor2);

// Matrix multiplication for 2D tensors
// This function multiplies tensors of shape [seq_len, intermediate_dim] with [intermediate_dim, intermediate_dim]
Tensor2D GEMM_2D_float(const Tensor2D &tensor1, const Tensor2D &tensor2); 

// Element multiplication for 2D tensors
// This function multiplies tensors in element of shape [seq_len, intermediate_dim] with [seq_len, intermediate_dim]
Tensor2D element_mul_2D_float(const Tensor2D &tensor1, const Tensor2D &tensor2); 

#endif // FLOAT_KERNEL_H