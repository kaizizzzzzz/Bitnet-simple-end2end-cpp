import numpy as np

def unpack_weights(packed_weights, weight_rows, weight_cols):
    """
    Unpack the 2-bit packed weights into a full weight matrix.
    
    Parameters:
    - packed_weights: A numpy array of shape (weight_cols, weight_rows // 4) with packed 2-bit values.
    - weight_rows: The number of rows in the unpacked weight matrix.
    - weight_cols: The number of columns in the unpacked weight matrix.

    Returns:
    - unpacked_weights: A numpy array of shape (weight_rows, weight_cols) with values -1, 0, or 1.
    """
    unpacked_weights = np.zeros((weight_rows, weight_cols), dtype=np.int8)

    for col in range(weight_cols):
        for row in range(0, weight_rows, 4):
            packed_value = packed_weights[col, row // 4]

            # Extract 2-bit values for 4 elements in this column
            for l in range(4):
                shift = 2 * l
                value = (packed_value >> shift) & 0b11
                
                # Decode the 2-bit value
                if value == 0b01:
                    unpacked_weights[row + l, col] = 1
                elif value == 0b10:
                    unpacked_weights[row + l, col] = -1
                else:
                    unpacked_weights[row + l, col] = 0

    return unpacked_weights

def linear_forward_column_python(activation, packed_weights, weight_rows, weight_cols):
    """
    Perform the GEMM using the unpacked weight matrix in Python.

    Parameters:
    - activation: A numpy array of shape (100, 1536).
    - packed_weights: A numpy array of shape (1536, 384) with packed 2-bit values.
    - weight_rows: The number of rows in the unpacked weight matrix.
    - weight_cols: The number of columns in the unpacked weight matrix.

    Returns:
    - result: The resulting matrix after GEMM, of shape (100, 1536).
    """
    # Unpack the 2-bit weights
    weight = unpack_weights(packed_weights, weight_rows, weight_cols)

    # Perform matrix multiplication using NumPy's dot product
    result = np.dot(activation, weight)

    return result

# Load activation and packed weights from the binary files
activation = np.fromfile("activation.bin", dtype=np.float32).reshape(100, 1536)
packed_weights = np.fromfile("packed_weight.bin", dtype=np.uint8).reshape(1536, 1536 // 4)

# Perform the GEMM using the Python function
result_python = linear_forward_column_python(activation, packed_weights, 1536, 1536)

# Print a small portion of the result for verification
print(result_python[:5, :5])
