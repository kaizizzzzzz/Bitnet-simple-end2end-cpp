import numpy as np

# Verify the generated activation.bin
activation = np.fromfile("data/input_set_1/activation.bin", dtype=np.float32).reshape((100, 1536))
print(activation[:2, :2])  # Print a small part of the matrix to verify non-zero values
