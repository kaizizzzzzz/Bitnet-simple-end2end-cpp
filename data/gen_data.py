import numpy as np
import os

# Set the random seed for reproducibility
np.random.seed(42)

# Model parameter sizes
hidden_size = 1536
num_heads = 24
head_dim = hidden_size // num_heads
seq_len = 100

# Number of input sets
num_sets = 5

for set_idx in range(1, num_sets + 1):
    # Create a directory for each set
    dir_name = f"data/input_set_{set_idx}"
    os.makedirs(dir_name, exist_ok=True)

    # Initialize activation matrix (100 x 1536) with random float values
    activation = np.random.uniform(-1.0, 1.0, (seq_len, hidden_size)).astype(np.float32)

    # Initialize weight matrices (1536 x 1536) for Q, K, V, O with random values in the range of -1, 0, and 1
    weights = {}
    for key in ['q_weight', 'k_weight', 'v_weight', 'o_weight']:
        weight = np.random.choice([-1, 0, 1], size=(hidden_size, hidden_size)).astype(np.int8)
        packed_weight = np.zeros((hidden_size, hidden_size // 4), dtype=np.uint8)

        # Pack weight matrix into 2-bit values, 4 per byte (8 bits)
        for col in range(hidden_size):
            for row in range(0, hidden_size, 4):
                # Pack 4 weights into a single byte
                byte_value = (weight[row, col] & 0b11) | \
                             ((weight[row + 1, col] & 0b11) << 2) | \
                             ((weight[row + 2, col] & 0b11) << 4) | \
                             ((weight[row + 3, col] & 0b11) << 6)
                packed_weight[col, row // 4] = byte_value

        weights[key] = packed_weight

    # Save activation and packed weights to binary files in their respective directories
    activation.tofile(os.path.join(dir_name, "activation.bin"))
    for key, packed_weight in weights.items():
        packed_weight.tofile(os.path.join(dir_name, f"{key}.bin"))

    # Create and save inv_freq and ln_weight
    inv_freq = (1.0 / np.power(10000.0, (2.0 * np.arange(0, head_dim, 2, dtype=np.float32)) / head_dim)).astype(np.float32)
    inv_freq.tofile(os.path.join(dir_name, "inv_freq.bin"))

    ln_weight = np.ones(hidden_size, dtype=np.float32)
    ln_weight.tofile(os.path.join(dir_name, "ln_weight.bin"))
