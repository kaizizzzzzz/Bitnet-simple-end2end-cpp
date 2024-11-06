import torch
import safetensors.torch
import struct
import numpy as np
import os

# Code for reading
def read_model_from_bin(bin_path):
    model_data = {"float_params": {}, "quantized_params": {}}

    with open(bin_path, 'rb') as f:
        while True:
            name_len_bytes = f.read(4)
            if not name_len_bytes:
                break
            name_len = struct.unpack('I', name_len_bytes)[0]

            name = f.read(name_len).decode('utf-8')
            is_quantized = any(q in name for q in ["down_proj", "gate_proj", "up_proj", "q_proj", "k_proj", "v_proj", "o_proj"])

            if is_quantized:
                scale = struct.unpack('f', f.read(4))[0]
                data_size = struct.unpack('I', f.read(4))[0]
                packed_data = np.frombuffer(f.read(data_size), dtype=np.uint8)
                model_data["quantized_params"][name] = {"scale": scale, "packed_data": packed_data}
                print(f"Loaded quantized param: {name}, Scale: {scale}, Packed Data Size: {len(packed_data)}")
                # breakpoint()
            else:
                data_size = struct.unpack('I', f.read(4))[0]
                param_data = np.frombuffer(f.read(data_size * 4), dtype=np.float32)
                model_data["float_params"][name] = param_data
                print(f"Loaded float param: {name}, Size: {len(param_data)}")

    return model_data

model_data = read_model_from_bin("model.bin")