import torch
import safetensors.torch
import struct
import numpy as np
import os

def weight_quant_true(weight, num_bits=2):
    weight = weight.float()
    scale = 1 / weight.abs().mean().clamp(min=1e-5)
    quantized = (weight * scale).round().clamp(-1, 1).to(torch.int8)  # Quantize to -1, 0, 1
    return quantized, scale

def pack_weights_to_int8(quantized_weight):
    # quantized_weight = quantized_weight.to("cpu")
    packed_shape = (quantized_weight.shape[1] // 4, quantized_weight.shape[0])
    quantized_weight_flat = quantized_weight.flatten()
    
    packed_weight = []
    for i in range(0, len(quantized_weight_flat), 4):
        byte = 0
        for j in range(4):
            if i + j < len(quantized_weight_flat):
                val = quantized_weight_flat[i + j].item()
                if val == 1:
                    val = 0b01
                elif val == -1:
                    val = 0b10
                else:
                    val = 0b00
                byte |= (val << (2 * j))  # Shift and combine
        packed_weight.append(byte)
    
    packed_weight = np.array(packed_weight, dtype=np.uint8)
    packed_weight = packed_weight.reshape(packed_shape)  # Reorder for C++ usage
    return packed_weight

def save_model_to_bin(safetensors_path, output_bin_path):
    model_data = safetensors.torch.load_file(safetensors_path)
    quantize_list = ["down_proj", "gate_proj", "up_proj", "q_proj", "k_proj", "v_proj", "o_proj"]

    with open(output_bin_path, 'wb') as f:
        for param_name, param_data in model_data.items():
            name_bytes = param_name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            # breakpoint()
            f.write(name_bytes)
            
            if "weight" in param_name and any(q in param_name for q in quantize_list):
                print(f"Quantizing and packing {param_name}")
                quantized, scale = weight_quant_true(param_data)
                packed_weight = pack_weights_to_int8(quantized)
                
                scale = scale.item() if isinstance(scale, torch.Tensor) else scale
                f.write(struct.pack('f', scale))  
                # breakpoint()
                f.write(struct.pack('I', packed_weight.size))  
                f.write(packed_weight.tobytes())  
            else:
                print(f"Directly saving {param_name}")
                param_data = param_data.numpy().astype(np.float32)
                f.write(struct.pack('I', param_data.size))
                f.write(param_data.tobytes())

if os.path.exists("model.bin"):
    os.remove("model.bin")
    # breakpoint()
save_model_to_bin('model.safetensors', 'model.bin')