import math
import torch
from torch import nn


def weight_quant(weight, num_bits=1):
    dtype = weight.dtype
    weight = weight.float()
    s =  1 / weight.abs().mean().clamp(min=1e-5)
    result = (weight * s).round().clamp(-1, 1) 
    # breakpoint()
    result = result / s
    return result.type(dtype)

def weight_quant_true(weight, num_bits=1):
    dtype = weight.dtype
    weight = weight.float()
    s =  1 / weight.abs().mean().clamp(min=1e-5)
    result = (weight * s).round().clamp(-1, 1) 
    return result.float(), s


def activation_quant(x, num_bits=8):
    dtype = x.dtype
    x = x.float()
    Qn = -2 ** (num_bits - 1)
    Qp = 2 ** (num_bits - 1) - 1
    s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * s).round().clamp(Qn, Qp) 
    result = result / s
    # breakpoint()
    return result.type(dtype)   

def activation_quant_true(x, num_bits=8):
    dtype = x.dtype
    x = x.float()
    Qn = -2 ** (num_bits - 1)
    Qp = 2 ** (num_bits - 1) - 1
    s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * s).round().clamp(Qn, Qp) 
    # breakpoint()
    return result.type(dtype).float(), s

import torch

def batched_low_bit_gemm(act, act_s, weight, weight_s):
    """
    Performs low-bit GEMM on quantized `act` and `weight`, then dequantizes the result.
    
    Parameters:
    - act: Quantized activations of shape [batch, l, d] (int8 or int4).
    - act_s: Scale factors for activations of shape [batch, l, 1].
    - weight: Quantized weights of shape [d, d] (int8 or int4).
    - weight_s: Scalar scale factor for the weight matrix.
    
    Returns:
    - Dequantized result of shape [batch, l, d].
    """

    # Step 1: Perform integer GEMM on quantized tensors
    # Result will be in integer form, with shape [batch, l, d]
    quantized_result = torch.bmm(act, weight)  # Shape: [batch, l, d]

    # Step 2: Dequantize the result
    # Apply both scaling factors: (act_s * weight_s)
    # Since act_s is [batch, l, 1], broadcasting will apply it across the last dimension
    # breakpoint ()
    dequantized_result = quantized_result.float() / (act_s * weight_s)

    return dequantized_result.half()  # Shape: [batch, l, d]


class BitLinear(nn.Linear):

    def __init__(self,
            *kargs,
            weight_bits=1,
            input_bits=8,
            **kwargs
        ):
        super(BitLinear, self).__init__(*kargs, **kwargs)
        """
        RMSNorm is placed outside BitLinear
        """
        self.weight_bits = weight_bits
        self.input_bits = input_bits

    def forward(self, input):
        # breakpoint()
        # quant_input = input + (activation_quant(input, self.input_bits) - input).detach()
        # quant_weight = self.weight + (weight_quant(self.weight, self.weight_bits) - self.weight).detach()
        # out1 = nn.functional.linear(quant_input, quant_weight)
        # breakpoint()
        # quant_input = activation_quant(input, self.input_bits) 
        # quant_weight = weight_quant(self.weight, self.weight_bits)
        # # out = nn.functional.linear(quant_input, quant_weight)
        # out = torch.bmm(quant_input, quant_weight.T.unsqueeze(0))
        # breakpoint()
        input_q, input_s = activation_quant_true(input, self.input_bits)
        weight_q, weight_s = weight_quant_true(self.weight, self.weight_bits)
        weight_q = weight_q.T.unsqueeze(0)
        # breakpoint()
        # breakpoint()
        # out_c = batched_low_bit_gemm(input_q, input_s, weight_q, weight_s)
        out = batched_low_bit_gemm(input_q, input_s, weight_q, weight_s)
        # tolerance = 1e-2
        # are_close = torch.allclose(out1, out, atol=tolerance, rtol=tolerance)
        # if are_close:
        #     print("out1 and out are approximately equal within the tolerance.")
        # else:
        #     print("out1 and out are NOT approximately equal.")
        #     # Optionally, print the difference to understand where they diverge
        #     print("Max difference:", (out1 - out).abs().max())
        # assert are_close
        # breakpoint()
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out