from tokenization_bitnet import BitnetTokenizer 
import argparse
import numpy
import torch

output_id = numpy.fromfile("generated_ids.bin", dtype=numpy.uint64)
output_id = torch.tensor(output_id)
if output_id[0] == 1: #start token
    output_id = output_id[1:]
tokenizer = BitnetTokenizer.from_pretrained('1bitLLM/bitnet_b1_58-large', use_fast=False, cache_dir="tokenizer_model")
tokens = tokenizer.decode(output_id)
print(tokens)