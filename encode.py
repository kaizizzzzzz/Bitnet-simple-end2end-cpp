from tokenization_bitnet import BitnetTokenizer 
import argparse
import numpy
import os

# Delete the file if it already exists
if os.path.exists("encoded_prompt.bin"):
    os.remove("encoded_prompt.bin")

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str)

args = parser.parse_args()
tokenizer = BitnetTokenizer.from_pretrained('1bitLLM/bitnet_b1_58-large', use_fast=False, cache_dir="tokenizer_model")
id = tokenizer(args.prompt, return_tensors="pt").input_ids
id = id[0].cpu().numpy().astype(numpy.uint64)
id.tofile("encoded_prompt.bin")