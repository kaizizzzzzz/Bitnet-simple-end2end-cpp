from tokenization_bitnet import BitnetTokenizer 
import argparse
import numpy

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str)

args = parser.parse_args()
tokenizer = BitnetTokenizer.from_pretrained('1bitLLM/bitnet_b1_58-large', use_fast=False, cache_dir="tokenizer_model")
id = tokenizer(args.prompt, return_tensors="pt").input_ids
id = id[0].cpu().numpy()
id.tofile("encoded_prompt.bin")