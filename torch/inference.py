import math
import argparse
import torch
import random

from modeling_bitnet import BitnetForCausalLM
from tokenization_bitnet import BitnetTokenizer 

from tqdm import tqdm
import torch.nn.functional as F
import time
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', default='1bitLLM/bitnet_b1_58-large', type=str)
parser.add_argument('--seqlen', default=2048, type=int)
parser.add_argument('--input', type=str)

def generate_tokens(model, input, top_k=50, temperature=0.8, max_length=30):
    time_s = time.time()
    for _ in range(max_length):
        output = model(input,
                        use_cache=True,
                        output_hidden_states=False,
                        output_attentions=False)[0]
        # Sampling parameters
        logits = output[:, -1, :] / temperature
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        probs = F.softmax(top_k_logits, dim=-1)
        # breakpoint()
        # Sample from the distribution
        next_token_id = top_k_indices[0][torch.multinomial(probs, num_samples=1)]
        generated_ids = torch.cat([input, next_token_id], dim=1)
        input = generated_ids
    time_e = time.time()
    time_cost = time_e - time_s
    print(f"latency per token: {time_cost / max_length}s")
    return input

def generate_tokens_cache(model, input, top_k=50, temperature=0.8, max_length=30):
    time_s = time.time()
    for _ in range(max_length):
        if _ == 0:
            outputs = model(input,
                            use_cache=True,
                            output_hidden_states=False,
                            output_attentions=False)
            output, cache = outputs[0], outputs[1]
            # breakpoint()
        else:
            outputs = model(generated_ids,
                            use_cache=True,
                            output_hidden_states=False,
                            output_attentions=False,
                            past_key_values=cache)
            output, cache = outputs[0], outputs[1]
        # Sampling parameters
        logits = output[:, -1, :] / temperature
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        probs = F.softmax(top_k_logits, dim=-1)

        # Sample from the distribution
        next_token_id = top_k_indices[0][torch.multinomial(probs, num_samples=1)]
        generated_ids = torch.cat([input, next_token_id], dim=1)
        input = generated_ids
    time_e = time.time()
    time_cost = time_e - time_s
    print(f"latency per token: {time_cost / max_length}s")
    return input

def main(args):
    # breakpoint()
    model = BitnetForCausalLM.from_pretrained(
        args.hf_path,
        device_map='auto',
        low_cpu_mem_usage=True, 
        use_flash_attention_2=False,
        torch_dtype=torch.float16,
        cache_dir="/work/zhang-capra/users/ky427"
    ).half()
    # breakpoint()
    tokenizer = BitnetTokenizer.from_pretrained(args.hf_path, use_fast=False, cache_dir="/work/zhang-capra/users/ky427")
    # id1 = tokenizer.encode(args.input, add_special_tokens=False) 
    # id2 = tokenizer(args.input, add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
    id = tokenizer(args.input, return_tensors="pt").input_ids
    # breakpoint()
    # id1 = torch.tensor(id1).cuda().view(1, -1)
    # output = model.generate(input_ids=torch.tensor([id]).cuda(), max_length=100)
    # output = model(id2,
    #                 use_cache=True,
    #                 output_hidden_states=False,
    #                 output_attentions=False)[0]
    # breakpoint()
    # # breakpoint()
    # # Sampling parameters
    # top_k = args.top_k if hasattr(args, 'top_k') else 50
    # temperature = args.temperature if hasattr(args, 'temperature') else 0.8

    # # Apply temperature scaling and top-k sampling
    # # breakpoint()
    # logits = output[:, -1, :] / temperature
    # top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
    # probs = F.softmax(top_k_logits, dim=-1)

    # # Sample from the distribution
    # next_token_id = top_k_indices[0][torch.multinomial(probs, num_samples=1)]
    # generated_ids = torch.cat([id2, next_token_id], dim=1)

    # # Decode to text
    # generated_text = tokenizer.decode(generated_ids[0])
    # print(generated_text)
    generated_ids = generate_tokens(model, id, max_length=30)
    generated_text = tokenizer.decode(generated_ids[0])
    print(generated_text)

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)