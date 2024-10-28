# Bitnet-attn

This repository contains a simplified C++ implementation of the attention mechanism for the `1bitLLM/bitnet_b1_58-large` model. This version uses quantization techniques and random weight initialization to provide a basic framework without the complex optimization for specific CPU Architecture 

## Key Features
- **Attention Mechanism without Multiplication:** Implements core bitnet-attention logic without traditional multiplication.
- **Eager Attention:** Doesn't support flash_attention right now.
- **8-bit Activation Quantization:** Efficiently quantizes activations using 8-bit precision.
- **Random Weights & Embeddings:** Test using random initialization of weights, rotary embeddings,RMSNorm weights, and random activations.

## Dependencies
- C++11 compiler

## How to Build and Run

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/kaizizzzzzz/Bitnet-attn.git
   cd Bitnet-attn
2. **Compile the Code:**
g++ -std=c++11 -Wall -Wextra -Werror -fsanitize=address -g -I./src_c -I./test -o testbench ./src_c/attention.cpp ./test/testbench.cpp
3. **Run the Testbench:**
./testbench