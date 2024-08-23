
# FlashAttention Module

This repository contains an implementation of the `FlashAttention` module using PyTorch. FlashAttention is a custom attention mechanism designed to process large sequences in smaller chunks, reducing memory usage while maintaining computational efficiency. This can be particularly useful when working with long sequences that would otherwise require too much memory for standard attention mechanisms.

## Overview

The `FlashAttention` module implements a multi-head attention mechanism where the attention computation is performed in chunks. This approach helps in reducing the memory footprint, making it feasible to handle long sequences.

### Features

- **Multi-Head Attention:** The module supports multiple attention heads for better representation learning.
- **Chunk Processing:** The attention is computed in chunks, which allows handling longer sequences with lower memory requirements.
- **Scalable:** Designed to scale with sequence length by adjusting the `chunk_size` parameter.

## Installation

To use the `FlashAttention` module, ensure that you have PyTorch installed. You can install PyTorch via pip:

\`\`\`bash
pip install torch
\`\`\`

Clone this repository:

\`\`\`bash
git clone https://github.com/yourusername/flashattention.git
\`\`\`

## Usage

Here is a basic example of how to use the `FlashAttention` module:

\`\`\`python
import torch
from flash_attention import FlashAttention

# Define the model
embed_size = 128
heads = 8
chunk_size = 10  # Process 10 elements at a time to reduce memory usage
seq_length = 50

# Dummy input (batch size of 1, sequence length, embedding size)
dummy_input = torch.rand(1, seq_length, embed_size)

# Initialize the FlashAttention module
attention_layer = FlashAttention(embed_size, heads, chunk_size)

# Forward pass
output = attention_layer(dummy_input, dummy_input, dummy_input)
print(output.shape)  # Expected output: torch.Size([1, seq_length, embed_size])
\`\`\`

### Parameters

- **embed_size**: The size of the input embeddings.
- **heads**: The number of attention heads.
- **chunk_size**: The size of each chunk for processing.
- **mask**: (Optional) A mask tensor to prevent attention to certain positions.

### Methods

- **forward(values, keys, queries, mask=None)**: Computes the attention output for the given inputs.

## Example

Here's how to create an instance of the `FlashAttention` class and perform a forward pass:

\`\`\`python
# Example usage
embed_size = 128
heads = 8
chunk_size = 10
seq_length = 50
dummy_input = torch.rand(1, seq_length, embed_size)

attention_layer = FlashAttention(embed_size, heads, chunk_size)
output = attention_layer(dummy_input, dummy_input, dummy_input)
print(output.shape)  # Expected output: torch.Size([1, seq_length, embed_size])
\`\`\`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.
