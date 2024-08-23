import torch
import torch.nn as nn
import torch.nn.functional as F

class FlashAttention(nn.Module):
    def __init__(self, embed_size, heads, chunk_size):
        super(FlashAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.chunk_size = chunk_size

        # Ensure the embedding size is divisible by the number of heads
        assert embed_size % heads == 0, "Embed size must be divisible by number of heads"
        self.depth = embed_size // heads

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, queries, mask=None):
        N, seq_length, _ = queries.shape

        # Split the embeddings into multiple heads
        values = self.values(values).view(N, seq_length, self.heads, self.depth)
        keys = self.keys(keys).view(N, seq_length, self.heads, self.depth)
        queries = self.queries(queries).view(N, seq_length, self.heads, self.depth)

        # Initialize the output tensor
        outputs = torch.zeros_like(queries)

        # Process each chunk separately
        for chunk_start in range(0, seq_length, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, seq_length)
            local_keys = keys[:, chunk_start:chunk_end, :, :]
            local_values = values[:, chunk_start:chunk_end, :, :]
            local_queries = queries[:, chunk_start:chunk_end, :, :]

            # Calculate the dot product attention scores
            scores = torch.einsum('nbhd,mbhd->nhbm', local_queries, local_keys)

            if mask is not None:
                scores += mask[:, chunk_start:chunk_end].unsqueeze(1).unsqueeze(1)

            # Softmax to get the attention weights
            attention = F.softmax(scores, dim=-1)

            # Multiply by values to get the final output for the chunk
            out = torch.einsum('nhbm,mbhd->nbhd', attention, local_values)
            outputs[:, chunk_start:chunk_end, :, :] = out

        # Reshape and project the output
        outputs = outputs.reshape(N, seq_length, -1)
        return self.fc_out(outputs)

# Example usage
embed_size = 128
heads = 8
chunk_size = 10  # Processing 10 elements at a time to reduce memory usage
seq_length = 50
dummy_input = torch.rand(1, seq_length, embed_size)  # Batch size of 1

attention_layer = FlashAttention(embed_size, heads, chunk_size)
output = attention_layer(dummy_input, dummy_input, dummy_input)
print(output.shape)  # Should return torch.Size([1, seq_length, embed_size])
