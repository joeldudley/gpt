import math

import torch
import torch.nn as nn

from transformer.constants import EMBED_DIM, NUM_BLOCKS, DROPOUT_PROB
from transformer.transformerblock import TransformerBlock


class Transformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len):
        super().__init__()
        self.token_embedding_weights = nn.Embedding(vocab_size, EMBED_DIM)
        self.position_embedding_weights = nn.Embedding(max_seq_len, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT_PROB)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(max_seq_len) for _ in range(NUM_BLOCKS)])
        self.layer_norm_feedforward = nn.LayerNorm(EMBED_DIM)

    def forward(self, inputs):
        _, seq_len = inputs.size()
        position = torch.arange(0, seq_len, dtype=torch.long, device=inputs.device).unsqueeze(0)

        token_embedding = self.token_embedding_weights(inputs)
        position_embeddings = self.position_embedding_weights(position)
        x = self.dropout(token_embedding + position_embeddings)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        return self.layer_norm_feedforward(x)

    def init_weights(self):
        for block in self.transformer_blocks:
            for param_name, param in block.feedforward.c_proj.named_parameters():
                if param_name == 'weight':
                    torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * NUM_BLOCKS))
