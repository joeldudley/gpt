import torch
import torch.nn as nn

from gpt.constants import EMBED_DIM, DROPOUT_PROB, NUM_BLOCKS
from gpt.transformer.transformerblock import TransformerBlock


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
        position = torch.arange(0, seq_len).unsqueeze(0)
        position_embeddings = self.position_embedding_weights.forward(position)

        token_embeddings = self.token_embedding_weights.forward(inputs)
        block_outputs = self.dropout.forward(token_embeddings + position_embeddings)
        for transformer_block in self.transformer_blocks:
            block_outputs = transformer_block.forward(block_outputs)
        return self.layer_norm_feedforward.forward(block_outputs)
