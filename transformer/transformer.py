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
        self.hidden_state = nn.ModuleList([TransformerBlock(max_seq_len) for _ in range(NUM_BLOCKS)])
        self.layer_norm_feedforward = nn.LayerNorm(EMBED_DIM)

    def forward(self, inputs):
        _, seq_len = inputs.size()
        position = torch.arange(0, seq_len, dtype=torch.long, device=inputs.device).unsqueeze(0)

        token_embedding = self.token_embedding_weights(inputs)
        position_embeddings = self.position_embedding_weights(position)
        x = self.dropout(token_embedding + position_embeddings)
        for transformer_block in self.hidden_state:
            x = transformer_block(x)
        return self.layer_norm_feedforward(x)
