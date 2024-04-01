import torch.nn as nn

from gpt.constants import EMBED_DIM, DROPOUT_PROB
from gpt.gelu import GaussianErrorLinearUnit
from gpt.multiheadattention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, max_seq_len):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(EMBED_DIM)
        self.attention_block = MultiHeadAttention(max_seq_len)
        self.layer_norm_2 = nn.LayerNorm(EMBED_DIM)
        self.feedforward = nn.ModuleDict(dict(
            output_linear_transform=nn.Linear(EMBED_DIM, 4 * EMBED_DIM),
            # todo - joel - rename to output_projection
            c_proj=nn.Linear(4 * EMBED_DIM, EMBED_DIM),
            activation=GaussianErrorLinearUnit(),
            dropout=nn.Dropout(DROPOUT_PROB),
        ))

    def forward(self, inputs):
        attn_outputs = self.attention_block(self.layer_norm_1(inputs))
        attn_outputs_with_skip = attn_outputs + inputs
        attn_outputs_normed = self.layer_norm_2(attn_outputs_with_skip)
        feedforward_outputs = self._feedforward(attn_outputs_normed)
        return feedforward_outputs + attn_outputs_with_skip

    def _feedforward(self, x):
        activations = self.feedforward.activation(self.feedforward.output_linear_transform(x))
        projections = self.feedforward.c_proj(activations)
        return self.feedforward.dropout(projections)
