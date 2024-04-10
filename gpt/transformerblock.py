import torch.nn as nn

from gpt.constants import EMBED_DIM
from gpt.multiheadattention import MultiHeadAttention
from gpt.transformerblockfeedforward import TransformerBlockFeedForward


class TransformerBlock(nn.Module):
    def __init__(self, max_seq_len):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(EMBED_DIM)
        self.attention_block = MultiHeadAttention(max_seq_len)
        self.layer_norm_2 = nn.LayerNorm(EMBED_DIM)
        self.feedforward = TransformerBlockFeedForward()

    def forward(self, inputs):
        attn_outputs = self.attention_block(self.layer_norm_1(inputs))
        attn_outputs_with_skip = attn_outputs + inputs
        attn_outputs_normed = self.layer_norm_2(attn_outputs_with_skip)
        feedforward_outputs = self.feedforward(attn_outputs_normed)
        return feedforward_outputs + attn_outputs_with_skip
