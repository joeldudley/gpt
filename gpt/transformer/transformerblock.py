import torch.nn as nn

from gpt.constants import EMBED_DIM
from gpt.transformer.multiheadattention import MultiHeadAttention
from gpt.transformer.transformerblockfeedforward import TransformerBlockFeedForward


class TransformerBlock(nn.Module):
    def __init__(self, max_seq_len):
        super().__init__()
        self.norm_1 = nn.LayerNorm(EMBED_DIM)
        self.multi_head_attention = MultiHeadAttention(max_seq_len)
        self.norm_2 = nn.LayerNorm(EMBED_DIM)
        self.feedforward = TransformerBlockFeedForward()

    def forward(self, inputs):
        inputs_normed = self.norm_1(inputs)
        attn_outputs = self.multi_head_attention(inputs_normed)
        attn_outputs_with_skip = attn_outputs + inputs
        attn_outputs_normed = self.norm_2(attn_outputs_with_skip)
        feedforward_outputs = self.feedforward(attn_outputs_normed)
        return feedforward_outputs + attn_outputs_with_skip
