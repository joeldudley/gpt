import torch.nn as nn
from torch.nn import init

from gpt.constants import EMBED_DIM
from gpt.transformer.multiheadattention import MultiHeadAttention
from gpt.transformer.transformerblockfeedforward import TransformerBlockFeedForward


class TransformerBlock(nn.Module):
    def __init__(self, max_seq_len):
        super().__init__()
        self.norm_inputs = nn.LayerNorm(EMBED_DIM)
        self.multi_head_attention = MultiHeadAttention(max_seq_len)
        self.norm_multi_head_attention = nn.LayerNorm(EMBED_DIM)
        self.feedforward = TransformerBlockFeedForward()

        init.zeros_(self.norm_inputs.bias)
        init.ones_(self.norm_inputs.weight)
        init.zeros_(self.norm_multi_head_attention.bias)
        init.ones_(self.norm_multi_head_attention.weight)

    def forward(self, inputs):
        inputs_normed = self.norm_inputs.forward(inputs)
        attn_outputs = self.multi_head_attention.forward(inputs_normed)
        attn_outputs_with_skip = attn_outputs + inputs
        attn_outputs_normed = self.norm_multi_head_attention.forward(attn_outputs_with_skip)
        feedforward_outputs = self.feedforward.forward(attn_outputs_normed)
        feedforward_outputs_with_skip = feedforward_outputs + attn_outputs_with_skip
        return feedforward_outputs_with_skip
