import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init, Module
from torch.nn.functional import softmax

from simplegpt.config.config import EMBED_DIM, NUM_ATTN_HEADS, DROPOUT_PROB, STD


class MultiHeadAttention(Module):
    def __init__(self, max_seq_len: int):
        super().__init__()
        self.keys_queries_values = nn.Linear(EMBED_DIM, NUM_ATTN_HEADS * EMBED_DIM)
        init.normal_(self.keys_queries_values.weight, std=STD)
        init.zeros_(self.keys_queries_values.bias)

        self.output_projection = nn.Linear(EMBED_DIM, EMBED_DIM)
        init.normal_(self.output_projection.weight, std=STD)
        init.zeros_(self.output_projection.bias)

        self.attn_dropout = nn.Dropout(DROPOUT_PROB)
        self.resid_dropout = nn.Dropout(DROPOUT_PROB)
        self.mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, seq_len, _ = inputs.size()

        query, key, value = self.keys_queries_values.forward(inputs).split(EMBED_DIM, dim=2)
        query_per_head = self._split_by_head(query, batch_size, seq_len)
        key_per_head = self._split_by_head(key, batch_size, seq_len)
        value_per_head = self._split_by_head(value, batch_size, seq_len)

        outputs = self._get_attention(key_per_head, query_per_head, seq_len, value_per_head)
        outputs_per_head = outputs.transpose(1, 2).contiguous().view(batch_size, seq_len, EMBED_DIM)
        output_projection = self.output_projection.forward(outputs_per_head)
        return self.resid_dropout.forward(output_projection)

    def _get_attention(self, key_per_head: Tensor, query_per_head: Tensor, seq_len: int,
                       value_per_head: Tensor) -> Tensor:
        compatibility = query_per_head @ key_per_head.transpose(-2, -1)
        compatibility_scaled = compatibility * (1.0 / math.sqrt(key_per_head.size(-1)))
        compatibility_masked = compatibility_scaled.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        compatibility_softmaxed = softmax(compatibility_masked, dim=-1)
        compatibility_with_dropout = self.attn_dropout.forward(compatibility_softmaxed)
        return compatibility_with_dropout @ value_per_head

    @staticmethod
    def _split_by_head(vector: Tensor, batch_size: int, seq_len: int) -> Tensor:
        return vector.view(batch_size, seq_len, NUM_ATTN_HEADS, EMBED_DIM // NUM_ATTN_HEADS).transpose(1, 2)
