import math

import torch
import torch.nn as nn
from torch.nn.functional import softmax

from gpt.constants import EMBED_DIM, NUM_ATTN_HEADS, DROPOUT_PROB


class MultiHeadAttention(nn.Module):
    def __init__(self, max_seq_len):
        super().__init__()
        self.keys_queries_values = nn.Linear(EMBED_DIM, NUM_ATTN_HEADS * EMBED_DIM)
        self.output_projection = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.attn_dropout = nn.Dropout(DROPOUT_PROB)
        self.resid_dropout = nn.Dropout(DROPOUT_PROB)
        self.mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)

    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()

        query, key, value = self.keys_queries_values(inputs).split(EMBED_DIM, dim=2)
        query_per_head = self._split_by_head(query, batch_size, seq_len)
        key_per_head = self._split_by_head(key, batch_size, seq_len)
        value_per_head = self._split_by_head(value, batch_size, seq_len)

        attn = (query_per_head @ key_per_head.transpose(-2, -1)) * (1.0 / math.sqrt(key_per_head.size(-1)))
        attn_masked = attn.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        attn_softmaxed = softmax(attn_masked, dim=-1)
        attn_with_dropout = self.attn_dropout(attn_softmaxed)

        outputs = attn_with_dropout @ value_per_head
        outputs_per_head = outputs.transpose(1, 2).contiguous().view(batch_size, seq_len, EMBED_DIM)
        output_projection = self.output_projection(outputs_per_head)
        return self.resid_dropout(output_projection)

    @staticmethod
    def _split_by_head(vector, batch_size, seq_len):
        return vector.view(batch_size, seq_len, NUM_ATTN_HEADS, EMBED_DIM // NUM_ATTN_HEADS).transpose(1, 2)
