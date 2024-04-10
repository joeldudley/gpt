import torch
import torch.nn as nn
from torch.nn import functional

from gpt.constants import EMBED_DIM
from gpt.transformer.transformer import Transformer


class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.transformer = Transformer(vocab_size, max_seq_len)
        self.language_modeling_head = nn.Linear(EMBED_DIM, vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        self.transformer.init_residual_projection_weights()

    def forward(self, inputs, targets=None):
        transformer_outputs = self.transformer(inputs)
        logits = self.language_modeling_head(transformer_outputs)
        return logits, self._get_loss(logits, targets)

    @torch.no_grad()
    def generate(self, inputs, max_new_tokens, temperature=1.0):
        tokens = inputs
        for _ in range(max_new_tokens):
            tokens = self.generate_token(tokens, temperature)
        return tokens

    def generate_token(self, prev_tokens, temperature):
        cropped_tokens = prev_tokens if prev_tokens.size(1) <= self.max_seq_len else prev_tokens[:, -self.max_seq_len:]
        logits, _ = self(cropped_tokens)
        scaled_logits = logits[:, -1, :] / temperature
        probabilities = functional.softmax(scaled_logits, dim=-1)
        _, next_token = torch.topk(probabilities, k=1, dim=-1)
        return torch.cat((prev_tokens, next_token), dim=1)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @staticmethod
    def _get_loss(logits, targets):
        if targets is not None:
            return functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return None
