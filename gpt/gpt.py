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
        self.transformer.init_weights()

    def forward(self, inputs, targets=None):
        transformer_outputs = self.transformer(inputs)
        logits = self.language_modeling_head(transformer_outputs)
        return logits, self._get_loss(logits, targets)

    @torch.no_grad()
    def generate(self, inputs, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at max_seq_len
            cropped_inputs = inputs if inputs.size(1) <= self.max_seq_len else inputs[:, -self.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(cropped_inputs)
            # pluck the logits at the final step and scale by desired temperature
            scaled_logits = logits[:, -1, :] / temperature
            # apply softmax to convert logits to (normalized) probabilities
            probs = functional.softmax(scaled_logits, dim=-1)
            # take the most likely element
            _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            inputs = torch.cat((inputs, idx_next), dim=1)

        return inputs

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
