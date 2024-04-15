import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.functional import cross_entropy, softmax

from gpt.constants import EMBED_DIM, STD
from gpt.transformer.transformer import Transformer


class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.transformer = Transformer(vocab_size, max_seq_len)

        self.language_modeling_head = nn.Linear(EMBED_DIM, vocab_size, bias=False)
        init.normal_(self.language_modeling_head.weight, std=STD)

    def forward(self, inputs, targets=None):
        transformer_outputs = self.transformer.forward(inputs)
        logits = self.language_modeling_head.forward(transformer_outputs)
        loss = None if targets is None else self._get_loss(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, num_tokens_to_generate):
        for _ in range(num_tokens_to_generate):
            tokens = torch.cat((tokens, self._generate_token(tokens)), dim=1)
        return tokens

    def _generate_token(self, prev_tokens):
        cropped_tokens = prev_tokens if prev_tokens.size(1) <= self.max_seq_len else prev_tokens[:, -self.max_seq_len:]
        logits, _ = self.forward(cropped_tokens)  # Skip the scaling of logits by temp. from the original GPT-2 paper.
        probabilities = softmax(logits[:, -1, :], dim=-1)
        return torch.topk(probabilities, k=1)[1]

    @staticmethod
    def _get_loss(logits, targets):
        return cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
