import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init, Module
from torch.nn.functional import cross_entropy, softmax

from simplegpt.config.config import EMBED_DIM, STD
from simplegpt.transformer import Transformer


class GPT(Module):
    def __init__(self, vocab_size: int, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.transformer = Transformer(vocab_size, max_seq_len)

        self.language_modeling_head = nn.Linear(EMBED_DIM, vocab_size, bias=False)
        init.normal_(self.language_modeling_head.weight, std=STD)

    def forward(self, inputs: Tensor, targets: Tensor = None) -> (Tensor, Tensor):
        transformer_outputs = self.transformer.forward(inputs)
        logits = self.language_modeling_head.forward(transformer_outputs)
        loss = None if targets is None else self._get_loss(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens: Tensor, num_tokens_to_generate: int) -> Tensor:
        for _ in range(num_tokens_to_generate):
            tokens = torch.cat((tokens, self._generate_token(tokens)), dim=1)
        return tokens

    @staticmethod
    def _get_loss(logits: Tensor, targets: Tensor) -> Tensor:
        return cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

    def _generate_token(self, prev_tokens: Tensor) -> Tensor:
        cropped_tokens = prev_tokens if prev_tokens.size(1) <= self.max_seq_len else prev_tokens[:, -self.max_seq_len:]
        logits, _ = self.forward(cropped_tokens)  # Skip the scaling of logits by temp. from the original GPT-2 paper.
        probabilities = softmax(logits[:, -1, :], dim=-1)
        return torch.topk(probabilities, 1)[1]
