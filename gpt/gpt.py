import math

import torch
import torch.nn as nn
from torch.nn import functional

from gpt.constants import EMBED_DIM, NUM_BLOCKS, DROPOUT_PROB
from gpt.transformerblock import TransformerBlock


class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.transformer = nn.ModuleDict(dict(
            token_embedding_weights=nn.Embedding(vocab_size, EMBED_DIM),
            position_embedding_weights=nn.Embedding(max_seq_len, EMBED_DIM),
            dropout=nn.Dropout(DROPOUT_PROB),
            hidden_state=nn.ModuleList([TransformerBlock(max_seq_len) for _ in range(NUM_BLOCKS)]),
            layer_norm_feedforward=nn.LayerNorm(EMBED_DIM),
        ))
        self.language_modeling_head = nn.Linear(EMBED_DIM, vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for param_name, param in self.named_parameters():
            if param_name.endswith('c_proj.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * NUM_BLOCKS))

    def forward(self, inputs, targets=None):
        _, seq_len = inputs.size()
        position = torch.arange(0, seq_len, dtype=torch.long, device=inputs.device).unsqueeze(0)

        # forward the GPT model itself
        token_embedding = self.transformer.token_embedding_weights(inputs)
        position_embeddings = self.transformer.position_embedding_weights(position)
        x = self.transformer.dropout(token_embedding + position_embeddings)
        for transformer_block in self.transformer.hidden_state:
            x = transformer_block(x)
        x = self.transformer.layer_norm_feedforward(x)
        logits = self.language_modeling_head(x)

        return logits, self._get_loss(logits, targets)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at max_seq_len
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = functional.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

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
