import torch.nn as nn
from torch import Tensor
from torch.nn import init

from gpt.constants import EMBED_DIM, DROPOUT_PROB, STD
from gpt.transformer.gelu import GaussianErrorLinearUnit


class TransformerBlockFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_linear_transform = nn.Linear(EMBED_DIM, 4 * EMBED_DIM)
        init.normal_(self.output_linear_transform.weight, std=STD)
        init.zeros_(self.output_linear_transform.bias)

        self.residual_projections = nn.Linear(4 * EMBED_DIM, EMBED_DIM)
        # We skip the special scaling of residual layer weights in the original GPT-2 paper.
        init.normal_(self.residual_projections.weight, std=STD)
        init.zeros_(self.residual_projections.bias)

        self.gelu = GaussianErrorLinearUnit()
        self.dropout = nn.Dropout(DROPOUT_PROB)

    def forward(self, inputs: Tensor) -> Tensor:
        linear_transform = self.output_linear_transform.forward(inputs)
        activations = self.gelu.forward(linear_transform)
        projections = self.residual_projections.forward(activations)
        return self.dropout.forward(projections)
