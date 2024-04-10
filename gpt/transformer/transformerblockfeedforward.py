import torch.nn as nn

from gpt.constants import EMBED_DIM, DROPOUT_PROB
from gpt.transformer.gelu import GaussianErrorLinearUnit


class TransformerBlockFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_linear_transform = nn.Linear(EMBED_DIM, 4 * EMBED_DIM)
        self.output_projection = nn.Linear(4 * EMBED_DIM, EMBED_DIM)
        self.activation = GaussianErrorLinearUnit()
        self.dropout = nn.Dropout(DROPOUT_PROB)

    def forward(self, inputs):
        linear_transform = self.output_linear_transform(inputs)
        activations = self.activation(linear_transform)
        projections = self.output_projection(activations)
        return self.dropout(projections)
