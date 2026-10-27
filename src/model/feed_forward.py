"""Position-wise Feed-Forward Network (MLP) for the Transformer block."""

import torch
import torch.nn as nn
from src.model.config import GPTConfig


class MLP(nn.Module):
    """Position-wise feed-forward sublayer.

    Applies two linear transformations with a non-linear GELU activation in between:
    FFN(x) = Dropout(Linear_2(GELU(Linear_1(x))))

    Shapes:
        Input:  (B, T, C) where B is batch size, T is sequence length, C is d_model
        Hidden: (B, T, d_ff)
        Output: (B, T, C)
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP sublayer.

        Args:
            x: Input hidden states of shape (B, T, d_model).

        Returns:
            Projected hidden states of shape (B, T, d_model).
        """
        # (B, T, C) -> (B, T, d_ff)
        h = self.c_fc(x)
        h = self.act(h)
        # (B, T, d_ff) -> (B, T, C)
        out = self.c_proj(h)
        out = self.dropout(out)
        return out
