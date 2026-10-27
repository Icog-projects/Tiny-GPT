"""Transformer Decoder Block incorporating Pre-LayerNorm and residual connections."""

import torch
import torch.nn as nn
from src.model.attention import CausalSelfAttention
from src.model.config import GPTConfig
from src.model.feed_forward import MLP


class TransformerBlock(nn.Module):
    """Single decoder-only Transformer block with Pre-LayerNorm architecture.

    Structure:
        x = x + Attention(LayerNorm_1(x))
        x = x + MLP(LayerNorm_2(x))

    Pre-LN design places LayerNorm before the attention and feed-forward operations,
    ensuring a direct, unobstructed identity path through the residual stream.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=1e-5)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer block.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        # Pre-LN Attention with residual connection
        x = x + self.attn(self.ln_1(x))
        # Pre-LN MLP with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x
