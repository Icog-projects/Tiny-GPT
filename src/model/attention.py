"""Causal Multi-Head Self-Attention module using torch.nn.MultiheadAttention."""

import torch
import torch.nn as nn
from src.model.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Causal Multi-Head Self-Attention layer using torch.nn.MultiheadAttention.

    Computes multi-head attention where each sequence position can only attend
    to preceding and current positions (j <= i), enforcing autoregressive causality.
    Uses PyTorch's native torch.nn.MultiheadAttention with batch_first=True
    and an upper-triangular boolean attention mask.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_head = config.n_head

        # Native PyTorch MultiheadAttention with batch_first=True
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_head,
            dropout=config.dropout,
            bias=config.bias,
            batch_first=True,
        )
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask buffer: upper-triangular boolean mask.
        # mask[i, j] == True indicates positions to mask out (future positions where j > i).
        # mask[i, j] == False indicates unmasked positions (past & current where j <= i).
        mask = torch.triu(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("mask", mask, persistent=True)

        # Legacy lower-triangular bias buffer retained for backward compatibility
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.context_length, config.context_length))
                 .view(1, 1, config.context_length, config.context_length),
            persistent=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of causal self-attention.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        B, T, C = x.size()

        # Sliced causal mask for current sequence length T
        causal_mask = self.mask[:T, :T]

        # MultiheadAttention forward pass: Query=x, Key=x, Value=x
        out, _ = self.attn(
            x,
            x,
            x,
            attn_mask=causal_mask,
            need_weights=False,
            is_causal=False,
        )
        out = self.resid_dropout(out)
        return out
