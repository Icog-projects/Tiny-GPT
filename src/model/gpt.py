"""Decoder-only GPT Language Model implementation."""

import torch
import torch.nn as nn
from src.model.block import TransformerBlock
from src.model.config import GPTConfig


class GPT(nn.Module):
    """Decoder-only Transformer language model for next-token prediction.

    Architecture:
    1. Token Embedding: wte(idx) -> (B, T, d_model)
    2. Learned Positional Embedding: wpe(pos) -> (T, d_model)
    3. Combined Representation: x = Dropout(wte + wpe)
    4. N TransformerDecoderBlocks with Pre-LayerNorm and residual connections
    5. Final LayerNorm: ln_f(x)
    6. Language Model Head: lm_head(x) -> (B, T, vocab_size) raw logits
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # Embedding tables
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.context_length, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Transformer decoder blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])

        # Final LayerNorm and output projection
        self.ln_f = nn.LayerNorm(config.d_model, eps=1e-5)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize all model weights from scratch
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes weights using Gaussian normal distribution (mean=0.0, std=0.02).

        Biases are initialized to zeros, and LayerNorm scales are initialized to ones.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            if module.out_proj is not None and module.out_proj.weight is not None:
                nn.init.normal_(module.out_proj.weight, mean=0.0, std=0.02)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Performs forward pass for causal next-token prediction.

        Args:
            idx: Tensor of integer token IDs of shape (B, T).

        Returns:
            Raw unnormalized logits of shape (B, T, vocab_size).

        Raises:
            ValueError: If input tensor does not have 2 dimensions, has length 0,
                or sequence length T exceeds context_length.
        """
        if idx.dim() != 2:
            raise ValueError(f"Input tensor must have shape (B, T) with 2 dimensions, got {idx.dim()}D tensor.")

        B, T = idx.size()
        if T == 0:
            raise ValueError("Input sequence length T must be at least 1, got T=0.")
        if T > self.config.context_length:
            raise ValueError(
                f"Input sequence length T={T} exceeds maximum configured context length {self.config.context_length}."
            )

        # Position indices: [0, 1, ..., T-1] on the same device as inputs
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Embeddings: token vectors + positional vectors
        tok_emb = self.wte(idx)           # (B, T, d_model)
        pos_emb = self.wpe(pos)           # (T, d_model) -> broadcasts to (B, T, d_model)
        x = self.drop(tok_emb + pos_emb)

        # Sequentially apply Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization and projection to vocabulary logits
        x = self.ln_f(x)
        logits = self.lm_head(x)          # (B, T, vocab_size)
        return logits

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Computes the total number of trainable parameters in the model.

        Args:
            non_embedding: If True, subtracts the parameters of the token and positional
                embedding tables, returning only the Transformer body and head parameters.

        Returns:
            Number of trainable parameters.
        """
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n_params -= self.wte.weight.numel()
            n_params -= self.wpe.weight.numel()
        return n_params
