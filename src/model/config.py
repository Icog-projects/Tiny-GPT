"""Configuration dataclass for the decoder-only GPT model."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Hyperparameter configuration for the GPT language model.

    Attributes:
        vocab_size: Size of the vocabulary (default: 50,257 for GPT-2).
        context_length: Maximum sequence length the model can process (T).
        d_model: Hidden embedding dimension (C).
        n_head: Number of attention heads (H).
        n_layer: Number of Transformer decoder blocks (L).
        d_ff: Expansion dimension of the feed-forward network (typically 4 * d_model).
        dropout: Dropout probability applied to embeddings, attention, and residual projections.
        bias: Whether to include learnable additive biases in Linear and LayerNorm layers.
        tie_weights: Flag for tying input embeddings to output LM head weights (disabled in baseline).
    """

    vocab_size: int = 50257
    context_length: int = 64
    d_model: int = 64
    n_head: int = 4
    n_layer: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = False

    def __post_init__(self) -> None:
        """Validates configuration parameters upon instantiation."""
        self.validate()

    def validate(self) -> None:
        """Enforces structural and numerical validity of configuration attributes.

        Raises:
            ValueError: If any configuration attribute violates architectural or numerical constraints.
        """
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be a positive integer, got {self.vocab_size}.")
        if self.context_length <= 0:
            raise ValueError(f"context_length must be a positive integer, got {self.context_length}.")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {self.d_model}.")
        if self.n_head <= 0:
            raise ValueError(f"n_head must be a positive integer, got {self.n_head}.")
        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be a positive integer, got {self.n_layer}.")
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be a positive integer, got {self.d_ff}.")

        # Head divisibility constraint
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be evenly divisible by n_head ({self.n_head}). "
                f"Current d_model % n_head = {self.d_model % self.n_head}."
            )

        # Dropout range constraint
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must satisfy 0.0 <= dropout < 1.0, got {self.dropout}.")

        # Weight tying constraint for baseline
        if self.tie_weights:
            raise NotImplementedError(
                "Weight tying (tie_weights=True) is not enabled in this baseline implementation. "
                "Keep tie_weights=False."
            )
