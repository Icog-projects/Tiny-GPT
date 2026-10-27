"""Offline unit tests for Task 2: Decoder-only GPT Transformer architecture.

All tests in this module run offline, deterministically, and without network access.
"""

import pytest
import torch
from src.model.attention import CausalSelfAttention
from src.model.config import GPTConfig
from src.model.feed_forward import MLP
from src.model.gpt import GPT


# ---------------------------------------------------------------------------
# 1. Configuration Validation Tests
# ---------------------------------------------------------------------------

def test_config_validation_valid_default():
    """Verifies that the default configuration is valid and initializes without error."""
    config = GPTConfig()
    assert config.vocab_size == 50257
    assert config.context_length == 64
    assert config.d_model == 64
    assert config.n_head == 4
    assert config.n_layer == 2
    assert config.d_ff == 256
    assert config.dropout == 0.1
    assert config.bias is True
    assert config.tie_weights is False


def test_config_validation_invalid_dimensions():
    """Verifies that non-positive dimensions raise clear ValueErrors."""
    with pytest.raises(ValueError, match="vocab_size must be a positive integer"):
        GPTConfig(vocab_size=0)

    with pytest.raises(ValueError, match="context_length must be a positive integer"):
        GPTConfig(context_length=-10)

    with pytest.raises(ValueError, match="d_model must be a positive integer"):
        GPTConfig(d_model=0)

    with pytest.raises(ValueError, match="n_head must be a positive integer"):
        GPTConfig(n_head=-2)

    with pytest.raises(ValueError, match="n_layer must be a positive integer"):
        GPTConfig(n_layer=0)

    with pytest.raises(ValueError, match="d_ff must be a positive integer"):
        GPTConfig(d_ff=-100)


def test_config_validation_head_divisibility():
    """Verifies that d_model must be evenly divisible by n_head."""
    with pytest.raises(ValueError, match="must be evenly divisible by n_head"):
        GPTConfig(d_model=65, n_head=4)  # 65 % 4 != 0


def test_config_validation_dropout_range():
    """Verifies that dropout must satisfy 0.0 <= dropout < 1.0."""
    with pytest.raises(ValueError, match="dropout must satisfy"):
        GPTConfig(dropout=-0.1)

    with pytest.raises(ValueError, match="dropout must satisfy"):
        GPTConfig(dropout=1.0)


def test_config_validation_tie_weights_not_implemented():
    """Verifies that tie_weights=True raises NotImplementedError in this baseline."""
    with pytest.raises(NotImplementedError, match="Weight tying"):
        GPTConfig(tie_weights=True)


# ---------------------------------------------------------------------------
# 2. Model Initialization & Sanity Checks
# ---------------------------------------------------------------------------

def test_model_construction_and_random_init():
    """Verifies that weights are initialized and LayerNorms have expected initial values."""
    config = GPTConfig(vocab_size=500, context_length=16, d_model=32, n_head=2, n_layer=1, d_ff=64)
    model = GPT(config)

    # Weights must not contain any NaNs or Infs
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), f"Parameter {name} contains non-finite values."

    # Linear weights should have mean roughly around 0 and positive variance
    wte_weight = model.wte.weight.data
    assert -0.05 < wte_weight.mean().item() < 0.05
    assert 0.01 < wte_weight.std().item() < 0.05

    # LayerNorm weights should be initialized to ones and biases to zeros
    assert torch.allclose(model.ln_f.weight, torch.ones_like(model.ln_f.weight))
    assert torch.allclose(model.ln_f.bias, torch.zeros_like(model.ln_f.bias))


# ---------------------------------------------------------------------------
# 3. Output Shapes & Tensor Properties
# ---------------------------------------------------------------------------

def test_model_output_shapes_variable_lengths():
    """Verifies output shape is (B, T, vocab_size) across multiple sequence lengths."""
    config = GPTConfig(vocab_size=1000, context_length=32, d_model=32, n_head=2, n_layer=1, d_ff=64)
    model = GPT(config)
    model.eval()

    test_lengths = [1, 7, 16, 32]
    batch_sizes = [1, 2]

    with torch.no_grad():
        for B in batch_sizes:
            for T in test_lengths:
                dummy_idx = torch.randint(0, config.vocab_size, (B, T), dtype=torch.long)
                logits = model(dummy_idx)
                assert logits.shape == (B, T, config.vocab_size), (
                    f"Expected shape ({B}, {T}, {config.vocab_size}), got {logits.shape}"
                )


def test_output_dtype_and_finite_logits():
    """Verifies that output logits are float32 and contain only finite numbers."""
    config = GPTConfig(vocab_size=500, context_length=16, d_model=32, n_head=2, n_layer=1, d_ff=64)
    model = GPT(config)
    model.eval()

    dummy_idx = torch.randint(0, config.vocab_size, (2, 8), dtype=torch.long)
    with torch.no_grad():
        logits = model(dummy_idx)

    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf values."


# ---------------------------------------------------------------------------
# 4. Attention & Feed-Forward Sublayer Unit Tests
# ---------------------------------------------------------------------------

def test_attention_internal_dimensions_and_bias_buffer():
    """Verifies internal shapes, buffer registration, and causal mask lower-triangular values."""
    config = GPTConfig(context_length=8, d_model=16, n_head=2)
    attn = CausalSelfAttention(config)

    # MultiheadAttention verification
    assert isinstance(attn.attn, torch.nn.MultiheadAttention)
    assert attn.attn.batch_first is True

    # Causal mask checks: upper-triangular boolean mask where True means masked (future)
    assert hasattr(attn, "mask")
    assert attn.mask.shape == (8, 8)
    assert attn.mask.dtype == torch.bool
    for i in range(8):
        for j in range(8):
            if j > i:
                assert attn.mask[i, j].item() is True
            else:
                assert attn.mask[i, j].item() is False

    # Bias buffer check (backward compatibility)
    assert hasattr(attn, "bias")
    assert attn.bias.shape == (1, 1, 8, 8)
    assert not any(p is attn.bias for p in attn.parameters())

    # Forward pass on (B, T, C)
    x = torch.randn(2, 6, 16)
    out = attn(x)
    assert out.shape == (2, 6, 16)


def test_mlp_feed_forward_dimensions():
    """Verifies MLP forward pass preserves (B, T, C) dimensions."""
    config = GPTConfig(d_model=16, d_ff=64, dropout=0.0)
    mlp = MLP(config)

    x = torch.randn(2, 5, 16)
    out = mlp(x)
    assert out.shape == (2, 5, 16)


# ---------------------------------------------------------------------------
# 5. Causal Masking Strict Invariance Test
# ---------------------------------------------------------------------------

def test_causal_masking_strict_invariance():
    """Essential causal invariant: modifying tokens strictly after position i
    must NOT change output logits at positions up to and including i in eval mode.
    """
    config = GPTConfig(vocab_size=1000, context_length=16, d_model=32, n_head=4, n_layer=2)
    model = GPT(config)
    model.eval()

    # Create two sequences that are IDENTICAL up to index 3 (first 4 tokens: indices 0, 1, 2, 3),
    # but DIFFER completely at indices 4 and 5.
    seq_a = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long)
    seq_b = torch.tensor([[10, 20, 30, 40, 990, 991]], dtype=torch.long)

    with torch.no_grad():
        logits_a = model(seq_a)  # (1, 6, vocab_size)
        logits_b = model(seq_b)  # (1, 6, vocab_size)

    # Invariant: Logits at positions 0, 1, 2, 3 must be strictly identical
    prefix_length = 4
    torch.testing.assert_close(
        logits_a[:, :prefix_length, :],
        logits_b[:, :prefix_length, :],
        rtol=1e-5,
        atol=1e-5,
        msg="Causal masking violated: past logits changed when future tokens were modified.",
    )


# ---------------------------------------------------------------------------
# 6. Boundary Conditions & Input Validation
# ---------------------------------------------------------------------------

def test_rejection_of_empty_and_over_context_inputs():
    """Verifies that invalid input sequence lengths raise clear ValueErrors."""
    config = GPTConfig(vocab_size=500, context_length=8, d_model=16, n_head=2, n_layer=1)
    model = GPT(config)
    model.eval()

    # Empty sequence (T = 0)
    empty_idx = torch.empty((1, 0), dtype=torch.long)
    with pytest.raises(ValueError, match="Input sequence length T must be at least 1"):
        model(empty_idx)

    # Over context length (T = 9 > 8)
    over_idx = torch.randint(0, 500, (1, 9), dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds maximum configured context length"):
        model(over_idx)

    # Non-2D input (1D tensor)
    dim_1d_idx = torch.tensor([1, 2, 3], dtype=torch.long)
    with pytest.raises(ValueError, match="Input tensor must have shape"):
        model(dim_1d_idx)


# ---------------------------------------------------------------------------
# 7. Gradient Propagation & Trainable Parameters
# ---------------------------------------------------------------------------

def test_gradient_backpropagation():
    """Verifies that backpropagation populates finite, non-zero gradients on model weights."""
    config = GPTConfig(vocab_size=500, context_length=16, d_model=32, n_head=2, n_layer=1, d_ff=64)
    model = GPT(config)
    model.train()

    idx = torch.randint(0, config.vocab_size, (2, 8), dtype=torch.long)
    logits = model(idx)

    # Compute a dummy scalar loss and backpropagate
    loss = logits.sum()
    loss.backward()

    # All trainable parameters must receive finite gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive a gradient."
        assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient."

    # Representative weight matrices must have non-zero gradient norm
    assert model.wte.weight.grad.norm().item() > 0.0
    assert model.wpe.weight.grad.norm().item() > 0.0
    assert model.blocks[0].attn.attn.in_proj_weight.grad.norm().item() > 0.0
    assert model.lm_head.weight.grad.norm().item() > 0.0


def test_parameter_count_computation():
    """Verifies dynamic parameter counting consistency with actual instantiated parameters."""
    config = GPTConfig()  # default baseline
    model = GPT(config)

    total_params = model.get_num_params(non_embedding=False)
    non_emb_params = model.get_num_params(non_embedding=True)

    expected_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected_non_emb = expected_total - model.wte.weight.numel() - model.wpe.weight.numel()

    assert total_params == expected_total
    assert non_emb_params == expected_non_emb
    # Confirm parameters are positive and non-embedding count is strictly smaller
    assert total_params > non_emb_params > 0
