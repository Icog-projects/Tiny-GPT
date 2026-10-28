"""Offline unit tests for Task 3: Autoregressive text generation and standalone inference."""

import pytest
import torch

from src.data.tokenizer import GPT2Tokenizer
from src.generation.generate import generate_text
from src.model.config import GPTConfig
from src.model.gpt import GPT
from src.training.trainer import load_checkpoint, save_checkpoint


@pytest.fixture
def tokenizer() -> GPT2Tokenizer:
    return GPT2Tokenizer()


@pytest.fixture
def mini_gpt() -> GPT:
    config = GPTConfig(
        vocab_size=50257,
        context_length=16,
        d_model=32,
        n_head=2,
        n_layer=1,
        d_ff=64,
        dropout=0.0,
    )
    return GPT(config)


def test_greedy_generation_basic(mini_gpt: GPT, tokenizer: GPT2Tokenizer):
    """Verifies that greedy text generation appends requested new tokens and starts with prompt."""
    prompt = "Hello world"
    max_new_tokens = 6

    output, token_ids = generate_text(
        mini_gpt,
        tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        return_token_ids=True,
    )

    assert isinstance(output, str)
    assert output.startswith(prompt)
    # Check that exactly max_new_tokens were appended to the input sequence
    assert len(token_ids) == len(tokenizer.encode(prompt)) + max_new_tokens


def test_generation_respects_context_length_limit(mini_gpt: GPT, tokenizer: GPT2Tokenizer):
    """Verifies that prompt longer than context_length is automatically cropped to most recent context."""
    # Context length of mini_gpt is 16
    # Construct a prompt longer than 16 tokens
    long_prompt = "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen"
    prompt_tokens = tokenizer.encode(long_prompt)
    assert len(prompt_tokens) > mini_gpt.config.context_length

    # Must generate without ValueError even though len(prompt) > context_length
    output = generate_text(mini_gpt, tokenizer, prompt=long_prompt, max_new_tokens=4)
    assert isinstance(output, str)
    assert output.startswith(long_prompt)


def test_generation_empty_and_whitespace_prompt(mini_gpt: GPT, tokenizer: GPT2Tokenizer):
    """Verifies that empty or whitespace prompts trigger unconditional generation from EOT token."""
    output_empty = generate_text(mini_gpt, tokenizer, prompt="", max_new_tokens=5)
    assert isinstance(output_empty, str)
    assert len(output_empty) > 0

    output_whitespace = generate_text(mini_gpt, tokenizer, prompt="   \n\t  ", max_new_tokens=5)
    assert isinstance(output_whitespace, str)
    assert len(output_whitespace) > 0


def test_generation_invalid_max_tokens_raises(mini_gpt: GPT, tokenizer: GPT2Tokenizer):
    """Verifies that non-positive max_new_tokens values raise ValueError."""
    with pytest.raises(ValueError, match="max_new_tokens must be a positive integer"):
        generate_text(mini_gpt, tokenizer, prompt="Test", max_new_tokens=0)

    with pytest.raises(ValueError, match="max_new_tokens must be a positive integer"):
        generate_text(mini_gpt, tokenizer, prompt="Test", max_new_tokens=-10)


def test_generation_does_not_modify_weights_and_restores_mode(mini_gpt: GPT, tokenizer: GPT2Tokenizer):
    """Verifies that generation executes in eval mode without mutating weights or gradients."""
    mini_gpt.train()
    initial_wte = mini_gpt.wte.weight.clone().detach()

    generate_text(mini_gpt, tokenizer, prompt="Sample prompt", max_new_tokens=4)

    # Weights must be untouched
    assert torch.equal(mini_gpt.wte.weight, initial_wte)
    # Mode must be restored
    assert mini_gpt.training is True


def test_standalone_inference_from_saved_checkpoint(mini_gpt: GPT, tokenizer: GPT2Tokenizer, tmp_path):
    """Verifies that a checkpoint can be saved, loaded into a fresh model, and generate text."""
    checkpoint_file = str(tmp_path / "inference_test.pt")
    save_checkpoint(checkpoint_file, mini_gpt, mini_gpt.config)

    # Load into fresh model instance
    loaded_model, _, _ = load_checkpoint(checkpoint_file, device="cpu")

    prompt = "Artificial intelligence is"
    out1 = generate_text(mini_gpt, tokenizer, prompt=prompt, max_new_tokens=5)
    out2 = generate_text(loaded_model, tokenizer, prompt=prompt, max_new_tokens=5)

    # Both models must produce bit-for-bit identical text under greedy decoding
    assert out1 == out2


def test_backend_service_default_checkpoint_path():
    """Verifies that the backend service defaults to checkpoints/tiny_gpt_wikitext2_60k_best.pt."""
    import backend.service as svc
    assert svc.DEFAULT_CHECKPOINT_PATH == "checkpoints/tiny_gpt_wikitext2_60k_best.pt"


def test_resolve_checkpoint_path_relative():
    """Verifies that resolve_checkpoint_path resolves relative paths against the project root."""
    import os
    import backend.service as svc
    resolved = svc.resolve_checkpoint_path("checkpoints/tiny_gpt_wikitext2_60k_best.pt")
    assert os.path.isabs(resolved)
    assert os.path.isfile(resolved)


def test_backend_service_missing_checkpoint_handling(monkeypatch):
    """Verifies that backend service handles missing checkpoints gracefully without crashing."""
    import backend.service as svc

    # Ensure clean service state
    svc.unload_model()
    monkeypatch.setenv("TINY_GPT_CHECKPOINT", "checkpoints/non_existent_dummy_checkpoint.pt")

    svc.load_model()
    assert not svc.is_model_loaded()
    assert svc._load_error is not None
    assert "TINY_GPT_CHECKPOINT" in svc._load_error

    with pytest.raises(RuntimeError, match="No valid checkpoint is configured"):
        svc.run_generation(prompt="test", max_new_tokens=5)

    svc.unload_model()


def test_backend_service_env_variable_override(monkeypatch, tmp_path):
    """Verifies that TINY_GPT_CHECKPOINT environment variable overrides the default path."""
    import backend.service as svc

    custom_path = str(tmp_path / "custom_checkpoint.pt")
    monkeypatch.setenv("TINY_GPT_CHECKPOINT", custom_path)

    svc.unload_model()
    svc.load_model()

    assert svc._checkpoint_path == custom_path
    assert not svc.is_model_loaded()

    svc.unload_model()


def test_backend_service_loads_real_default_checkpoint(monkeypatch):
    """Verifies that the backend service successfully loads the real trained checkpoint."""
    import backend.service as svc

    svc.unload_model()
    monkeypatch.delenv("TINY_GPT_CHECKPOINT", raising=False)

    svc.load_model()
    assert svc.is_model_loaded()
    assert svc._load_error is None
    assert svc._model is not None
    assert svc._tokenizer is not None

    output = svc.run_generation(prompt="The history of science", max_new_tokens=5)
    assert isinstance(output, str)
    assert len(output) > 0

    svc.unload_model()
