"""Offline unit tests for Task 3: Training loop, loss computation, evaluation, and checkpointing."""

import json
import os
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import GPTDataset
from src.model.config import GPTConfig
from src.model.gpt import GPT
from src.training.trainer import (
    compute_loss,
    create_synthetic_verification_dataset,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train,
    train_one_step,
)


@pytest.fixture
def mini_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=100,
        context_length=8,
        d_model=16,
        n_head=2,
        n_layer=1,
        d_ff=32,
        dropout=0.0,
    )


@pytest.fixture
def mini_model(mini_config: GPTConfig) -> GPT:
    return GPT(mini_config)


def test_compute_loss_shapes_and_values():
    """Verifies that compute_loss accepts (B, T, V) logits and (B, T) targets and returns scalar loss."""
    B, T, V = 2, 4, 50
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.randint(0, V, (B, T), dtype=torch.long)

    loss = compute_loss(logits, targets)

    assert loss.dim() == 0, f"Expected scalar 0D loss, got {loss.dim()}D."
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss).item()
    assert loss.item() > 0.0


def test_compute_loss_shape_validation():
    """Verifies that compute_loss raises ValueError on mismatched or unexpected input dimensions."""
    V = 50
    # Mismatched sequence length
    logits = torch.randn(2, 4, V)
    mismatched_targets = torch.randint(0, V, (2, 5), dtype=torch.long)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_loss(logits, mismatched_targets)

    # 2D logits instead of 3D
    logits_2d = torch.randn(8, V)
    targets_2d = torch.randint(0, V, (2, 4), dtype=torch.long)
    with pytest.raises(ValueError, match="Expected 3D logits"):
        compute_loss(logits_2d, targets_2d)

    # 1D targets instead of 2D
    targets_1d = torch.randint(0, V, (8,), dtype=torch.long)
    with pytest.raises(ValueError, match="Expected 2D targets"):
        compute_loss(logits, targets_1d)


def test_train_one_step_updates_weights_and_gradients(mini_model: GPT, mini_config: GPTConfig):
    """Verifies that a single training step backpropagates gradients and updates trainable parameters."""
    B, T = 2, mini_config.context_length
    x = torch.randint(0, mini_config.vocab_size, (B, T), dtype=torch.long)
    y = torch.randint(0, mini_config.vocab_size, (B, T), dtype=torch.long)

    optimizer = torch.optim.Adam(mini_model.parameters(), lr=1e-2)

    # Record initial weights of a key parameter
    initial_wte = mini_model.wte.weight.clone().detach()

    loss_val = train_one_step(mini_model, x, y, optimizer, max_norm=1.0)

    assert isinstance(loss_val, float)
    assert loss_val > 0.0

    # Ensure gradients were computed and are finite
    assert mini_model.wte.weight.grad is not None
    assert torch.isfinite(mini_model.wte.weight.grad).all()

    # Ensure parameter weights were updated by optimizer.step()
    assert not torch.equal(mini_model.wte.weight, initial_wte), "Model parameters were not updated by optimizer step."


def test_evaluate_does_not_update_weights_or_track_gradients(mini_model: GPT, mini_config: GPTConfig):
    """Verifies that evaluate() runs without updating weights or computing gradients."""
    B, T = 2, mini_config.context_length
    val_samples = [
        (
            torch.randint(0, mini_config.vocab_size, (T,), dtype=torch.long),
            torch.randint(0, mini_config.vocab_size, (T,), dtype=torch.long),
        )
        for _ in range(6)
    ]
    val_loader = DataLoader(GPTDataset(val_samples), batch_size=B)

    # Record initial state of all parameters
    initial_params = {name: p.clone().detach() for name, p in mini_model.named_parameters()}

    val_loss = evaluate(mini_model, val_loader, max_batches=2, device="cpu")

    assert val_loss is not None
    assert isinstance(val_loss, float)
    assert val_loss > 0.0

    # Confirm weights are bit-for-bit unchanged
    for name, p in mini_model.named_parameters():
        assert torch.equal(p, initial_params[name]), f"Parameter {name} was modified during evaluation!"
        assert p.grad is None, f"Parameter {name} accumulated gradient during evaluation!"


def test_evaluate_mode_restoration(mini_model: GPT, mini_config: GPTConfig):
    """Verifies that evaluate() restores the model's training mode accurately."""
    T = mini_config.context_length
    val_samples = [
        (
            torch.randint(0, mini_config.vocab_size, (T,), dtype=torch.long),
            torch.randint(0, mini_config.vocab_size, (T,), dtype=torch.long),
        )
        for _ in range(4)
    ]
    val_loader = DataLoader(GPTDataset(val_samples), batch_size=2)

    # Case A: Model starts in train mode
    mini_model.train()
    assert mini_model.training is True
    evaluate(mini_model, val_loader)
    assert mini_model.training is True, "Model training mode was not restored after evaluate()."

    # Case B: Model starts in eval mode
    mini_model.eval()
    assert mini_model.training is False
    evaluate(mini_model, val_loader)
    assert mini_model.training is False, "Model eval mode was unexpectedly changed to train after evaluate()."


def test_evaluate_handles_empty_and_none_loader(mini_model: GPT):
    """Verifies that evaluate() returns None gracefully when val_loader is empty or None."""
    assert evaluate(mini_model, None) is None

    empty_dataset = GPTDataset([])
    empty_loader = DataLoader(empty_dataset, batch_size=2)
    assert evaluate(mini_model, empty_loader) is None


def test_save_and_load_checkpoint_roundtrip(mini_model: GPT, mini_config: GPTConfig, tmp_path):
    """Verifies that save_checkpoint and load_checkpoint accurately serialize and restore model state."""
    checkpoint_file = str(tmp_path / "subfolder" / "test_checkpoint.pt")
    optimizer = torch.optim.Adam(mini_model.parameters(), lr=1e-3)

    save_checkpoint(
        filepath=checkpoint_file,
        model=mini_model,
        config=mini_config,
        optimizer=optimizer,
        epoch=2,
        step=50,
        train_loss=3.1415,
        val_loss=2.7182,
    )

    assert os.path.exists(checkpoint_file)

    # Load into a fresh model instance
    loaded_model, loaded_config, meta = load_checkpoint(checkpoint_file, device="cpu")

    # Verify config reconstruction
    assert loaded_config.vocab_size == mini_config.vocab_size
    assert loaded_config.context_length == mini_config.context_length
    assert loaded_config.d_model == mini_config.d_model
    assert loaded_config.n_head == mini_config.n_head
    assert loaded_config.n_layer == mini_config.n_layer

    # Verify metadata
    assert meta["epoch"] == 2
    assert meta["step"] == 50
    assert pytest.approx(meta["train_loss"]) == 3.1415
    assert pytest.approx(meta["val_loss"]) == 2.7182

    # Verify all loaded weights match exactly
    for name, param in mini_model.named_parameters():
        loaded_param = dict(loaded_model.named_parameters())[name]
        assert torch.equal(param, loaded_param), f"Loaded parameter {name} does not match original."


def test_load_checkpoint_missing_file_raises():
    """Verifies that load_checkpoint raises FileNotFoundError when given an invalid path."""
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        load_checkpoint("non_existent_directory/checkpoint.pt")


def test_train_orchestrator_synthetic_run(mini_model: GPT, mini_config: GPTConfig, tmp_path):
    """Verifies that train() completes a full micro-run on synthetic data with history and checkpoint."""
    checkpoint_file = str(tmp_path / "micro_run.pt")
    train_ds, val_ds = create_synthetic_verification_dataset(
        context_length=mini_config.context_length,
        num_samples=16,
        vocab_size=mini_config.vocab_size,
    )
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    history = train(
        model=mini_model,
        config=mini_config,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        lr=1e-3,
        device="cpu",
        eval_interval=2,
        max_train_steps=4,
        max_val_batches=2,
        checkpoint_path=checkpoint_file,
    )

    assert "train_losses" in history
    assert len(history["train_losses"]) == 4
    assert os.path.exists(checkpoint_file)
    history_file = str(tmp_path / "micro_run_history.json")
    assert os.path.exists(history_file), "Expected training history JSON file to be saved."
    with open(history_file, "r", encoding="utf-8") as f:
        saved_history = json.load(f)
    assert "train_losses" in saved_history
    assert len(saved_history["train_losses"]) == 4


def test_cuda_device_fails_fast_when_unavailable(monkeypatch):
    """Verifies that requesting CUDA when unavailable raises RuntimeError instead of falling back to CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from src.training.trainer import main as trainer_main

    monkeypatch.setattr("sys.argv", ["trainer.py", "--device", "cuda"])
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        trainer_main()
