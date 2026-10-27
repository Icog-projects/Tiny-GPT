"""Training loop, validation evaluation, and model checkpointing."""

import argparse
from dataclasses import asdict
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.dataset import (
    GPTDataset,
    build_chunks_from_token_lists,
    create_dataloader,
    extract_documents_from_raw_lines,
    load_wikitext_split,
    tokenize_documents,
)
from src.data.tokenizer import GPT2Tokenizer
from src.model.config import GPTConfig
from src.model.gpt import GPT


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculates cross-entropy loss for causal next-token prediction.

    Args:
        logits: Unnormalized logits of shape (B, T, vocab_size).
        targets: Target integer token IDs of shape (B, T).

    Returns:
        Scalar cross-entropy loss tensor.
    """
    if logits.dim() != 3:
        raise ValueError(f"Expected 3D logits of shape (B, T, vocab_size), got {logits.dim()}D.")
    if targets.dim() != 2:
        raise ValueError(f"Expected 2D targets of shape (B, T), got {targets.dim()}D.")

    B, T, V = logits.shape
    if targets.shape != (B, T):
        raise ValueError(f"Shape mismatch: logits (B={B}, T={T}) vs targets {targets.shape}.")

    # Reshape (B, T, V) -> (B*T, V) and (B, T) -> (B*T,)
    flat_logits = logits.view(-1, V)
    flat_targets = targets.view(-1)

    return nn.functional.cross_entropy(flat_logits, flat_targets)


def train_one_step(
    model: GPT,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: optim.Optimizer,
    max_norm: float = 1.0,
) -> float:
    """Executes a single forward and backward optimization step.

    Args:
        model: GPT model instance.
        x: Input batch tensor of shape (B, T).
        y: Target batch tensor of shape (B, T).
        optimizer: PyTorch optimizer (e.g. AdamW).
        max_norm: Maximum gradient norm threshold for clipping (disabled if <= 0).

    Returns:
        Loss value as a float.
    """
    model.train()
    optimizer.zero_grad()

    logits = model(x)
    loss = compute_loss(logits, y)
    loss.backward()

    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

    optimizer.step()
    return loss.item()


def evaluate(
    model: GPT,
    val_loader: Optional[DataLoader],
    max_batches: Optional[int] = None,
    device: str = "cpu",
) -> Optional[float]:
    """Evaluates average validation loss without updating model weights.

    Args:
        model: GPT model instance.
        val_loader: DataLoader yielding validation batches (x, y).
        max_batches: Optional ceiling on number of batches to evaluate.
        device: Device to place tensors on ("cpu" or "cuda").

    Returns:
        Mean validation loss as float, or None if val_loader is empty or unavailable.
    """
    if val_loader is None or len(val_loader) == 0:
        return None

    was_training = model.training
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = compute_loss(logits, y)
            total_loss += loss.item()
            num_batches += 1

    if was_training:
        model.train()

    if num_batches == 0:
        return None

    return total_loss / num_batches


def save_checkpoint(
    filepath: str,
    model: GPT,
    config: GPTConfig,
    optimizer: Optional[optim.Optimizer] = None,
    epoch: int = 0,
    step: int = 0,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
) -> None:
    """Saves model weights, configuration, and training metadata to disk.

    Args:
        filepath: Destination path for the .pt checkpoint file.
        model: Trained GPT model instance.
        config: Model configuration dataclass.
        optimizer: Optional optimizer to preserve momentum states.
        epoch: Current training epoch index.
        step: Current global training step.
        train_loss: Most recent training loss.
        val_loss: Most recent validation loss.
    """
    dir_path = os.path.dirname(os.path.abspath(filepath))
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config_dict": asdict(config),
        "epoch": epoch,
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    device: str = "cpu",
    model: Optional[GPT] = None,
) -> Tuple[GPT, GPTConfig, Dict[str, Any]]:
    """Loads a saved checkpoint into a freshly constructed or existing GPT model.

    Args:
        filepath: Path to the .pt checkpoint file.
        device: Device to place the loaded model onto ("cpu" or "cuda").
        model: Optional existing GPT model instance to load weights into.

    Returns:
        Tuple of (model, config, raw_checkpoint_dict).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["config_dict"])

    if model is None:
        model = GPT(config)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, config, checkpoint


def create_synthetic_verification_dataset(
    context_length: int = 32,
    num_samples: int = 32,
    vocab_size: int = 50257,
) -> Tuple[GPTDataset, GPTDataset]:
    """Creates synthetic (x, y) datasets for offline micro-verification runs without network access.

    Args:
        context_length: Sequence length T.
        num_samples: Number of synthetic training samples.
        vocab_size: Upper bound for random token IDs.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    train_samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
    val_samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for i in range(num_samples):
        chunk = torch.randint(0, min(vocab_size, 1000), (context_length + 1,), dtype=torch.long)
        x = chunk[:context_length]
        y = chunk[1 : context_length + 1]
        train_samples.append((x, y))

    for i in range(max(4, num_samples // 4)):
        chunk = torch.randint(0, min(vocab_size, 1000), (context_length + 1,), dtype=torch.long)
        x = chunk[:context_length]
        y = chunk[1 : context_length + 1]
        val_samples.append((x, y))

    return GPTDataset(train_samples), GPTDataset(val_samples)


def train(
    model: GPT,
    config: GPTConfig,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 1,
    lr: float = 3e-4,
    device: str = "cpu",
    eval_interval: int = 10,
    max_train_steps: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    checkpoint_path: str = "checkpoints/tiny_gpt.pt",
) -> Dict[str, Any]:
    """Runs the training and validation loops with automatic checkpointing.

    Args:
        model: GPT model initialized with random weights.
        config: Model configuration.
        train_loader: DataLoader yielding training batches.
        val_loader: Optional DataLoader yielding validation batches.
        epochs: Number of complete training epochs.
        lr: Learning rate for AdamW optimizer.
        device: Device to train on ("cpu" or "cuda").
        eval_interval: Steps between validation evaluations.
        max_train_steps: Optional maximum global training steps to execute.
        max_val_batches: Optional limit on batches evaluated per validation pass.
        checkpoint_path: Filepath where best model checkpoint is saved.

    Returns:
        Dictionary containing training history metrics.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    global_step = 0
    history: Dict[str, List[Any]] = {
        "train_steps": [],
        "train_losses": [],
        "val_steps": [],
        "val_losses": [],
    }

    print(f"--- Starting Training on {device.upper()} ---")
    print(f"Model parameters: {model.get_num_params():,} (non-embedding: {model.get_num_params(non_embedding=True):,})")
    print(f"Configuration: layers={config.n_layer}, heads={config.n_head}, d_model={config.d_model}, context={config.context_length}")

    start_time = time.time()

    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            if max_train_steps is not None and global_step >= max_train_steps:
                break

            x = x.to(device)
            y = y.to(device)

            loss = train_one_step(model, x, y, optimizer)
            global_step += 1

            history["train_steps"].append(global_step)
            history["train_losses"].append(loss)

            if global_step % eval_interval == 0 or (max_train_steps and global_step == max_train_steps):
                perplexity = math.exp(loss) if loss < 20 else float("inf")
                print(f"Epoch {epoch + 1}/{epochs} | Step {global_step} | Train Loss: {loss:.4f} | Perplexity: {perplexity:.2f}")

                # Run validation
                if val_loader is not None:
                    val_loss = evaluate(model, val_loader, max_batches=max_val_batches, device=device)
                    if val_loss is not None:
                        val_perp = math.exp(val_loss) if val_loss < 20 else float("inf")
                        history["val_steps"].append(global_step)
                        history["val_losses"].append(val_loss)

                        if val_loss < best_val_loss:
                            improvement = best_val_loss - val_loss if best_val_loss != float("inf") else 0.0
                            best_val_loss = val_loss
                            save_checkpoint(
                                checkpoint_path,
                                model,
                                config,
                                optimizer=optimizer,
                                epoch=epoch + 1,
                                step=global_step,
                                train_loss=loss,
                                val_loss=val_loss,
                            )
                            print(f"  --> [IMPROVEMENT] Val Loss: {val_loss:.4f} (improved by {improvement:.4f}) | Checkpoint saved: {checkpoint_path}")
                        else:
                            print(f"  --> [NO IMPROVEMENT] Val Loss: {val_loss:.4f} (best: {best_val_loss:.4f}) | Retaining previous checkpoint")

        if max_train_steps is not None and global_step >= max_train_steps:
            break

    total_time = time.time() - start_time
    print(f"--- Training Finished in {total_time:.2f}s across {global_step} steps ---")

    # If no validation was available, save the final model checkpoint
    if not os.path.exists(checkpoint_path):
        save_checkpoint(
            checkpoint_path,
            model,
            config,
            optimizer=optimizer,
            epoch=epochs,
            step=global_step,
            train_loss=history["train_losses"][-1] if history["train_losses"] else None,
            val_loss=None,
        )
        print(f"Saved checkpoint to {checkpoint_path}")

    # Preserve training and validation history to a structured JSON file alongside the checkpoint
    history_path = os.path.splitext(checkpoint_path)[0] + "_history.json"
    dir_path = os.path.dirname(os.path.abspath(history_path))
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history metrics saved to {history_path}")

    return history


def main() -> None:
    """CLI entry point for training and verification."""
    parser = argparse.ArgumentParser(description="Train or verify the Tiny GPT language model.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--context_length", type=int, default=32, help="Context length T.")
    parser.add_argument("--d_model", type=int, default=64, help="Embedding dimension d_model.")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--n_layer", type=int, default=2, help="Number of Transformer blocks.")
    parser.add_argument("--d_ff", type=int, default=256, help="Feed-forward expansion dimension.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use ('cpu' or 'cuda').")
    parser.add_argument("--eval_interval", type=int, default=5, help="Steps between validation evaluations.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Ceiling on total training steps.")
    parser.add_argument("--max_val_batches", type=int, default=10, help="Max batches to evaluate in validation.")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/tiny_gpt.pt", help="Checkpoint output path.")
    parser.add_argument("--verification", action="store_true", help="Run a quick CPU verification run on synthetic data.")

    args = parser.parse_args()

    # Determine execution device - fail immediately if CUDA requested but unavailable
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested (--device cuda), but torch.cuda.is_available() is False. "
            "Please ensure a GPU runtime is selected (e.g., in Google Colab: Runtime -> Change runtime type -> T4 GPU) "
            "or pass --device cpu explicitly."
        )

    config = GPTConfig(
        vocab_size=50257,
        context_length=args.context_length,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        d_ff=args.d_ff,
    )
    model = GPT(config)

    if args.verification:
        print("=== Running Quick Verification Run on Synthetic Data ===")
        train_ds, val_ds = create_synthetic_verification_dataset(
            context_length=args.context_length,
            num_samples=32,
            vocab_size=config.vocab_size,
        )
        train_loader = create_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = create_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
        max_steps = args.max_train_steps if args.max_train_steps is not None else 10

        train(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            eval_interval=args.eval_interval,
            max_train_steps=max_steps,
            max_val_batches=args.max_val_batches,
            checkpoint_path=args.checkpoint_path,
        )
        print("Verification completed successfully.")
        return

    # Attempt to load real WikiText-2 dataset
    print("Loading WikiText-2 dataset...")
    try:
        raw_train_lines = load_wikitext_split("train")
        raw_val_lines = load_wikitext_split("validation")
    except Exception as e:
        print(f"\n[NOTICE] Could not load WikiText-2 from Hugging Face: {e}")
        print("WikiText-2 must be downloaded once before a full training run can proceed.")
        print("To verify the complete training, validation, and checkpointing pipeline offline on CPU, run:")
        print("  python -m src.training.trainer --verification\n")
        return

    tokenizer = GPT2Tokenizer()

    print("Extracting and tokenizing train documents...")
    train_docs = extract_documents_from_raw_lines(raw_train_lines)
    train_tokens = tokenize_documents(train_docs, tokenizer, append_eot=True)
    train_chunks = build_chunks_from_token_lists(train_tokens, context_length=args.context_length)
    train_ds = GPTDataset(train_chunks)
    train_loader = create_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)

    print("Extracting and tokenizing validation documents...")
    val_docs = extract_documents_from_raw_lines(raw_val_lines)
    val_tokens = tokenize_documents(val_docs, tokenizer, append_eot=True)
    val_chunks = build_chunks_from_token_lists(val_tokens, context_length=args.context_length)
    val_ds = GPTDataset(val_chunks)
    val_loader = create_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Loaded {len(train_ds):,} training samples and {len(val_ds):,} validation samples.")

    train(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        eval_interval=args.eval_interval,
        max_train_steps=args.max_train_steps,
        max_val_batches=args.max_val_batches,
        checkpoint_path=args.checkpoint_path,
    )


if __name__ == "__main__":
    main()
