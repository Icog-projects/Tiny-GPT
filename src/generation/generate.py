"""Autoregressive text generation using greedy decoding."""

import argparse
import os
from typing import List, Optional, Tuple

import torch

from src.data.tokenizer import GPT2Tokenizer
from src.model.gpt import GPT
from src.training.trainer import load_checkpoint


def generate_text(
    model: GPT,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    context_length: Optional[int] = None,
    device: str = "cpu",
    return_token_ids: bool = False,
) -> str | Tuple[str, List[int]]:
    """Generates text autoregressively using greedy decoding.

    Args:
        model: Trained GPT model.
        tokenizer: Initialized GPT2Tokenizer.
        prompt: Initial text string. If empty or whitespace, generation starts
            unconditionally with the EOT (<|endoftext|>) token.
        max_new_tokens: Number of subsequent tokens to generate. Must be >= 1.
        context_length: Maximum sequence length the model accepts. Defaults to
            model.config.context_length.
        device: Device to place input tensors on ("cpu" or "cuda").
        return_token_ids: If True, returns a tuple of (decoded_text, token_ids_list).

    Returns:
        The generated text as a string (including the prompt), or a tuple of
        (generated_text, token_ids_list) if return_token_ids is True.

    Raises:
        ValueError: If max_new_tokens < 1.
    """
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be a positive integer (>= 1), got {max_new_tokens}.")

    ctx_len = context_length if context_length is not None else model.config.context_length
    model.to(device)

    # Preserve previous training state
    was_training = model.training
    model.eval()

    # Handle prompt encoding: empty prompt starts unconditionally with EOT token
    stripped_prompt = prompt.strip()
    if not stripped_prompt:
        token_ids = [tokenizer.eot_token_id]
    else:
        token_ids = tokenizer.encode(prompt)
        if not token_ids:
            token_ids = [tokenizer.eot_token_id]

    idx = torch.tensor([token_ids], dtype=torch.long, device=device)  # (1, N)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to the most recent context_length tokens if the sequence exceeds it
            idx_cond = idx if idx.size(1) <= ctx_len else idx[:, -ctx_len:]

            # Forward pass to obtain raw vocabulary logits
            logits = model(idx_cond)  # (1, T, vocab_size)

            # Focus on the logits at the final position
            next_token_logits = logits[:, -1, :]  # (1, vocab_size)

            # Greedy selection: pick the token with the highest predicted score
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (1, 1)

            # Append the new token to the running sequence
            idx = torch.cat((idx, next_token), dim=1)

    # Restore prior training mode
    if was_training:
        model.train()

    generated_ids = idx[0].tolist()

    # If we prepended an EOT token solely because prompt was empty, remove it before decoding
    if not stripped_prompt and generated_ids and generated_ids[0] == tokenizer.eot_token_id:
        generated_ids = generated_ids[1:]

    decoded_text = tokenizer.decode(generated_ids)
    if return_token_ids:
        return decoded_text, generated_ids
    return decoded_text


def main() -> None:
    """CLI entry point for standalone text generation from a saved checkpoint."""
    parser = argparse.ArgumentParser(description="Generate text using a trained Tiny GPT checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/tiny_gpt.pt",
        help="Path to the saved .pt checkpoint file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The meaning of life is",
        help="Initial text prompt for generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use ('cpu' or 'cuda').",
    )

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested (--device cuda), but torch.cuda.is_available() is False. "
            "Please ensure a GPU runtime is selected or pass --device cpu explicitly."
        )

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at '{args.checkpoint}'.")
        print("Please train or verify the model first to generate a checkpoint, for example:")
        print("  python -m src.training.trainer --verification --checkpoint_path checkpoints/tiny_gpt.pt\n")
        return

    print(f"Loading checkpoint from '{args.checkpoint}' onto {device.upper()}...")
    model, config, meta = load_checkpoint(args.checkpoint, device=device)
    print(f"Model loaded: {model.get_num_params():,} parameters (layer={config.n_layer}, heads={config.n_head}, d_model={config.d_model}).")
    if meta.get("val_loss") is not None:
        print(f"Checkpoint metadata: epoch={meta.get('epoch')}, step={meta.get('step')}, val_loss={meta.get('val_loss'):.4f}")

    tokenizer = GPT2Tokenizer()

    print("\n--- Generating Text ---")
    print(f"Prompt: {args.prompt!r}")
    print(f"Requested new tokens: {args.max_new_tokens}")
    print("-" * 40)

    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )

    print(f"Result:\n{output}")
    print("-" * 40)


if __name__ == "__main__":
    main()
