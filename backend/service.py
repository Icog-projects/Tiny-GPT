"""Model loading and inference service for the Tiny GPT backend.

Responsibilities
----------------
* Read the checkpoint path from the ``TINY_GPT_CHECKPOINT`` environment variable.
* Load the GPT model once at application start-up (when the checkpoint exists).
* Expose a single ``run_generation`` function that delegates to the existing
  ``generate_text`` implementation without modifying it.
* Return clear, client-safe error messages; never expose filesystem paths or
  raw exception tracebacks to API callers.
"""

import logging
import os
from typing import Optional

import torch

from src.data.tokenizer import GPT2Tokenizer
from src.generation.generate import generate_text
from src.model.config import GPTConfig
from src.model.gpt import GPT
from src.training.trainer import load_checkpoint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Checkpoint path configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CHECKPOINT_PATH = "checkpoints/tiny_gpt_wikitext2_60k_best.pt"


def resolve_checkpoint_path(path: str) -> str:
    """Resolves a checkpoint path reliably against the project root.

    Args:
        path: Absolute or relative filesystem path.

    Returns:
        Absolute resolved path string.
    """
    if os.path.isabs(path):
        return os.path.abspath(path)

    # First resolve relative to project root
    root_resolved = os.path.abspath(os.path.join(PROJECT_ROOT, path))
    if os.path.isfile(root_resolved):
        return root_resolved

    # Fallback to current working directory if it exists there
    if os.path.isfile(path):
        return os.path.abspath(path)

    # Default to root_resolved for consistent error reporting
    return root_resolved


# At import time we resolve the path from the environment variable.
# The value is stored here so tests can inspect or monkeypatch it.
_checkpoint_path: str = resolve_checkpoint_path(
    os.environ.get("TINY_GPT_CHECKPOINT", DEFAULT_CHECKPOINT_PATH)
)

# ---------------------------------------------------------------------------
# Module-level singletons — populated during lifespan startup
# ---------------------------------------------------------------------------

_model: Optional[GPT] = None
_tokenizer: Optional[GPT2Tokenizer] = None
_device: str = "cpu"
_load_error: Optional[str] = None   # human-readable reason the model is unavailable


def _select_device() -> str:
    """Returns ``'cuda'`` when a GPU is available, otherwise ``'cpu'``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model() -> None:
    """Attempt to load the model from the configured checkpoint path.

    Sets the module-level ``_model``, ``_tokenizer``, ``_device``, and
    ``_load_error`` variables.  Safe to call repeatedly; a second call while
    the model is already loaded is a no-op.

    Reads ``TINY_GPT_CHECKPOINT`` from the environment at call time so that
    tests can monkeypatch ``os.environ`` before calling this function.
    """
    global _model, _tokenizer, _device, _load_error, _checkpoint_path

    # Re-read the env var so callers (e.g. tests) can patch it before invoking.
    raw_path = os.environ.get("TINY_GPT_CHECKPOINT", DEFAULT_CHECKPOINT_PATH)
    _checkpoint_path = resolve_checkpoint_path(raw_path)

    if _model is not None:
        # Already loaded — skip.
        logger.info("Model already loaded; skipping repeated load.")
        return

    if not os.path.isfile(_checkpoint_path):
        _load_error = (
            "No valid checkpoint is configured. "
            f"Set the TINY_GPT_CHECKPOINT environment variable to the path of "
            f"a trained .pt checkpoint file and restart the server."
        )
        logger.warning("Checkpoint not found at %r — model unavailable.", _checkpoint_path)
        return

    _device = _select_device()
    logger.info("Loading checkpoint from %r onto %s …", _checkpoint_path, _device.upper())

    try:
        loaded_model, config, meta = load_checkpoint(_checkpoint_path, device=_device)
        loaded_model.eval()
        _model = loaded_model
        _tokenizer = GPT2Tokenizer()
        _load_error = None
        logger.info(
            "Model loaded successfully: %s parameters "
            "(layers=%d, heads=%d, d_model=%d, context=%d).",
            f"{_model.get_num_params():,}",
            config.n_layer,
            config.n_head,
            config.d_model,
            config.context_length,
        )
        if meta.get("val_loss") is not None:
            logger.info(
                "Checkpoint metadata — epoch=%s, step=%s, val_loss=%.4f",
                meta.get("epoch"),
                meta.get("step"),
                meta["val_loss"],
            )
    except Exception as exc:
        _load_error = (
            "The model could not be loaded from the configured checkpoint. "
            "Ensure the checkpoint file is a valid Tiny GPT .pt file produced "
            "by this project's training pipeline."
        )
        # Log the full detail server-side; never forward it to the client.
        logger.exception("Failed to load checkpoint: %s", exc)


def unload_model() -> None:
    """Reset all module-level singletons (used during application shutdown)."""
    global _model, _tokenizer, _load_error
    _model = None
    _tokenizer = None
    _load_error = None


def is_model_loaded() -> bool:
    """Return ``True`` when the model is ready to serve requests."""
    return _model is not None


def run_generation(prompt: str, max_new_tokens: int) -> str:
    """Generate text using the loaded model.

    Args:
        prompt: Input text.  May be empty — the existing ``generate_text``
            implementation handles empty prompts by using the EOT token.
        max_new_tokens: Number of new tokens to append.  Must be >= 1.

    Returns:
        The generated text string (includes the prompt).

    Raises:
        RuntimeError: When no model is available (checkpoint missing or failed
            to load).
        ValueError: When ``max_new_tokens < 1`` (propagated from
            ``generate_text``).
    """
    if _model is None or _tokenizer is None:
        reason = _load_error or (
            "No checkpoint has been configured. "
            "Set TINY_GPT_CHECKPOINT to a valid .pt file path and restart."
        )
        raise RuntimeError(reason)

    # Clamp to the model's actual context length so the caller cannot request
    # more tokens than the architecture can produce in one pass.
    ctx_len: int = _model.config.context_length

    with torch.no_grad():
        result = generate_text(
            model=_model,
            tokenizer=_tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            context_length=ctx_len,
            device=_device,
        )

    return result
