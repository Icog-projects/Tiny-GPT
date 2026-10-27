"""Training loop, validation, and checkpointing module."""


def __getattr__(name: str):
    import src.training.trainer as trainer
    if hasattr(trainer, name):
        return getattr(trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "compute_loss",
    "train_one_step",
    "evaluate",
    "save_checkpoint",
    "load_checkpoint",
    "train",
    "create_synthetic_verification_dataset",
]
