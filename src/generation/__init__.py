"""Autoregressive text generation and decoding module."""


def __getattr__(name: str):
    import src.generation.generate as generate
    if hasattr(generate, name):
        return getattr(generate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "generate_text",
]
