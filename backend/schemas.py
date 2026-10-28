"""Pydantic request and response models for the POST /generate endpoint."""

from pydantic import BaseModel, Field, field_validator

# Hard limits — keep reasonable for a tiny model with context_length <= 64
MAX_PROMPT_CHARS: int = 2000
MAX_NEW_TOKENS_LIMIT: int = 500


class GenerateRequest(BaseModel):
    """Request body for POST /generate.

    Attributes:
        prompt: The text prompt to continue. May be empty (unconditional generation).
        max_new_tokens: Number of new tokens the model should append. Must be >= 1.
    """

    prompt: str = Field(
        ...,
        description="Input text prompt. May be empty for unconditional generation.",
        max_length=MAX_PROMPT_CHARS,
    )
    max_new_tokens: int = Field(
        default=50,
        ge=1,
        le=MAX_NEW_TOKENS_LIMIT,
        description=f"Number of new tokens to generate (1–{MAX_NEW_TOKENS_LIMIT}).",
    )

    @field_validator("prompt")
    @classmethod
    def prompt_must_be_string(cls, v: str) -> str:
        """Ensure prompt is a plain string (Pydantic already enforces type; belt-and-suspenders)."""
        return v


class GenerateResponse(BaseModel):
    """Response body returned by POST /generate.

    Attributes:
        prompt: The original prompt echoed back.
        generated_text: Full model output (includes the prompt text).
        max_new_tokens: The number of new tokens that were requested.
    """

    prompt: str
    generated_text: str
    max_new_tokens: int
