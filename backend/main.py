"""FastAPI application — Tiny GPT backend.

Exposes exactly ONE custom endpoint:
    POST /generate

The model is loaded once during application start-up.  Automatic API docs
are available at /docs (FastAPI default; not a custom endpoint).
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import GenerateRequest, GenerateResponse
from backend import service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan — load model once at startup, release at shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the GPT model on startup; unload on shutdown."""
    logger.info("Application starting — attempting model load …")
    service.load_model()
    if service.is_model_loaded():
        logger.info("Model is ready to serve requests.")
    else:
        logger.warning(
            "Model is NOT loaded. POST /generate will return 503 until a valid "
            "checkpoint is configured via TINY_GPT_CHECKPOINT."
        )
    yield
    logger.info("Application shutting down — releasing model resources.")
    service.unload_model()


app = FastAPI(
    title="Tiny GPT API",
    description=(
        "A minimal REST API for the iCog Labs Tiny GPT model. "
        "Exposes one endpoint: POST /generate."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local Vite development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# POST /generate — the single custom endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate text from a prompt",
    description=(
        "Accepts a text prompt and a token count, then returns the model-generated "
        "continuation.  Requires a trained checkpoint to be configured via the "
        "TINY_GPT_CHECKPOINT environment variable."
    ),
)
def generate(request: GenerateRequest) -> GenerateResponse:
    """Run autoregressive text generation and return the result.

    Returns HTTP 503 when no model is loaded.
    Returns HTTP 422 automatically for invalid request bodies (Pydantic).
    """
    if not service.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail=(
                "The model is not available. "
                "Configure a valid checkpoint path with the TINY_GPT_CHECKPOINT "
                "environment variable and restart the server."
            ),
        )

    try:
        generated_text = service.run_generation(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
        )
    except RuntimeError as exc:
        # Model became unavailable between the is_model_loaded() check and generation.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        # Propagated from generate_text for invalid max_new_tokens values.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return GenerateResponse(
        prompt=request.prompt,
        generated_text=generated_text,
        max_new_tokens=request.max_new_tokens,
    )
