"""Offline unit tests for the Tiny GPT FastAPI backend.

All tests run without:
- downloading WikiText-2 or any other dataset
- loading pretrained weights
- requiring a real checkpoint file

The model and generation function are mocked so the suite is fully offline.

Real-checkpoint integration is explicitly NOT tested here; it remains untested
until the user provides a trained checkpoint path.
"""

import os
import importlib
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app_with_service(svc_module):
    """Re-import backend.main against the given (possibly patched) service module."""
    import backend.main as main_mod
    # Reload so it picks up any monkeypatches applied to backend.service
    importlib.reload(main_mod)
    return main_mod.app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def model_unavailable_client():
    """TestClient where no model is loaded (checkpoint missing)."""
    # Patch service so no model is present
    with patch("backend.service.load_model", return_value=None), \
         patch("backend.service.is_model_loaded", return_value=False), \
         patch("backend.service.unload_model", return_value=None):
        import backend.main as main_mod
        importlib.reload(main_mod)
        with TestClient(main_mod.app, raise_server_exceptions=False) as client:
            yield client


@pytest.fixture()
def model_available_client():
    """TestClient where the model is 'loaded' (mocked generate function)."""
    fake_output = "The history of science is a story of progress"

    with patch("backend.service.load_model", return_value=None), \
         patch("backend.service.is_model_loaded", return_value=True), \
         patch("backend.service.unload_model", return_value=None), \
         patch("backend.service.run_generation", return_value=fake_output) as mock_gen:
        import backend.main as main_mod
        importlib.reload(main_mod)
        with TestClient(main_mod.app, raise_server_exceptions=False) as client:
            yield client, mock_gen


# ===========================================================================
# 1. Valid generation request — response structure
# ===========================================================================

class TestValidGenerationRequest:
    def test_returns_200(self, model_available_client):
        client, _ = model_available_client
        resp = client.post("/generate", json={"prompt": "The history of science", "max_new_tokens": 30})
        assert resp.status_code == 200

    def test_response_contains_prompt(self, model_available_client):
        client, _ = model_available_client
        resp = client.post("/generate", json={"prompt": "The history of science", "max_new_tokens": 30})
        data = resp.json()
        assert data["prompt"] == "The history of science"

    def test_response_contains_generated_text(self, model_available_client):
        client, _ = model_available_client
        resp = client.post("/generate", json={"prompt": "The history of science", "max_new_tokens": 30})
        data = resp.json()
        assert "generated_text" in data
        assert isinstance(data["generated_text"], str)
        # Must NOT be empty
        assert len(data["generated_text"]) > 0

    def test_response_contains_max_new_tokens(self, model_available_client):
        client, _ = model_available_client
        resp = client.post("/generate", json={"prompt": "The history of science", "max_new_tokens": 30})
        data = resp.json()
        assert data["max_new_tokens"] == 30

    def test_response_is_not_hardcoded(self, model_available_client):
        """The returned text must come from the (mocked) generation function, not a hardcoded string."""
        client, mock_gen = model_available_client
        mock_gen.return_value = "UNIQUE_SENTINEL_OUTPUT_XYZ"
        resp = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 5})
        assert resp.json()["generated_text"] == "UNIQUE_SENTINEL_OUTPUT_XYZ"

    def test_generation_function_is_called(self, model_available_client):
        """Confirms that run_generation is invoked with the correct arguments."""
        client, mock_gen = model_available_client
        client.post("/generate", json={"prompt": "Test prompt", "max_new_tokens": 10})
        mock_gen.assert_called_once_with(prompt="Test prompt", max_new_tokens=10)

    def test_empty_prompt_is_accepted(self, model_available_client):
        """Empty prompts are valid — generate_text handles unconditional generation."""
        client, _ = model_available_client
        resp = client.post("/generate", json={"prompt": "", "max_new_tokens": 5})
        assert resp.status_code == 200

    def test_default_max_new_tokens(self, model_available_client):
        """max_new_tokens has a default of 50 when omitted."""
        client, mock_gen = model_available_client
        client.post("/generate", json={"prompt": "hello"})
        mock_gen.assert_called_once_with(prompt="hello", max_new_tokens=50)


# ===========================================================================
# 2. Invalid request bodies — Pydantic validation (HTTP 422)
# ===========================================================================

class TestInvalidRequestValidation:
    def test_missing_prompt_field(self, model_available_client):
        client, _ = model_available_client
        resp = client.post("/generate", json={"max_new_tokens": 10})
        assert resp.status_code == 422

    def test_null_prompt(self, model_available_client):
        client, _ = model_available_client
        resp = client.post("/generate", json={"prompt": None, "max_new_tokens": 10})
        assert resp.status_code == 422

    def test_negative_max_new_tokens(self, model_available_client):
        client, _ = model_available_client
        resp = client.post("/generate", json={"prompt": "hello", "max_new_tokens": -1})
        assert resp.status_code == 422

    def test_zero_max_new_tokens(self, model_available_client):
        client, _ = model_available_client
        resp = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 0})
        assert resp.status_code == 422

    def test_excessive_max_new_tokens(self, model_available_client):
        """max_new_tokens above MAX_NEW_TOKENS_LIMIT (500) is rejected."""
        client, _ = model_available_client
        resp = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 501})
        assert resp.status_code == 422

    def test_prompt_too_long(self, model_available_client):
        """Prompt exceeding MAX_PROMPT_CHARS (2000) is rejected."""
        client, _ = model_available_client
        resp = client.post("/generate", json={"prompt": "x" * 2001, "max_new_tokens": 5})
        assert resp.status_code == 422

    def test_non_integer_max_new_tokens(self, model_available_client):
        client, _ = model_available_client
        resp = client.post("/generate", json={"prompt": "hello", "max_new_tokens": "lots"})
        assert resp.status_code == 422

    def test_non_string_prompt(self, model_available_client):
        client, _ = model_available_client
        resp = client.post("/generate", json={"prompt": 12345, "max_new_tokens": 5})
        assert resp.status_code == 422


# ===========================================================================
# 3. Missing / unavailable checkpoint — HTTP 503
# ===========================================================================

class TestMissingCheckpoint:
    def test_generate_returns_503_when_model_unavailable(self, model_unavailable_client):
        resp = model_unavailable_client.post(
            "/generate", json={"prompt": "hello", "max_new_tokens": 5}
        )
        assert resp.status_code == 503

    def test_503_detail_mentions_checkpoint(self, model_unavailable_client):
        resp = model_unavailable_client.post(
            "/generate", json={"prompt": "hello", "max_new_tokens": 5}
        )
        detail = resp.json().get("detail", "")
        # The error message must guide the user without exposing raw paths.
        assert "checkpoint" in detail.lower() or "TINY_GPT_CHECKPOINT" in detail

    def test_server_starts_even_without_checkpoint(self):
        """The FastAPI app must be importable and start without a checkpoint."""
        with patch("backend.service.load_model", return_value=None), \
             patch("backend.service.is_model_loaded", return_value=False), \
             patch("backend.service.unload_model", return_value=None):
            import backend.main as main_mod
            importlib.reload(main_mod)
            # A TestClient context manager exercises the lifespan without error.
            with TestClient(main_mod.app, raise_server_exceptions=False):
                pass  # No exception means the server started cleanly.


# ===========================================================================
# 4. Model-loading failure — service.load_model error path
# ===========================================================================

class TestModelLoadingFailure:
    def test_load_model_sets_error_on_bad_checkpoint(self, tmp_path):
        """If the checkpoint file is corrupt / wrong format, load_model sets _load_error."""
        bad_file = tmp_path / "bad.pt"
        bad_file.write_bytes(b"not a valid pytorch file")

        import backend.service as svc
        # Reset state
        svc._model = None
        svc._tokenizer = None
        svc._load_error = None

        with patch.dict(os.environ, {"TINY_GPT_CHECKPOINT": str(bad_file)}):
            svc.load_model()

        assert svc._model is None
        assert svc._load_error is not None
        # Error message must NOT contain the full filesystem path
        assert str(bad_file) not in svc._load_error

        # Cleanup
        svc._model = None
        svc._tokenizer = None
        svc._load_error = None

    def test_run_generation_raises_runtime_error_when_no_model(self):
        """run_generation raises RuntimeError when no model is loaded."""
        import backend.service as svc
        original_model = svc._model
        original_tokenizer = svc._tokenizer
        svc._model = None
        svc._tokenizer = None
        svc._load_error = "Test: no model available."
        try:
            with pytest.raises(RuntimeError, match="no model available"):
                svc.run_generation("hello", 5)
        finally:
            svc._model = original_model
            svc._tokenizer = original_tokenizer
            svc._load_error = None

    def test_503_when_run_generation_raises_runtime_error(self):
        """If run_generation raises RuntimeError, endpoint returns 503."""
        with patch("backend.service.load_model", return_value=None), \
             patch("backend.service.is_model_loaded", return_value=True), \
             patch("backend.service.unload_model", return_value=None), \
             patch("backend.service.run_generation", side_effect=RuntimeError("model gone")):
            import backend.main as main_mod
            importlib.reload(main_mod)
            with TestClient(main_mod.app, raise_server_exceptions=False) as client:
                resp = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 5})
            assert resp.status_code == 503


# ===========================================================================
# 5. Training is never triggered
# ===========================================================================

class TestNoTrainingTriggered:
    def test_training_functions_not_called(self, model_available_client):
        """Confirms that train_one_step and train() are never invoked by the backend."""
        client, _ = model_available_client
        with patch("src.training.trainer.train_one_step") as mock_step, \
             patch("src.training.trainer.train") as mock_train:
            client.post("/generate", json={"prompt": "hello", "max_new_tokens": 5})
            mock_step.assert_not_called()
            mock_train.assert_not_called()


# ===========================================================================
# 6. Only POST /generate exists as a custom endpoint
# ===========================================================================

class TestSingleEndpoint:
    def test_only_generate_endpoint_is_registered(self):
        """Verify exactly one user-defined route: POST /generate."""
        import backend.main as main_mod
        importlib.reload(main_mod)
        # Collect all routes, excluding FastAPI's automatic openapi/docs routes
        auto_paths = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
        custom_routes = [
            r for r in main_mod.app.routes
            if hasattr(r, "path") and r.path not in auto_paths
        ]
        assert len(custom_routes) == 1, (
            f"Expected exactly 1 custom route, found: {[r.path for r in custom_routes]}"
        )
        route = custom_routes[0]
        assert route.path == "/generate"
        assert "POST" in route.methods
