/**
 * API client module for the Tiny GPT FastAPI backend.
 *
 * Exposes a single function `generateText` that sends a POST request
 * to the `POST /generate` endpoint.
 */

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
).replace(/\/+$/, '');

export const API_LIMITS = {
  MAX_PROMPT_CHARS: 2000,
  MIN_NEW_TOKENS: 1,
  MAX_NEW_TOKENS: 500,
  DEFAULT_NEW_TOKENS: 50,
};

/**
 * Generate text continuation from a prompt using the Tiny GPT backend.
 *
 * @param {string} prompt - The text prompt to continue.
 * @param {number} maxNewTokens - Number of tokens to generate (1 to 500).
 * @returns {Promise<{prompt: string, generatedText: string, maxNewTokens: number}>}
 */
export async function generateText(prompt, maxNewTokens = API_LIMITS.DEFAULT_NEW_TOKENS) {
  // Client-side validations matching backend schemas
  if (typeof prompt !== 'string' || !prompt.trim()) {
    throw new Error('Please enter a non-empty text prompt.');
  }

  if (prompt.length > API_LIMITS.MAX_PROMPT_CHARS) {
    throw new Error(
      `Prompt exceeds maximum limit of ${API_LIMITS.MAX_PROMPT_CHARS} characters (currently ${prompt.length}).`
    );
  }

  const tokenCount = Number.parseInt(maxNewTokens, 10);
  if (
    Number.isNaN(tokenCount) ||
    tokenCount < API_LIMITS.MIN_NEW_TOKENS ||
    tokenCount > API_LIMITS.MAX_NEW_TOKENS
  ) {
    throw new Error(
      `Tokens to generate must be an integer between ${API_LIMITS.MIN_NEW_TOKENS} and ${API_LIMITS.MAX_NEW_TOKENS}.`
    );
  }

  const endpoint = `${API_BASE_URL}/generate`;
  let response;

  try {
    response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt: prompt,
        max_new_tokens: tokenCount,
      }),
    });
  } catch (networkError) {
    // Network failure (server not running, connection refused, CORS blocked, etc.)
    throw new Error(
      `Cannot connect to the Tiny GPT backend at ${API_BASE_URL}. ` +
      `Please verify that the FastAPI server is running with: ` +
      `"python -m uvicorn backend.main:app --reload"`
    );
  }

  if (!response.ok) {
    let errorMessage = `Server error (${response.status} ${response.statusText})`;
    try {
      const errorJson = await response.json();
      if (typeof errorJson.detail === 'string') {
        errorMessage = errorJson.detail;
      } else if (Array.isArray(errorJson.detail)) {
        // FastAPI / Pydantic validation errors
        errorMessage = errorJson.detail
          .map((err) => `${err.loc?.join('.') || 'field'}: ${err.msg}`)
          .join('; ');
      }
    } catch {
      // If response body is not JSON, retain default status text
    }

    if (response.status === 503) {
      throw new Error(
        `Model Unavailable (503): ${errorMessage}`
      );
    }

    if (response.status === 422) {
      throw new Error(
        `Invalid Request (422): ${errorMessage}`
      );
    }

    throw new Error(errorMessage);
  }

  const data = await response.json();

  if (typeof data.generated_text !== 'string') {
    throw new Error('Unexpected response format from backend (missing "generated_text").');
  }

  return {
    prompt: data.prompt,
    generatedText: data.generated_text,
    maxNewTokens: data.max_new_tokens,
  };
}

