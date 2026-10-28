import React, { useState } from 'react';
import { generateText, API_LIMITS } from './api';

const SAMPLE_PROMPTS = [
  'The history of science',
  'Artificial intelligence is',
  'In the early twentieth century,',
  'The exploration of space',
];

export default function App() {
  const [prompt, setPrompt] = useState('The history of science');
  const [maxNewTokens, setMaxNewTokens] = useState(API_LIMITS.DEFAULT_NEW_TOKENS);
  const [generatedResult, setGeneratedResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [copied, setCopied] = useState(false);

  const trimmedPrompt = prompt.trim();
  const isPromptEmpty = trimmedPrompt.length === 0;
  const isPromptTooLong = prompt.length > API_LIMITS.MAX_PROMPT_CHARS;

  async function handleGenerate(e) {
    if (e) e.preventDefault();
    if (isPromptEmpty || isPromptTooLong || isLoading) return;

    setIsLoading(true);
    setErrorMessage('');
    setCopied(false);

    try {
      const result = await generateText(prompt, maxNewTokens);
      setGeneratedResult(result);
    } catch (err) {
      setErrorMessage(err.message || 'An unexpected error occurred.');
    } finally {
      setIsLoading(false);
    }
  }

  async function handleCopy() {
    if (!generatedResult?.generatedText) return;
    try {
      await navigator.clipboard.writeText(generatedResult.generatedText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback if clipboard API fails
      setCopied(false);
    }
  }

  function handleReset() {
    setPrompt('');
    setGeneratedResult(null);
    setErrorMessage('');
    setCopied(false);
  }

  return (
    <div className="page-wrapper">
      <div className="container">
        {/* ========================================================= */}
        {/* Header Section                                            */}
        {/* ========================================================= */}
        <header className="app-header">
          <h1 className="app-title">Tiny GPT</h1>
        </header>

        {/* ========================================================= */}
        {/* Main Card: Prompt Input & Controls                       */}
        {/* ========================================================= */}
        <main className="main-card">
          <form onSubmit={handleGenerate}>
            {/* Prompt Input Header & Textarea */}
            <div className="form-group">
              <div className="label-row">
                <label htmlFor="prompt-input" className="form-label">
                  Text Prompt
                </label>
                <span
                  className={`char-counter ${
                    isPromptTooLong ? 'char-counter-error' : ''
                  }`}
                  aria-live="polite"
                >
                  {prompt.length} / {API_LIMITS.MAX_PROMPT_CHARS} characters
                </span>
              </div>

              <textarea
                id="prompt-input"
                className={`text-area ${isPromptTooLong ? 'input-error' : ''}`}
                rows={4}
                placeholder="The history of science began..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                disabled={isLoading}
                aria-describedby="prompt-help"
              />

            </div>

            {/* Controls Row */}
            <div className="controls-row">
              <div className="action-buttons single-button-row">
                {generatedResult && (
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={handleReset}
                    disabled={isLoading}
                  >
                    New Prompt
                  </button>
                )}
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={isLoading || isPromptEmpty || isPromptTooLong}
                >
                  {isLoading ? (
                    <span className="spinner-label">
                      <span className="spinner" aria-hidden="true"></span>
                      Generating...
                    </span>
                  ) : (
                    'Generate'
                  )}
                </button>
              </div>
            </div>
          </form>

          {/* ========================================================= */}
          {/* Error Alert Display                                       */}
          {/* ========================================================= */}
          {errorMessage && (
            <div className="alert alert-error" role="alert">
              <div className="alert-icon" aria-hidden="true">⚠️</div>
              <div className="alert-content">
                <strong>Request Failed</strong>
                <p>{errorMessage}</p>
              </div>
            </div>
          )}

          {/* ========================================================= */}
          {/* Output Display Panel                                      */}
          {/* ========================================================= */}
          {isLoading && (
            <div className="output-panel output-loading" aria-live="polite">
              <div className="loading-placeholder">
                <div className="loading-bar"></div>
                <p>Generating text token-by-token from the trained checkpoint...</p>
              </div>
            </div>
          )}

          {!isLoading && generatedResult && (
            <div className="output-panel" aria-live="polite">
              <div className="output-header">
                <div className="output-title-group">
                  <span className="output-title">Generated Output</span>
                  <span className="output-meta">
                    ({generatedResult.maxNewTokens} tokens requested)
                  </span>
                </div>
                <button
                  type="button"
                  className={`btn btn-copy ${copied ? 'btn-copied' : ''}`}
                  onClick={handleCopy}
                  title="Copy full text to clipboard"
                  aria-label="Copy generated text"
                >
                  {copied ? '✓ Copied' : 'Copy Text'}
                </button>
              </div>

              <div className="output-body">
                <pre className="output-text">{generatedResult.generatedText}</pre>
              </div>
            </div>
          )}
        </main>

      </div>
    </div>
  );
}

