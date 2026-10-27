# From ChatGPT to a Tiny GPT: How Language Models Work

An educational project implementing a small decoder-only, autoregressive GPT-style language model from scratch using Python and PyTorch for the iCog Labs assignment.

---

## 1. Project Purpose & Learning Goals

The primary goal of this project is to demystify generative autoregressive language models (such as GPT-2 and ChatGPT) by constructing the underlying architecture, data pipeline, training loop, and inference engine from scratch:

- **Built From Scratch with Random Initialization:** We do not load pretrained GPT weights (such as standard OpenAI GPT-2 weights). The Transformer architecture is implemented directly in PyTorch, initialized randomly, and trained on a selected text corpus.
- **Conceptual Grounding:** Demonstrates hands-on understanding of tokenization, positional embeddings, causal multi-head self-attention, residual connections, layer normalization, autoregressive next-token prediction, cross-entropy loss, checkpointing, and model serving.
- **Realistic Output Expectations:** This model is designed for educational purposes with a compact parameter footprint. Because it is trained from scratch with randomly initialized weights on a modest dataset and compute budget, it will learn basic linguistic patterns and syntax, but it will **not** produce ChatGPT-level fluency or general reasoning. Output quality depends directly on dataset size, model capacity, training duration, and available compute.

---

## 2. Planned Model Architecture

The planned architecture is a small decoder-only Transformer based on the GPT design:

```mermaid
flowchart TD
    InputTokens["Input Token IDs (batch_size, context_length)"] --> TokenEmbed["Token Embedding (vocab_size -> d_model)"]
    InputPositions["Position Indices (0 to context_length - 1)"] --> PosEmbed["Positional Embedding (context_length -> d_model)"]
    TokenEmbed & PosEmbed --> AddEmbed["Elementwise Sum (batch_size, context_length, d_model)"]

    subgraph TransformerBlock["Transformer Block (repeated N times)"]
        AddEmbed --> LN1["LayerNorm 1"]
        LN1 --> CausalAttn["Causal Multi-Head Self-Attention (torch.nn.MultiheadAttention)"]
        CausalAttn --> Residual1["Residual Add (+)"]
        AddEmbed --> Residual1
        Residual1 --> LN2["LayerNorm 2"]
        LN2 --> FFN["Feed-Forward Network (Linear -> GELU -> Linear)"]
        FFN --> Residual2["Residual Add (+)"]
        Residual1 --> Residual2
    end

    Residual2 --> FinalLN["Final LayerNorm"]
    FinalLN --> LMHead["Linear Language Model Head (d_model -> vocab_size)"]
    LMHead --> Logits["Logits (batch_size, context_length, vocab_size)"]
```

### Key Architectural Components

1. **Token Embeddings:** Maps input token IDs from the GPT-2 vocabulary ($V = 50,257$) to a dense embedding space of dimension $d_{\text{model}}$ (e.g., 64). This requires $V \times d_{\text{model}} = 50,257 \times 64 \approx 3.22\text{M}$ learnable parameters.
2. **Positional Embeddings:** Learnable positional vectors representing token positions up to the maximum context length $T$ (e.g., 64), contributing $T \times d_{\text{model}} = 64 \times 64 = 4,096$ parameters.
3. **Causal Multi-Head Self-Attention:** Implemented using PyTorch's `torch.nn.MultiheadAttention` (`batch_first=True`) coupled with a strict upper-triangular causal mask (`attn_mask`). The causal mask ensures position $i$ can only attend to positions $j \le i$, preventing tokens from attending to subsequent tokens.
4. **Position-wise Feed-Forward Network (FFN / MLP):** Expands the hidden dimension to $4 \times d_{\text{model}}$, applies a non-linear activation (GELU), and projects back to $d_{\text{model}}$ ($\approx 33\text{k}$ parameters per block).
5. **Pre-Layer Normalization & Residual Connections:** Uses a pre-LN design (`x = x + Sublayer(LayerNorm(x))`) for numerical stability and effective gradient propagation through layers.
6. **Language Model Output Head:** Linear projection from $d_{\text{model}}$ to $V$ ($50,257$), producing raw logits of shape `(batch_size, context_length, vocab_size)`. In an untied model, this projection adds another $d_{\text{model}} \times V = 64 \times 50,257 \approx 3.22\text{M}$ parameters.

### Next-Token Target Shifting

During training, autoregressive language models predict the immediate next token for every position simultaneously in a single forward pass:
- Given a raw tokenized sequence of length $T + 1$: $(t_0, t_1, t_2, \dots, t_T)$
- **Input sequence ($x$):** $(t_0, t_1, \dots, t_{T-1})$ with shape `(batch_size, context_length)`
- **Target sequence ($y$):** $(t_1, t_2, \dots, t_T)$ with shape `(batch_size, context_length)`
- At position $i$, the model receives $x_i = t_i$ (with causal visibility of preceding tokens $t_0, \dots, t_i$) and calculates the cross-entropy loss against target $y_i = t_{i+1}$.

### Parameter Distribution & Weight-Tying Decision

- **Untied Baseline (Assignment Starting Point):** The token embedding table ($\approx 3.22\text{M}$) and LM output head ($\approx 3.22\text{M}$) together account for $\approx 6.43\text{M}$ parameters, dominating a 1-block Transformer body ($\approx 50\text{k}$ parameters).
- **Weight Tying (Optional Enhancement):** Weight tying reuses the input embedding weights for the output projection (`lm_head.weight = token_embeddings.weight`). While weight tying cuts total parameters almost in half and stabilizes training, we strictly implement the untied architecture first to fulfill the core baseline requirements before exploring weight tying as a future optimization.


---

## 3. Technology Stack & Design Decisions

| Category | Technology | Rationale | Status |
| :--- | :--- | :--- | :--- |
| **Language** | Python | Standard language for modern deep learning and PyTorch development. | Configured |
| **Deep Learning** | PyTorch (`torch`) | Tensor manipulation, autograd, `nn.Module` abstractions, and Adam optimizer. | Dependency specified |
| **Tokenizer** | `tiktoken` | Implements OpenAI's GPT-2 Byte-Pair Encoding (BPE) with 50,257 tokens, as required by Task 1. | Dependency specified |
| **Dataset Loading** | Hugging Face `datasets` | Standardized access to text corpora without custom scraping. | Dependency specified |
| **Testing** | `pytest` | Automated assertions for token shift invariants, tensor shapes, and causal masking. | Dependency specified |
| **Backend API** | FastAPI + Uvicorn | High-performance asynchronous REST API for serving model text generation. | Planned (Optional Bonus) |
| **Schema Validation** | Pydantic | Typed request and response models for the generation endpoint. | Planned (Optional Bonus) |
| **Frontend UI** | React + Vite | Clean single-page chat interface built with JavaScript, JSX, and CSS. | Planned (Optional Bonus) |

---

## 4. Target Repository Structure

The following tree represents the **intended final project architecture**. Files will be created and verified progressively across each assignment task. Currently, only the initial setup files and package initializers have been created.

```text
icog-gpt-model/
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ __init__.py           # [Implemented] Source package root
â”‚   â”œâ”€â”€ data/
â”‚   â”‚   â”œâ”€â”€ __init__.py       # [Implemented] Data package initializer
â”‚   â”‚   â”œâ”€â”€ tokenizer.py      # [Implemented] tiktoken wrapper, encode/decode assertions
â”‚   â”‚   â””â”€â”€ dataset.py        # [Implemented] Dataset, chunking, shifted targets (x, y)
â”‚   â”œâ”€â”€ model/
â”‚   â”‚   â”œâ”€â”€ __init__.py       # [Implemented] Model package initializer
â”‚   â”‚   â”œâ”€â”€ config.py         # [Implemented] GPTConfig dataclass (hyperparameters)
â”‚   â”‚   â”œâ”€â”€ attention.py      # [Implemented] Causal self-attention layer
â”‚   â”‚   â”œâ”€â”€ feed_forward.py   # [Implemented] Feed-forward MLP network
â”‚   â”‚   â”œâ”€â”€ block.py          # [Implemented] Transformer decoder block
â”‚   â”‚   â””â”€â”€ gpt.py            # [Implemented] Full GPT model with LM head
â”‚   â”œâ”€â”€ training/
â”‚   â”‚   â”œâ”€â”€ __init__.py       # [Implemented] Training package initializer
â”‚   â”‚   â””â”€â”€ trainer.py        # [Implemented] Training loop, loss tracking, checkpoint saving
â”‚   â””â”€â”€ generation/
â”‚       â”œâ”€â”€ __init__.py       # [Implemented] Generation package initializer
â”‚       â””â”€â”€ generate.py       # [Implemented] Autoregressive decoding and checkpoint loader
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ __init__.py           # [Implemented] Test package initializer
â”‚   â”œâ”€â”€ test_data.py          # [Implemented] Tests for tokenization, target shifts, and shapes
â”‚   â”œâ”€â”€ test_model.py         # [Implemented] Tests for forward pass shapes and causal mask
â”‚   â”œâ”€â”€ test_training.py       # [Implemented] Tests for loss calculation, optimization, and evaluation
â”‚   â””â”€â”€ test_generation.py    # [Implemented] Tests for autoregressive greedy decoding and checkpoint load
â”œâ”€â”€ checkpoints/
â”‚   â”œâ”€â”€ .gitkeep              # [Implemented] Keeps folder in git; binary weights are ignored
â”‚   â””â”€â”€ tiny_gpt.pt           # [Generated] Trained checkpoint from verification run
â”œâ”€â”€ backend/
â”‚   â”œâ”€â”€ __init__.py           # [Implemented] Backend package initializer
â”‚   â”œâ”€â”€ requirements.txt      # [Implemented] FastAPI & Uvicorn dependencies
â”‚   â”œâ”€â”€ main.py               # [Planned - Optional] FastAPI application
â”‚   â””â”€â”€ schemas.py            # [Planned - Optional] Pydantic request/response schemas
â”œâ”€â”€ frontend/
â”‚   â”œâ”€â”€ package.json          # [Planned - Optional] React + Vite dependencies
â”‚   â”œâ”€â”€ index.html            # [Planned - Optional] Web entry point
â”‚   â””â”€â”€ src/
â”‚       â”œâ”€â”€ App.jsx           # [Planned - Optional] Chat interface component
â”‚       â”œâ”€â”€ main.jsx          # [Planned - Optional] React DOM entry point
â”‚       â””â”€â”€ App.css           # [Planned - Optional] Chat styling
â”œâ”€â”€ README.md                 # [Implemented] Project documentation
â”œâ”€â”€ requirements.txt          # [Implemented] Core dependencies for model, data, and training
â”œâ”€â”€ .gitignore                # [Implemented] Excludes caches, venv, and large checkpoints
â””â”€â”€ LICENSE                   # [Implemented] MIT License
```

---

## 5. Development Environment Setup

Local development and tests are intended to run on CPU on your local computer. GPU compute is reserved for Google Colab training.

### Prerequisites
- **Recommended Python:** Python 3.11 or 3.12 is strongly recommended for this project. While Python 3.13 is supported in recent PyTorch releases (2.5+), Python 3.11 and 3.12 offer the most mature, reliable pre-compiled binary wheel support across data dependencies (`datasets`, `pyarrow`, and `tiktoken`) on Windows.
- **Git** installed.

### Step 1: Create a Virtual Environment (Windows PowerShell)

In your VS Code terminal, navigate to the repository directory and create a virtual environment:

```powershell
# Navigate into the repository directory
cd C:\Users\hp\Documents\Icog-gpt-model\icog-gpt-model

# Create a virtual environment named .venv
python -m venv .venv
```

### Step 2: Activate the Virtual Environment

```powershell
# Activate in PowerShell
.\.venv\Scripts\Activate.ps1

# (Note: If PowerShell displays an execution policy restriction, run once:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser)
```

### Step 3: Install Core Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Environment Installation

Execute this Python one-liner to verify that the core dependencies and tokenizer are accessible:

```powershell
python -c "import torch, tiktoken, datasets; print('PyTorch:', torch.__version__); print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu'); enc = tiktoken.get_encoding('gpt2'); print('Vocab size:', enc.n_vocab)"
```

**Expected output:**
- PyTorch version displayed (e.g. `2.5.x` or `2.6.x`).
- Device: `cpu`.
- Vocab size: `50257`.

---

## 6. How Tests, Training, and Inference Run

Tasks 1, 2, and 3 are fully implemented, verified, and accompanied by offline automated tests.

### Running the Tokenizer Demonstration (5 Sentences)
To run the assignment-required demonstration of the GPT-2 tokenizer with five example sentences, token IDs, decoded text, and an explanation of token IDs as categorical vocabulary indices:

```powershell
python -m src.data.tokenizer
```

---

### Running the Test Suite (49 Tests)
All 49 unit tests in `tests/` run completely offline without downloading datasets or network access:

```powershell
pytest -v
```

**Verification result:**
```text
============================= 49 passed in 10.98s ==============================
- tests/test_data.py: 20 passed (5-sentence demo, tokenization roundtrip, WikiText header parsing, (x, y) target shift, EOT placement, split isolation)
- tests/test_model.py: 14 passed (config bounds, initialization, torch.nn.MultiheadAttention forward shapes, causal invariance, backpropagation)
- tests/test_training.py: 9 passed (Adam optimizer, loss reshaping, gradient updates, evaluation mode, parameter non-mutation in eval, checkpoint roundtrip)
- tests/test_generation.py: 6 passed (greedy decoding, context-length cropping, empty prompts, standalone inference)
```

---

### Running the Real-Data WikiText-2 Smoke Test (CPU)
A fast, real-data smoke test verifying the data loader, `torch.nn.MultiheadAttention` architecture, and `torch.optim.Adam` optimizer on actual WikiText-2 data (`Salesforce/wikitext`, `wikitext-2-raw-v1`) without starting full training:

```powershell
python -m src.training.trainer --epochs 1 --batch_size 2 --context_length 32 --device cpu --eval_interval 5 --max_train_steps 15 --max_val_batches 10 --checkpoint_path checkpoints/wikitext2_smoke.pt
```

**Actual Smoke-Test Output on Real WikiText-2:**
```text
Loading WikiText-2 dataset...
Extracting and tokenizing train documents...
Extracting and tokenizing validation documents...
Loaded 74,355 training samples and 7,689 validation samples.
--- Starting Training on CPU ---
Model parameters: 6,535,040 (non-embedding: 3,316,544)
Configuration: layers=2, heads=4, d_model=64, context=32
Epoch 1/1 | Step 5 | Train Loss: 10.7878 | Perplexity: 48427.25
  --> [IMPROVEMENT] Val Loss: 10.7843 | Checkpoint saved: checkpoints/wikitext2_smoke.pt
Epoch 1/1 | Step 10 | Train Loss: 10.7666 | Perplexity: 47409.54
  --> [IMPROVEMENT] Val Loss: 10.7163 | Checkpoint saved: checkpoints/wikitext2_smoke.pt
Epoch 1/1 | Step 15 | Train Loss: 10.6820 | Perplexity: 43562.65
  --> [IMPROVEMENT] Val Loss: 10.6388 | Checkpoint saved: checkpoints/wikitext2_smoke.pt
--- Training Finished in 1.70s across 15 steps ---
```

---

### Standalone Inference from Checkpoint (In a Separate Process)
You can generate text in a completely separate Python process by loading the smoke test checkpoint without retraining:

```powershell
python -m src.generation.generate --checkpoint checkpoints/wikitext2_smoke.pt --prompt "The history of" --max_new_tokens 20 --device cpu
```

**Actual Output from 15-step Real WikiText-2 Checkpoint:**
```text
Loading checkpoint from 'checkpoints/wikitext2_smoke.pt' onto CPU...
Model loaded: 6,535,040 parameters (layer=2, heads=4, d_model=64).
Checkpoint metadata: epoch=1, step=15, val_loss=10.6388

--- Generating Text ---
Prompt: 'The history of'
Requested new tokens: 20
----------------------------------------
Result:
The history of the the the the the the the the the the and the the the the the the the the the
----------------------------------------
```
*(Note: At step 15, the model has only barely begun optimizing away from random uniform initialization, with loss dropping from ~10.82 toward 10.63, naturally starting to predict the most frequent English corpus tokens "the" and "and".)*

---

### Prepared Full Training Command (Do Not Run Automatically)
The exact, verified command for running full training on the complete WikiText-2 dataset:

**On Local CPU (Estimated 2-4 hours):**
```powershell
python -m src.training.trainer --epochs 3 --batch_size 16 --context_length 64 --device cpu --eval_interval 200 --checkpoint_path checkpoints/wikitext2_full.pt
```

**On Google Colab GPU (Estimated 5-10 minutes on T4):**
```bash
python -m src.training.trainer --epochs 5 --batch_size 32 --context_length 64 --device cuda --eval_interval 200 --checkpoint_path checkpoints/wikitext2_full.pt
```

---

## 7. Task 3 Implementation Details

Task 3 brings the data pipeline (Task 1) and decoder-only Transformer architecture (Task 2) together into an end-to-end training and inference system.

### A. Training Loop & Cross-Entropy Loss
- **Loss Computation (`compute_loss`):**
  Given model logits of shape `(B, T, vocab_size)` and targets `(B, T)`, the tensors are reshaped to `(B * T, vocab_size)` and `(B * T,)` respectively before passing into `torch.nn.functional.cross_entropy`. This calculates next-token prediction loss simultaneously across all sequence positions.
- **Optimization Step (`train_one_step`):**
  1. `model.train()` activates dropout.
  2. `optimizer.zero_grad()` clears old gradients.
  3. `logits = model(x)` executes the forward pass.
  4. `loss = compute_loss(logits, y)` computes the cross-entropy loss.
  5. `loss.backward()` calculates gradients via autograd.
  6. `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` clips gradient norms to prevent exploding gradients.
  7. `optimizer.step()` updates trainable parameters using `torch.optim.Adam` (learning rate $10^{-3}$).
- **Random Initialization:** Training begins strictly from the model's random Gaussian initialization ($\mathcal{N}(0, 0.02^2)$). No pretrained GPT-2 weights are loaded.

### B. Validation Process & Overfitting Detection
- **Validation Evaluation (`evaluate`):**
  - Switches the model to evaluation mode (`model.eval()`), disabling dropout.
  - Wraps the loop in `with torch.no_grad():` to disable PyTorch's autograd engine and avoid accumulating computational graphs in RAM.
  - Iterates through validation batches without updating model weights or accumulating gradients.
  - Restores `model.train()` mode before returning control to the training loop.
- **Training Loss vs. Validation Loss:**
  - **Training Loss:** Measures how well the model predicts next tokens on data it is currently optimizing on.
  - **Validation Loss:** Measures how well the model predicts next tokens on unseen held-out text.
  - **Overfitting Diagnostics:**
    - *Underfitting:* Both training loss and validation loss remain high and decrease together slowly.
    - *Healthy Convergence:* Training loss decreases steadily, and validation loss decreases alongside it.
    - *Overfitting:* Training loss continues to drop toward zero, while validation loss plateaus and begins to rise. This indicates the model is memorizing training sequences rather than learning general linguistic syntax.
- **Selective Checkpointing:** The trainer tracks `best_val_loss`. Only validation improvements trigger checkpoint writes, ensuring a worse checkpoint never overwrites a better one.

### C. Autoregressive Text Generation (`generate_text`)
- **Greedy Decoding:**
  1. The user's prompt string is encoded into token IDs using `GPT2Tokenizer.encode()`. If the prompt is empty or whitespace, generation starts unconditionally from the `<|endoftext|>` token (`50256`).
  2. In evaluation mode (`model.eval()`) under `torch.no_grad()`, the sequence is fed to `model(idx)`.
  3. If the sequence length exceeds `context_length`, it is dynamically cropped to the most recent `context_length` tokens (`idx[:, -context_length:]`).
  4. The logits at the final position `logits[:, -1, :]` are extracted, and greedy selection picks the token with the highest predicted score: `next_token = torch.argmax(logits[:, -1, :], dim=-1)`.
  5. The chosen token is appended to the sequence and the process repeats until `max_new_tokens` are produced.
  6. The complete token list is decoded back into human-readable text via `GPT2Tokenizer.decode()`.

---

## 8. Verification Run Results & Generated Samples

### Four Distinct Tiers of Verification
1. **Offline Unit Tests (49 tests):** Validates architecture components, causal invariance, shape dimensions, tokenizer invariants, evaluation mode snapshots, and checkpoint roundtrips without network access.
2. **Synthetic Data Verification (`--verification`):** Runs 10 synthetic steps to test edge-to-edge trainer mechanisms in under 1 second.
3. **Real-Data WikiText-2 Smoke Test (`wikitext2_smoke.pt`):** Verified on actual `Salesforce/wikitext` raw training and validation data for 15 steps on CPU, confirming dataset processing, loss calculation, and independent inference.
4. **Full Training (Pending Approval):** Multi-epoch training on the entire 74,355-sample WikiText-2 corpus.

---

## 9. Google Colab GPU Training Workflow (Planned for Full Corpus)

For training the full WikiText-2 dataset across multiple epochs, GPU acceleration (such as Google Colab T4) is recommended:

```bash
# In Google Colab:
!git clone https://github.com/Icog-projects/icog-gpt-model.git
%cd icog-gpt-model
!pip install -r requirements.txt
!python -m src.training.trainer --epochs 5 --batch_size 32 --context_length 64 --device cuda --checkpoint_path /content/drive/MyDrive/icog_checkpoints/tiny_gpt.pt
```

---

## 10. Assignment Implementation Roadmap

| Task | Description | Status | Evidence / Notes |
| :--- | :--- | :--- | :--- |
| **Phase 1: Project Setup** | Repository structure, package initialization, dependencies, `.gitignore`, documentation. | **Implemented** | Package `__init__.py` files, `requirements.txt`, `.gitignore`, root `README.md` verified. |
| **Task 1: Tokenization & Dataset** | `tiktoken` GPT-2 encoding, Hugging Face dataset loading, context sequences, shifted targets ($x_{1:T} \rightarrow y_{2:T+1}$), 5-sentence demo. | **Implemented** | `src/data/tokenizer.py`, `src/data/dataset.py`, 20 tests in `tests/test_data.py`. |
| **Task 2: GPT Architecture** | Decoder-only Transformer, token/pos embeddings, `torch.nn.MultiheadAttention`, residual connections, LayerNorm, LM head, shape tests. | **Implemented** | `src/model/`, 14 tests in `tests/test_model.py`. |
| **Task 3: Training & Generation** | Cross-entropy loss, `torch.optim.Adam` optimizer, training loop, validation tracking, autoregressive token generation, checkpoint saving. | **Implemented** | `src/training/trainer.py`, `src/generation/generate.py`, 15 tests in `tests/test_training.py` & `tests/test_generation.py`. |
| **Real WikiText-2 Smoke Test** | CPU smoke run on actual WikiText-2 data, loss logging, checkpoint saving. | **Implemented** | `checkpoints/wikitext2_smoke.pt` created and validated on CPU. |
| **Standalone Inference** | Loading saved checkpoint and generating text in a separate process without retraining. | **Implemented** | `python -m src.generation.generate` CLI verified in independent process. |
| **Colab GPU Training** | Cloud training on full WikiText-2 GPU, loss logging, checkpoint export to Google Drive. | **Planned** | Runnable on Google Colab using existing code. |
| **Task 4: Bonus Backend** | FastAPI server loading checkpoint at startup and exposing `POST /generate`. | **Optional / Planned** | Dependencies specified in `backend/requirements.txt`. |
| **Task 5: Bonus Frontend** | React + Vite single-page chat UI communicating with FastAPI. | **Optional / Planned** | Scaffolding in `frontend/`. |

---

## 11. Known Limitations & Planned Future Improvements

### Limitations of the Dataset Parsing Heuristic & Model
1. **Wikipedia Header Detection Heuristic:** The current document splitting heuristic identifies lines matching ` = ... = ` as article headers. While this captures top-level Wikipedia articles in WikiText-2, it can also match intermediate section headers (such as ` == Early Life == `). Sub-sections within an article may thus be treated as separate documents. However, all tokens within each extracted document are strictly windowed, and `<|endoftext|>` is appended to terminate documents safely.
2. **Vocabulary Domination in Untied Setup:** With a 50,257-token vocabulary and $d_{\text{model}} = 64$, the embedding table and untied output head account for $\approx 6.43\text{M}$ parameters, while the 2-block Transformer body contains only $\approx 100\text{k}$ parameters.
3. **Untrained / Short-Trained Output:** Because the model starts from random initialization, a 15-step smoke test only verifies data loading, forward propagation, loss calculation, backpropagation, and sampling mechanics. Producing coherent English sentences requires full multi-epoch training.
4. **Greedy Repetition:** Greedy decoding always chooses $\text{argmax}$. On small or early-stage models, this leads to repetitive loops or token degeneration.

### Planned Future Improvements
1. **Weight Tying:** Sharing the weights between the input embedding matrix and the output language model head (`lm_head.weight = wte.weight`), reducing total parameters by $\approx 3.22\text{M}$ and improving generalization.
2. **Stochastic Sampling Strategies:** Adding temperature scaling, Top-$k$ sampling, and nucleus (Top-$p$) sampling to produce diverse and natural text completions.
3. **Learning Rate Schedules:** Implementing linear warmup followed by cosine learning rate decay to stabilize optimization across thousands of steps.
4. **FastAPI & React Chat Serving:** Completing the optional backend and frontend to deploy the trained model as an interactive web chat application.
