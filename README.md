# Tiny GPT: From ChatGPT to a Tiny Language Model

An educational implementation of a decoder-only, autoregressive GPT-style language model built from scratch using Python, PyTorch, and the WikiText-2 dataset for the iCog Labs assignment.

---

<h2 align="center">Project Screenshot</h2>

<p align="center">
  <!-- Replace docs/images/project-screenshot.png with your actual image path if different -->
  <img
    src="docs/images/project-screenshot.png"
    alt="Tiny GPT application interface"
    width="800"
  />
</p>

---

## 1. Project Overview & Features

### Purpose
This project demystifies how generative autoregressive models work by building the data pipeline, Transformer architecture, training loop, inference engine, backend, and frontend from scratch:
- **Built from scratch:** No pretrained GPT weights are loaded. All weights are initialized and trained directly.
- **Educational baseline:** Compact 6.5M parameter footprint designed to run on consumer hardware or Google Colab.
- **Realistic expectations:** Designed to learn basic English syntax and vocabulary associations from a small corpus; it is not a general conversational assistant like ChatGPT.

### Key Features
- **GPT-2 BPE Tokenizer:** 50,257-token vocabulary using `tiktoken`.
- **Causal Transformer Architecture:** Pre-LayerNorm decoder blocks with `torch.nn.MultiheadAttention` and strict causal masking.
- **Shifted Target Training:** Contiguous token sequences ($x_i \rightarrow x_{i+1}$) optimized with Adam (`torch.optim.Adam`).
- **REST API Backend:** Single-endpoint FastAPI server (`POST /generate`) with Pydantic validation.
- **Interactive Web UI:** Clean, responsive one-page React + Vite frontend.

---

## 2. Model Architecture & Specifications

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Vocabulary Size ($V$)** | 50,257 | GPT-2 Byte-Pair Encoding (`tiktoken`) |
| **Context Length ($T$)** | 32 | Maximum sequence token window |
| **Embedding Dimension ($d_{\text{model}}$)** | 64 | Hidden representation dimension |
| **Attention Heads ($H$)** | 4 | Causal self-attention heads ($d_k = 16$) |
| **Transformer Layers ($L$)** | 2 | Sequential decoder blocks |
| **Feed-Forward Dimension ($d_{\text{ff}}$)** | 256 | Position-wise MLP expansion ($4 \times d_{\text{model}}$) with GELU |
| **Dropout** | 0.1 | Regularization on embeddings and attention |
| **Total Parameters** | 6,535,040 | Untied embeddings and output head (~3.3M non-embedding) |
| **Optimizer** | Adam | Standard `torch.optim.Adam` ($\text{lr} = 3 \times 10^{-4}$) |

---

## 3. Dataset & Tokenizer

- **Dataset:** Hugging Face `Salesforce/wikitext` (`wikitext-2-raw-v1`).
- **Document Boundary Extraction:** Raw text rows are split into documents using top-level Wikipedia article headers (`= Title =`).
- **Target Shifting:** For each document, contiguous chunks of length $T + 1$ produce inputs $x = t_{0:T-1}$ and shifted targets $y = t_{1:T}$.
- **Boundary Isolation:** Special `<|endoftext|>` token terminates each article; token windows never cross document boundaries.

---

## 4. Training Results & Analysis

The model was resumed from an earlier checkpoint and trained up to step 60,000 on WikiText-2.

<h3 align="center">Training Results</h3>

<table align="center">
  <tr>
    <th>Metric</th>
    <th>Training</th>
    <th>Validation</th>
  </tr>
  <tr>
    <th>Dataset</th>
    <td>WikiText-2 training split</td>
    <td>WikiText-2 validation split</td>
  </tr>
  <tr>
    <th>Training Step</th>
    <td>60,000</td>
    <td>60,000</td>
  </tr>
  <tr>
    <th>Cross-Entropy Loss</th>
    <td>4.5646</td>
    <td>5.1610</td>
  </tr>
  <tr>
    <th>Perplexity</th>
    <td>Approximately 96.0</td>
    <td>Approximately 174.3</td>
  </tr>
  <tr>
    <th>Evaluation Coverage</th>
    <td>Recorded training loss</td>
    <td>10 validation batches; not a full-split evaluation</td>
  </tr>
</table>

<br/>

> **Note on Evaluation:** Periodic validation loss was measured over 10 validation batches as a computational estimate rather than an exhaustive full-split evaluation. Perplexity is calculated as $P = e^{\text{loss}}$ ($e^{4.5646} \approx 96.0$, $e^{5.1610} \approx 174.3$).

### Training vs. Validation Loss Explained
- **Training loss** measures the average prediction error on batches seen during optimization.
- **Validation loss** measures prediction error on held-out text without parameter updates (`model.eval()`).
- Lower loss values reflect higher confidence on true next-token predictions.
- **Overfitting diagnostics:** When training loss decreases steadily while validation loss plateaus or rises, the network is memorizing training phrases rather than generalizing. Here, a training loss of 4.5646 and validation loss of 5.1610 indicate a modest, expected generalization gap.

---

## 5. Genuine Generated Text Samples

The following completions were generated directly by loading `checkpoints/tiny_gpt_wikitext2_60k_best.pt` using greedy decoding (`argmax`):

```text
Prompt: "The history of science"
Output:
The history of science fiction and the film 's music video .
= = = = = =
The film was released in the album , and the album 's album , and was released
```

```text
Prompt: "In the early twentieth century,"
Output:
In the early twentieth century,@ century , the church was the first church of the church , and the church was built in the city of the church . The church was built in the city of the church
```

```text
Prompt: "The exploration of space"
Output:
The exploration of space .
= = = = = =
The nucleus is a protein of the nucleus , and the nucleus is a protein of the nucleus , and is a protein of the nucleus
```

*Observations:* The model learns thematic associations (*science* $\rightarrow$ *science fiction*; *twentieth century* $\rightarrow$ *church, built, city*). Repetitive loops and `= = = = = =` Wikipedia header artifacts occur naturally due to compact capacity (6.5M parameters), small context length ($T=32$), and deterministic greedy decoding.

*(Note on loss graphs: Loss history metrics are saved in `checkpoints/tiny_gpt_wikitext2_60k_history.json`. Static plot images are omitted to keep repository size minimal.)*

---

## 6. Repository Structure

```text
icog-gpt-model/
├── src/
│   ├── data/
│   │   ├── tokenizer.py      # tiktoken BPE wrapper (GPT-2 vocabulary)
│   │   └── dataset.py        # Document parsing, chunking, and shifted targets (x, y)
│   ├── model/
│   │   ├── config.py         # GPTConfig dataclass and validation
│   │   ├── attention.py      # Causal multi-head self-attention module
│   │   ├── feed_forward.py   # Position-wise MLP network (GELU)
│   │   ├── block.py          # Pre-LN Transformer decoder block
│   │   └── gpt.py            # Complete GPT language model
│   ├── training/
│   │   └── trainer.py        # Cross-entropy loss, Adam loop, and checkpointing
│   └── generation/
│       └── generate.py       # Autoregressive greedy decoding CLI
├── tests/
│   ├── test_data.py          # Tokenizer roundtrip, document parsing, and window tests
│   ├── test_model.py         # Forward pass shapes, causal mask invariance, backprop tests
│   ├── test_training.py      # Loss computation and checkpoint roundtrip tests
│   ├── test_generation.py    # Autoregressive decoding and backend service tests
│   └── test_backend.py       # FastAPI TestClient endpoint validation tests
├── checkpoints/
│   ├── .gitkeep              # Git tracking placeholder (ignores binary .pt weights)
│   ├── tiny_gpt.pt           # Verification checkpoint
│   ├── wikitext2_smoke.pt    # CPU smoke-test checkpoint
│   └── tiny_gpt_wikitext2_60k_best.pt  # 60k-step trained checkpoint from Kaggle
├── backend/
│   ├── main.py               # FastAPI server with single POST /generate endpoint
│   ├── schemas.py            # Pydantic request (prompt, max_new_tokens) & response schemas
│   ├── service.py            # Model loader, device selection, and inference service
│   └── requirements.txt      # FastAPI, Uvicorn, Pydantic, and HTTPX dependencies
├── frontend/
│   ├── package.json          # React 18 and Vite dependencies
│   ├── vite.config.js        # Vite dev server configuration (127.0.0.1:5173)
│   ├── index.html            # Web entry point with Inter and JetBrains Mono fonts
│   ├── .env.example          # API URL configuration template
│   ├── README.md             # Frontend installation and usage guide
│   └── src/
│       ├── main.jsx          # React DOM entry point
│       ├── App.jsx           # Single-page interface (prompt, controls, output)
│       ├── App.css           # Responsive stylesheet (light/indigo palette)
│       └── api.js            # Fetch client for POST /generate
├── requirements.txt          # Core PyTorch and model dependencies
├── .gitignore                # Excludes virtual environments, caches, and weights
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

---

## 7. Backend API (`POST /generate`)

The backend exposes **strictly one custom endpoint**:

- **Endpoint:** `POST http://127.0.0.1:8000/generate`
- **Request Body:**
  ```json
  {
    "prompt": "The history of science",
    "max_new_tokens": 50
  }
  ```
- **Response Body:**
  ```json
  {
    "prompt": "The history of science",
    "generated_text": "The history of science fiction and the film...",
    "max_new_tokens": 50
  }
  ```
- **Status Codes:**
  - `200 OK`: Successful text generation.
  - `422 Unprocessable Entity`: Invalid parameters (e.g. `max_new_tokens` outside 1–500 or prompt exceeding 2,000 characters).
  - `503 Service Unavailable`: Model checkpoint missing or unavailable.

---

## 8. Frontend Interface

The frontend (`frontend/`) is a single-page React application built with Vite:
- **Prompt Input:** Multiline text area with character counter (max 2,000) and sample prompt chips.
- **Controls:** Token count selector (1–500, default 50) and disabled button states during active generation.
- **Output Panel:** Terminal-style panel preserving whitespace with one-click copy to clipboard.
- **Configuration:** Reads `VITE_API_BASE_URL` with a sensible default of `http://127.0.0.1:8000`.

---

## 9. Installation & Running Instructions

### Prerequisites
- Python 3.11–3.13 with virtual environment support.
- Node.js v18+ and npm (for the web frontend).
- Checkpoint `checkpoints/tiny_gpt_wikitext2_60k_best.pt` in place.

### A. Backend Setup & Run
```powershell
# From project root:
cd C:\Users\hp\Documents\Icog-gpt-model\icog-gpt-model

# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install backend dependencies (run once)
pip install "fastapi==0.115.5" "uvicorn==0.32.1" "pydantic>=2.9.0" "httpx>=0.27.0"

# 3. Start the FastAPI server
python -m uvicorn backend.main:app --reload
```
*API running at: `http://127.0.0.1:8000` (Swagger docs at `/docs`)*

### B. Frontend Setup & Run
```powershell
# In a second terminal:
cd C:\Users\hp\Documents\Icog-gpt-model\icog-gpt-model\frontend

# 1. Install npm dependencies (run once)
npm install

# 2. Start Vite development server
npm run dev
```
*Web application running at: `http://127.0.0.1:5173`*

### C. Standalone Inference CLI (No server needed)
```powershell
python -m src.generation.generate --checkpoint checkpoints\tiny_gpt_wikitext2_60k_best.pt --prompt "The history of science" --max_new_tokens 30 --device cpu
```

### D. Running Offline Unit Tests (56 tests)
```powershell
pytest tests\test_model.py tests\test_data.py tests\test_training.py tests\test_generation.py -v
```

---


## 10. Limitations & Future Improvements

### Current Limitations
1. **Model Capacity:** With 2 layers, 4 heads, and $d_{\text{model}}=64$, the model captures local word co-occurrences but cannot perform reasoning or maintain long-range coherence.
2. **Greedy Decoding Loops:** Greedy sampling ($\text{argmax}$) combined with a small context window ($T=32$) easily falls into repetitive text loops.
3. **Untied Embeddings:** With a 50,257 vocabulary, embedding and output head tables account for $\approx 6.43\text{M}$ of the 6.53M total parameters.

### Future Improvements
1. **Weight Tying:** Reusing input embedding weights for the output projection (`lm_head.weight = wte.weight`) to save $\approx 3.22\text{M}$ parameters.
2. **Stochastic Sampling:** Implementing temperature scaling, Top-$k$, and nucleus (Top-$p$) sampling to reduce repetition and improve variety.
3. **Larger Context Window:** Scaling context length from 32 to 128+ tokens for wider attention horizons.
