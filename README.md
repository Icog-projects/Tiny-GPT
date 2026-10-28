# Tiny GPT: From ChatGPT to a Tiny Language Model

A compact, educational GPT-style language model built from scratch in Python and PyTorch for the iCog Labs assignment “From ChatGPT to a Tiny GPT: How Language Models Work.” This project is a small experimental model trained on WikiText-2, not a pretrained ChatGPT.

## Download Trained Model Checkpoints

Large checkpoint files are stored on Google Drive because large binary weights are not kept in this GitHub repository.

[Download Trained Model Checkpoints](https://drive.google.com/drive/folders/1iKBE4K8TjKUqmkDzmuIw5j9ZAXTKQ1ac?usp=sharing)

After downloading, place the main checkpoint at:

```text
checkpoints/tiny_gpt_wikitext2_60k_best.pt
```

<h2 align="center">Training and Validation Loss</h2>

<p align="center">
  <!-- Use this exact path or update the image src to match the uploaded filename -->
  <img
    src="docs/images/training_validation_loss.png"
    alt="Training and validation loss across training steps"
    width="800"
  />
</p>

## Project Overview and Features

- Small decoder-only GPT-style language model built from scratch.
- Trained on the Hugging Face WikiText-2 dataset (`wikitext-2-raw-v1`).
- Uses GPT-2 byte-pair encoding via `tiktoken` with a vocabulary size of 50,257.
- Includes training code, generation utilities, a FastAPI backend, and a small React + Vite frontend.
- Intended as an educational baseline and experimental project, not a production chatbot.

## Model Architecture

The code in `src/model/config.py` defines a compact decoder-only Transformer with:

- Vocabulary size: 50,257
- Context length: 64
- Model dimension (`d_model`): 64
- Attention heads: 4
- Transformer layers: 2
- Feed-forward dimension (`d_ff`): 256
- Dropout: 0.1
- Optimizer: `torch.optim.Adam`
- Approximate parameter count: 6.5M

This is a small GPT-style architecture for next-token prediction, not a pretrained ChatGPT model.

## Dataset and Tokenizer

- Dataset: Hugging Face `Salesforce/wikitext`
- Configuration: `wikitext-2-raw-v1`
- Tokenizer: GPT-2 encoding via `tiktoken`
- Data is organized into document-style chunks, and training uses shifted token targets for causal prediction.

## Training Results

The final reported run used a checkpoint resumed from an earlier stage and continued to training step 60,000 on WikiText-2.

<table align="center">
  <tr>
    <th>Dataset and split</th>
    <th>Training step</th>
    <th>Training loss</th>
    <th>Validation loss</th>
    <th>Training perplexity</th>
    <th>Validation perplexity</th>
    <th>Evaluation coverage</th>
  </tr>
  <tr>
    <td>WikiText-2 training split</td>
    <td>60,000</td>
    <td>4.5646</td>
    <td>—</td>
    <td>~96.0</td>
    <td>—</td>
    <td>Recorded training loss</td>
  </tr>
  <tr>
    <td>WikiText-2 validation split</td>
    <td>60,000</td>
    <td>—</td>
    <td>5.1610</td>
    <td>—</td>
    <td>~174.3</td>
    <td>10 validation batches; not a full validation-split evaluation</td>
  </tr>
</table>

The validation loss was measured over 10 validation batches as a practical estimate rather than a full validation-set pass. Perplexity is approximately exponential of the loss, so the reported values are consistent with the numbers above.

## Generated Text Examples

These are genuine examples produced by the small model using the saved checkpoint and greedy decoding.

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

The model tends to form short thematic associations and repetitive loops, which is expected for a small architecture and deterministic greedy decoding.

## Backend API

The backend exposes one custom endpoint:

- `POST http://127.0.0.1:8000/generate`
- Request body: `{"prompt": "The history of science", "max_new_tokens": 50}`
- Response body: `{"prompt": ..., "generated_text": ..., "max_new_tokens": ...}`
- Status codes: `200` on success, `422` for invalid input, and `503` when no valid checkpoint is available

The model is loaded from the `TINY_GPT_CHECKPOINT` environment variable on startup.

## Frontend

The frontend is a small React + Vite app in `frontend/`.

- Default API URL: `http://127.0.0.1:8000`
- Uses `frontend/.env.example` as the template for custom API settings
- Includes a prompt box, token count control, and generated output panel

## Essential Setup and Usage

### Backend

```powershell
cd C:\Users\hp\Documents\Icog-gpt-model\icog-gpt-model
pip install -r requirements.txt
pip install -r backend\requirements.txt
$env:TINY_GPT_CHECKPOINT = "checkpoints\tiny_gpt_wikitext2_60k_best.pt"
python -m uvicorn backend.main:app --reload
```

### Frontend

```powershell
cd C:\Users\hp\Documents\Icog-gpt-model\icog-gpt-model\frontend
npm install
npm run dev
```

### Standalone CLI

```powershell
python -m src.generation.generate --checkpoint checkpoints\tiny_gpt_wikitext2_60k_best.pt --prompt "The history of science" --max_new_tokens 30 --device cpu
```

### Tests

```powershell
pytest tests\test_model.py tests\test_data.py tests\test_training.py tests\test_generation.py -v
```

## Limitations and Future Improvements

- Model capacity is limited by the small architecture and short context window.
- Greedy decoding can lead to repetitive text loops.
- The project is best viewed as an educational baseline for understanding autoregressive Transformer behavior.
- Larger models, more context, and stronger decoding strategies would improve coherence and diversity.
