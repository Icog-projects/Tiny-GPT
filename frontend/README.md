# Tiny GPT Frontend

A clean, responsive, single-page React application built with [Vite](https://vitejs.dev/) that connects to the Tiny GPT FastAPI backend for interactive text generation.

---

## 1. Prerequisites

- **Node.js**: v18.0.0 or later (verified compatible with Node v22+)
- **npm**: v9.0.0 or later
- **Tiny GPT Backend**: Running FastAPI server at `http://127.0.0.1:8000`

---

## 2. Setup & Installation

Navigate into the `frontend/` directory and install the dependencies:

```powershell
cd frontend
npm install
```

> **Note:** As per project guidelines, packages are not installed automatically. Run the command above once before launching the development server.

---

## 3. Configuration (`.env`)

The frontend communicates with the backend via the `VITE_API_BASE_URL` environment variable.

By default, if no `.env` file exists, it defaults to:

```text
http://127.0.0.1:8000
```

To configure a custom URL or port, copy `.env.example`:

```powershell
cp .env.example .env
```

And update the file if needed:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

---

## 4. Running the Complete System

Running Tiny GPT involves two processes: the FastAPI backend and the Vite frontend.

### Step 1: Start the Backend (Terminal 1)

From the project root (`icog-gpt-model`):

```powershell
cd C:\Users\hp\Documents\Icog-gpt-model\icog-gpt-model
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

The backend API will be available at:
- **API URL:** `http://127.0.0.1:8000`
- **Interactive Swagger Docs:** `http://127.0.0.1:8000/docs`

### Step 2: Start the Frontend (Terminal 2)

From the `frontend/` directory:

```powershell
cd C:\Users\hp\Documents\Icog-gpt-model\icog-gpt-model\frontend
npm run dev
```

The frontend application will be available at:
- **Local URL:** [http://127.0.0.1:5173](http://127.0.0.1:5173)

---

## 5. Using the Application

1. Open [http://127.0.0.1:5173](http://127.0.0.1:5173) in your browser.
2. Enter a text prompt (e.g., *"The history of science"* or click one of the sample prompt chips).
3. Set the desired number of new tokens (default: `50`, range: `1` to `500`).
4. Click **Generate Text**.
5. View the generated text in the output panel below.
6. Click **Copy Text** to copy the complete output to your clipboard, or **New Prompt** to clear and start a new prompt.

---

## 6. Project Structure

```text
frontend/
├── src/
│   ├── App.jsx          # One-page layout, prompt input, controls, loading & output states
│   ├── App.css          # Responsive styling (light palette, dark output, accessible contrast)
│   ├── main.jsx         # React 18 DOM entry point
│   └── api.js           # Fetch API client connecting strictly to POST /generate
├── index.html           # HTML template with Google Fonts (Inter & JetBrains Mono)
├── package.json         # React 18 & Vite dependencies
├── vite.config.js       # Vite configuration (port 5173 on 127.0.0.1)
├── .env.example         # Example environment configuration
└── README.md            # Frontend documentation
```

---

## 7. Troubleshooting

- **Error: "Cannot connect to the Tiny GPT backend at http://127.0.0.1:8000":**
  Ensure the FastAPI server is running in another terminal. Run:
  `.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload`
- **Error: "Model Unavailable (503)":**
  The server is running, but the checkpoint file is missing or failed to load. Ensure `checkpoints/tiny_gpt_wikitext2_60k_best.pt` exists in the repository.
- **Port 5173 is already in use:**
  Vite will automatically offer another port (e.g. `5174`). You can specify a different port in `vite.config.js` or with `npm run dev -- --port 3000`.

