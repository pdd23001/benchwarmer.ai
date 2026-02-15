# 🪑 benchwarmer.ai

**Benchmarking should not be a bottleneck of innovation.**

benchwarmer.ai automates the painful workflow of algorithm benchmarking. Upload your algorithm and the research papers you want to compete against — our multi-agent framework extracts algorithms from the papers, generates runnable implementations, executes everything in sandboxed environments, and produces comparison charts. What used to take days now takes minutes.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![React](https://img.shields.io/badge/react-19-61DAFB)
![Vite](https://img.shields.io/badge/vite-7-646CFF)
![FastAPI](https://img.shields.io/badge/fastapi-0.109-009688)

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the App](#running-the-app)
- [API Reference](#api-reference)
- [Environment Variables](#environment-variables)
- [Contributing](#contributing)
- [License](#license)

---

## How It Works

1. **Upload** — Drop in your `.py` algorithm and the research papers you want to benchmark against.
2. **Intake** — An AI agent (Claude or Nemotron) parses your description and PDFs, classifies the problem, and builds a structured benchmark configuration.
3. **Implementation** — Claude generates runnable Python implementations of each challenger algorithm extracted from the papers, then smoke-tests them before they proceed.
4. **Execution** — All algorithms run in parallel inside isolated sandboxes (local subprocesses or Modal cloud sandboxes). One crash doesn't take down the benchmark.
5. **Analysis** — Results are aggregated into a DataFrame and an AI-powered plot agent generates comparison charts on demand.
6. **Conversation** — The entire flow is driven through a multi-turn chat interface. Ask follow-up questions, tweak parameters, re-run with different instances — all in natural language.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Frontend (Vite + React)                   │
│  ChatPage ──► SSE stream ◄── /api/chat ──► OrchestratorAgent    │
│  Sidebar  ──► REST       ◄── /api/sessions, /api/algorithms     │
└──────────────────────────────────────────────────────────────────┘
                              │
                    FastAPI (uvicorn :8000)
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                      ▼
  IntakeAgent          ImplementationAgent       PlotAgent
  (problem config)     (code generation)         (visualisations)
        │                     │                      │
        ▼                     ▼                      ▼
  LLM Backends         AlgorithmWrapper          matplotlib
  ├─ ClaudeBackend      smoke-test → register
  └─ OpenAIBackend
     (Nemotron)               │
                              ▼
                       BenchmarkRunner
                       ├─ Local subprocess
                       └─ Modal sandbox (cloud)
```

### Multi-Agent Pipeline

| Agent | Role | Model |
|---|---|---|
| **Orchestrator** | Conversational router — dispatches tools based on user intent | Claude Sonnet 4 |
| **Intake** | Parses NL problem descriptions + PDFs into structured configs | Claude Sonnet 4 / Nemotron |
| **Implementation** | Generates `AlgorithmWrapper` subclasses from algorithm specs | Claude Sonnet 4 |
| **Plot** | Generates matplotlib code from NL visualisation requests | Claude Sonnet 4 |

### Execution Modes

- **Local** — Each algorithm runs in an isolated subprocess with hard timeout enforcement via `multiprocessing`.
- **Modal** — Each algorithm runs in its own [Modal](https://modal.com) cloud sandbox for full isolation, parallel execution, and scalability.

---

## Tech Stack

### Backend (`agent-backend/`)
- **Python 3.10+**
- **FastAPI** + **Uvicorn** — API server with SSE streaming
- **Anthropic SDK** — Claude Sonnet 4 for all AI agents
- **OpenAI SDK** — Nemotron via OpenAI-compatible endpoint (NVIDIA DGX Spark)
- **Modal** — Serverless sandboxed execution
- **PyMuPDF** — PDF text extraction
- **Pandas / NumPy / NetworkX / SciPy** — Graph generation, data processing
- **Matplotlib** — Chart generation
- **Pydantic** — Data validation and configuration models
- **SQLite** — Chat session and algorithm persistence

### Frontend (`frontend-vite/`)
- **React 19** + **TypeScript**
- **Vite 7** — Dev server and build tool
- **Tailwind CSS 3** — Styling
- **Radix UI** — Accessible primitives (dialogs, tooltips, selects, etc.)
- **Recharts** — Interactive benchmark charts
- **React Router 7** — Client-side routing
- **Lucide React** — Icons
- **React Markdown** — Rendering LLM responses
- **Axios** — HTTP client

---

## Project Structure

```
Benchwarmer.AI/
├── agent-backend/
│   ├── server.py                    # FastAPI app — SSE chat, REST endpoints
│   ├── benchwarmer/
│   │   ├── config.py                # Pydantic models (BenchmarkConfig, AlgorithmSpec, etc.)
│   │   ├── database.py              # SQLite session/message/algorithm persistence
│   │   ├── agents/
│   │   │   ├── orchestrator.py      # Conversational orchestrator (tool-use loop)
│   │   │   ├── intake.py            # NL → structured config agent
│   │   │   ├── implementation.py    # Algorithm code generation agent
│   │   │   ├── plot.py              # NL → matplotlib visualisation agent
│   │   │   ├── backends.py          # LLM abstraction (Claude / Nemotron)
│   │   │   └── tools.py             # Tool definitions for the orchestrator
│   │   ├── engine/
│   │   │   ├── runner.py            # Core benchmark execution engine
│   │   │   ├── modal_runner.py      # Modal cloud execution
│   │   │   └── sandbox_pool.py      # Sandbox lifecycle management
│   │   ├── generators/              # Graph instance generators (Erdős-Rényi, etc.)
│   │   ├── problem_classes/         # Problem-specific validation & objectives
│   │   ├── algorithms/              # AlgorithmWrapper base class
│   │   └── utils/
│   │       ├── loader.py            # Dynamic algorithm loading
│   │       ├── sandbox.py           # Local sandbox execution
│   │       ├── modal_sandbox.py     # Modal sandbox execution
│   │       ├── algorithm_sandbox.py # Algorithm smoke-testing
│   │       └── benchmark_suites.py  # Standard benchmark instances (DIMACS, BiqMac)
│   ├── tests/                       # Pytest test suite
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── .env.example
│
├── frontend-vite/
│   ├── src/
│   │   ├── App.tsx                  # Router setup
│   │   ├── pages/
│   │   │   └── ChatPage.tsx         # Main chat interface
│   │   ├── components/
│   │   │   ├── Sidebar.tsx          # Session management sidebar
│   │   │   ├── Header.tsx           # App header
│   │   │   ├── BenchmarkChart.tsx   # Recharts visualisation
│   │   │   ├── BenchwarmerLogo.tsx  # Animated logo
│   │   │   ├── FileViewer.tsx       # File upload preview
│   │   │   ├── CodeViewer.tsx       # Algorithm code viewer
│   │   │   └── chat/               # Chat message components
│   │   └── hooks/                   # Custom React hooks
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── vercel.json
│
├── SPEC.md                          # Original technical specification
└── README.md                        # ← You are here
```

---

## Prerequisites

- **Python 3.10+** — [python.org](https://www.python.org/downloads/)
- **Node.js 18+** — [nodejs.org](https://nodejs.org/)
- **npm** (comes with Node.js)
- **Anthropic API Key** — [console.anthropic.com](https://console.anthropic.com/)
- *(Optional)* **Modal account** — for cloud sandbox execution ([modal.com](https://modal.com))
- *(Optional)* **NVIDIA DGX Spark** — for Nemotron backend

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-org/Benchwarmer.AI.git
cd Benchwarmer.AI
```

### 2. Backend setup

```bash
cd agent-backend

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
# Copy the example env file
cp .env.example .env
```

Edit `.env` and add your API key:

```env
ANTHROPIC_API_KEY=sk-ant-...
```

See [Environment Variables](#environment-variables) for the full list of options.

### 4. Frontend setup

```bash
cd ../frontend-vite

# Install dependencies
npm install
```

---

## Running the App

You need **two terminals** — one for the backend, one for the frontend.

### Terminal 1 — Backend (FastAPI)

```bash
cd agent-backend

# Activate virtual environment (if not already active)
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Start the API server on port 8000
python -m uvicorn server:app --reload --port 8000
```

The backend will be available at `http://localhost:8000`.

### Terminal 2 — Frontend (Vite)

```bash
cd frontend-vite

# Start the dev server (proxies /api to localhost:8000)
npm run dev
```

The frontend will be available at `http://localhost:5173` (default Vite port).

> **Note:** The Vite dev server is configured to proxy all `/api` requests to `http://localhost:8000`, so both servers work together seamlessly during development.

### Running with Modal (Cloud Sandboxes)

To execute benchmarks in Modal cloud sandboxes instead of local subprocesses:

1. Install and authenticate Modal:
   ```bash
   pip install modal
   modal token new
   ```
2. In the chat UI, select **Modal** as the execution mode when starting a new conversation.

## Environment Variables

Create a `.env` file in `agent-backend/` with the following:

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | **Yes** | Your Anthropic API key for Claude |
| `NEMOTRON_URL` | No | OpenAI-compatible endpoint for Nemotron (default: `http://10.19.177.52:11434/v1`) |
| `NEMOTRON_MODEL` | No | Nemotron model identifier (default: `hf.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q4_K_M`) |
| `MODAL_TOKEN_ID` | No | Modal API token ID (for cloud execution) |
| `MODAL_TOKEN_SECRET` | No | Modal API token secret |
