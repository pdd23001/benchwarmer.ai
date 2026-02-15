# benchwarmer.ai

**Benchmarking should not be a bottleneck of innovation.**

benchwarmer.ai automates the painful workflow of algorithm benchmarking. Upload your algorithm and the research papers you want to compete against — a multi-agent orchestration framework extracts algorithms from the papers, generates runnable implementations, executes everything in parallel cloud sandboxes, and produces publication-ready comparison charts. What used to take days now takes minutes.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![React](https://img.shields.io/badge/react-19-61DAFB)
![Vite](https://img.shields.io/badge/vite-7-646CFF)
![FastAPI](https://img.shields.io/badge/fastapi-0.109-009688)
![Modal](https://img.shields.io/badge/modal-sandboxes-7C3AED)

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Execution Modes](#execution-modes)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the App](#running-the-app)
- [Environment Variables](#environment-variables)
- [License](#license)

---

## How It Works

1. **Upload** — Drop in your `.py` algorithm and the research papers you want to benchmark against.
2. **Orchestration** — The orchestrator agent understands your intent, routes through the pipeline, and drives the entire session conversationally — no manual steps required.
3. **Intake** — An AI agent parses your description and PDFs, classifies the problem class (Max-Cut, TSP, etc.), and builds a structured benchmark configuration.
4. **Implementation** — Claude generates runnable Python implementations of each challenger algorithm extracted from the papers, smoke-tests them in an isolated sandbox, and registers them for execution.
5. **Execution** — All algorithms run in parallel inside isolated [Modal](https://modal.com) cloud sandboxes — one sandbox per algorithm, full isolation, automatic scaling. One crash never takes down the benchmark.
6. **Analysis** — Results are aggregated into a DataFrame and an AI-powered plot agent generates comparison charts on demand.
7. **Conversation** — The entire flow is driven through a multi-turn chat interface. Ask follow-up questions, tweak parameters, re-run with different instances, request new visualizations — all in natural language.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                       Frontend (Vite + React + TS)                     │
│                                                                        │
│  ChatPage ─── SSE stream ◄──── /api/chat ────► OrchestratorAgent      │
│  Sidebar  ─── REST       ◄──── /api/sessions, /api/algorithms         │
│  SandboxPanel ◄──────────────── benchmark_progress events             │
└────────────────────────────────────────────────────────────────────────┘
                                │
                      FastAPI (uvicorn :8000)
                                │
                ┌───────────────┴───────────────────┐
                │                                   │
                ▼                                   ▼
         OrchestratorAgent                    SQLite (chat.db)
         (conversational router)              session persistence
                │
    ┌───────────┼────────────────┐
    │           │                │
    ▼           ▼                ▼
 IntakeAgent  ImplementationAgent  PlotAgent
 (config)     (code generation)    (visualizations)
    │           │                │
    ▼           ▼                ▼
 LLM Backends  AlgorithmWrapper  matplotlib
 ├─ Claude     smoke-test →
 └─ Nemotron   register
   (DGX Spark)        │
                      ▼
               BenchmarkRunner
               ├─ Sequential (local subprocesses)
               └─ Modal CPU Sandbox (parallel cloud)
                  ├─ 1 sandbox per algorithm
                  ├─ all instances × runs per sandbox
                  └─ real-time progress via SSE
```

### Multi-Agent Orchestration

The system is driven by a central **Orchestrator Agent** that acts as the conversational router. It receives every user message, maintains the full pipeline state, and dispatches tools to specialized sub-agents as needed. The orchestrator supports context forwarding — if the user's initial message specifies algorithms, instances, and parameters, it drives straight through the pipeline without re-asking.

| Agent | Role | Model |
|---|---|---|
| **Orchestrator** | Central router — understands intent, dispatches tools, manages multi-turn state | Claude Sonnet 4 |
| **Intake** | Parses NL problem descriptions + research PDFs into structured benchmark configs | Claude Sonnet 4 / Nemotron |
| **Implementation** | Generates `AlgorithmWrapper` subclasses from algorithm specs, smoke-tests in sandbox | Claude Opus 4.6 |
| **Plot** | Generates matplotlib visualizations from NL requests over benchmark results | Claude Sonnet 4 |

### LLM Backend: Claude vs Nemotron

The intake and orchestration stages are the most token-intensive parts of the pipeline — they process full research papers, lengthy problem descriptions, and maintain multi-turn conversation context. We designed the system to support two LLM backends:

- **Claude Opus 4.6** (Anthropic) — Used for the **Implementation Agent** where code generation accuracy is critical. Opus 4.6 is Anthropic's most capable model for coding tasks, ensuring the generated algorithm implementations are correct, efficient, and faithful to the source papers.
- **Claude Sonnet 4** (Anthropic) — Used for orchestration, intake, and plot generation where speed and tool-use capability matter more than raw coding power.
- **Nemotron-3-Nano-30B** (NVIDIA, open-source) — Deployed locally on **NVIDIA DGX Spark** hardware. As an open-source model, Nemotron eliminates per-token API costs entirely, making it a strong choice for the high-context intake stage where papers and descriptions can consume tens of thousands of tokens per request. Running on DGX Spark also means inference stays on-premises with zero network latency and full data privacy — important when processing unpublished research.

Users can select either backend from the chat UI. This dual-backend design lets teams balance cost, speed, and capability based on their workload.

---

## Execution Modes

### Sequential (Local)

Each algorithm runs in an isolated subprocess on your machine with hard timeout enforcement via `multiprocessing`. Algorithms execute one at a time. Best for quick tests and debugging.

### Modal CPU Sandbox (Parallel)

Each algorithm gets its own [Modal](https://modal.com) cloud sandbox — a fully isolated container running in Modal's infrastructure. All algorithm sandboxes run **in parallel**, with instances and runs executing sequentially within each sandbox. This means a benchmark with 5 algorithms runs ~5x faster than sequential mode.

The frontend provides a real-time **sandbox visualization panel** during Modal execution: each algorithm gets a visual progress indicator showing completion percentage, with a sand-fill animation that rises as runs complete. When all sandboxes finish, the panel shows "Complete" and transitions back to the full chat view.

```
Modal Execution Architecture:

  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │  Sandbox #1  │  │  Sandbox #2  │  │  Sandbox #3  │
  │  random_cut  │  │   goemans    │  │  your_algo   │
  │              │  │              │  │              │
  │ inst1×run1   │  │ inst1×run1   │  │ inst1×run1   │
  │ inst1×run2   │  │ inst1×run2   │  │ inst1×run2   │
  │ inst2×run1   │  │ inst2×run1   │  │ inst2×run1   │
  │    ...       │  │    ...       │  │    ...       │
  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
         │                 │                 │
         └─────── progress events ───────────┘
                         │
                    SSE → Frontend
                  (real-time updates)
```

### Fetch.ai Agentverse

The benchwarmer agent has also been deployed on [Fetch.ai's Agentverse](https://agentverse.ai) using their **ASI:One Pro** model. This makes the benchmarking agent discoverable and callable by other autonomous agents in the Agentverse ecosystem — enabling multi-agent workflows where, for example, a research agent could automatically trigger benchwarmer to validate algorithmic claims from a newly published paper.

---

## Tech Stack

### Backend (`agent-backend/`)
- **Python 3.10+**
- **FastAPI** + **Uvicorn** — API server with SSE streaming for real-time progress
- **Anthropic SDK** — Claude Opus 4.6 for code generation, Claude Sonnet 4 for orchestration and intake
- **NVIDIA Nemotron** — Nemotron-3-Nano-30B deployed on **NVIDIA DGX Spark** as an alternative LLM backend for intake, running locally on DGX hardware for low-latency inference without cloud API costs
- **Modal** — Serverless sandboxed execution with per-algorithm parallelism
- **PyMuPDF** — PDF text extraction for research paper parsing
- **Pandas / NumPy / NetworkX / SciPy** — Graph generation, data processing
- **Matplotlib** — AI-generated comparison charts
- **Pydantic** — Data validation and configuration models
- **SQLite** — Chat session history and algorithm persistence

### Frontend (`frontend-vite/`)
- **React 19** + **TypeScript**
- **Vite 7** — Dev server and build tool
- **Tailwind CSS 3** — Styling
- **Radix UI** — Accessible primitives (dialogs, tooltips, selects, scroll areas)
- **React Router 7** — Client-side routing
- **React Markdown** + **remark-gfm** — Rich rendering of LLM responses with table support
- **Lucide React** — Icons
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
│   │   │   ├── orchestrator.py      # Central orchestrator (tool-use loop, state machine)
│   │   │   ├── intake.py            # NL + PDF → structured config agent
│   │   │   ├── implementation.py    # Algorithm code generation agent
│   │   │   ├── plot.py              # NL → matplotlib visualization agent
│   │   │   ├── backends.py          # LLM abstraction (Claude / Nemotron)
│   │   │   └── tools.py             # Tool definitions for the orchestrator
│   │   ├── engine/
│   │   │   ├── runner.py            # Core benchmark execution engine
│   │   │   ├── modal_runner.py      # Modal cloud sandbox execution
│   │   │   └── sandbox_pool.py      # Sandbox lifecycle management
│   │   ├── generators/              # Graph instance generators (Erdos-Renyi, etc.)
│   │   ├── problem_classes/         # Problem-specific validation & objectives
│   │   ├── algorithms/              # AlgorithmWrapper base class
│   │   └── utils/
│   │       ├── loader.py            # Dynamic algorithm loading
│   │       ├── sandbox.py           # Local sandbox execution
│   │       ├── modal_sandbox.py     # Modal sandbox utilities
│   │       ├── algorithm_sandbox.py # Algorithm smoke-testing
│   │       └── benchmark_suites.py  # Standard benchmark instances (DIMACS, BiqMac)
│   ├── benchmarks/                  # Bundled benchmark instances
│   ├── tests/                       # Pytest test suite
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── .env.example
│
├── frontend-vite/
│   ├── src/
│   │   ├── App.tsx                  # Router setup
│   │   ├── pages/
│   │   │   └── ChatPage.tsx         # Main chat interface + split panels
│   │   ├── components/
│   │   │   ├── Sidebar.tsx          # Navigation, algorithms, chat history
│   │   │   ├── Layout.tsx           # App shell layout
│   │   │   ├── BenchwarmerLogo.tsx  # Animated orbiting logo
│   │   │   └── chat/
│   │   │       ├── MessageList.tsx  # Chat message rendering
│   │   │       ├── ChatInput.tsx    # Message input with file uploads
│   │   │       ├── CodeViewer.tsx   # Algorithm code split-view
│   │   │       ├── SandboxPanel.tsx # Real-time sandbox progress visualization
│   │   │       ├── AlgorithmSelector.tsx
│   │   │       ├── ChoiceSelector.tsx
│   │   │       └── UploadZone.tsx   # Drag-and-drop file uploads
│   │   └── hooks/
│   │       └── use-chat.ts          # Chat state, SSE handling, session management
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── vercel.json                  # SPA rewrite rules for deployment
│
├── SPEC.md                          # Original technical specification
└── README.md
```

---

## Prerequisites

- **Python 3.10+** — [python.org](https://www.python.org/downloads/)
- **Node.js 18+** — [nodejs.org](https://nodejs.org/)
- **Anthropic API Key** — [console.anthropic.com](https://console.anthropic.com/)
- *(Optional)* **Modal account** — for parallel cloud sandbox execution ([modal.com](https://modal.com))
- *(Optional)* **NVIDIA DGX Spark** — for running Nemotron locally as an alternative LLM backend

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

# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure environment variables

Inside `agent-backend/`, copy the example env file:

```bash
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
npm install
```

---

## Running the App

You need **two terminals** — one for the backend, one for the frontend.

### Terminal 1 — Backend

```bash
cd agent-backend
source venv/bin/activate   # or venv\Scripts\activate on Windows
python server.py
```

The backend will be available at `http://localhost:8000`.

### Terminal 2 — Frontend

```bash
cd frontend-vite
npm run dev
```

The frontend will be available at `http://localhost:5173`.

> The Vite dev server proxies all `/api` requests to `http://localhost:8000`, so both servers work together seamlessly during development.

### Running with Modal (Parallel Cloud Sandboxes)

To execute benchmarks in parallel Modal sandboxes:

1. Install and authenticate Modal:
   ```bash
   pip install modal
   modal token new
   ```
2. Select **Modal CPU Sandbox** as the execution mode in the chat UI.

Each algorithm will spin up its own isolated cloud sandbox and run in parallel — you'll see real-time progress for each sandbox in the visualization panel.

### Running with Nemotron on DGX Spark

To use NVIDIA Nemotron as the LLM backend (instead of Claude) for the intake and orchestration agent:

1. Deploy Nemotron-3-Nano-30B on your DGX Spark using [Ollama](https://ollama.com) or any inference server that exposes an OpenAI-compatible API.
2. Set the endpoint in your `.env`:
   ```env
   NEMOTRON_URL=http://<your-dgx-spark-ip>:11434/v1
   NEMOTRON_MODEL=hf.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q4_K_M
   ```
3. Select **Nemotron** as the LLM backend in the chat UI.

This runs inference entirely on local DGX hardware — no cloud API costs, low latency, and full data privacy.

---

## Environment Variables

Create a `.env` file in `agent-backend/` with the following:

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | **Yes** | Anthropic API key for Claude Opus 4.6 and Sonnet 4 |
| `NEMOTRON_URL` | No | Nemotron inference endpoint on DGX Spark (e.g., `http://10.19.177.52:11434/v1`) |
| `NEMOTRON_MODEL` | No | Nemotron model identifier (default: `hf.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q4_K_M`) |
| `MODAL_TOKEN_ID` | No | Modal API token ID (for cloud sandbox execution) |
| `MODAL_TOKEN_SECRET` | No | Modal API token secret |

---

## License

MIT
