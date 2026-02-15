# Benchwarmer.AI — Fetch.ai Agentverse Agent

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)

**AI-powered benchmarking for combinatorial optimization algorithms**, now discoverable on ASI:One via Fetch.ai's Agentverse.

## What it does

Benchwarmer.AI automates the end-to-end benchmarking pipeline for combinatorial optimization problems. Describe your problem in plain English, and the agent will:

- Parse your problem description and any attached papers
- Generate or load problem instances
- Code algorithm implementations from papers
- Run benchmarks across instances
- Analyze and visualize results

This wrapper agent bridges the Fetch.ai Chat Protocol to the Benchwarmer.AI backend, making the full pipeline accessible from ASI:One.

## Setup

1. Install dependencies:
   ```bash
   cd fetch-agent
   pip install -r requirements.txt
   ```

2. Create a `.env` file:
   ```
   AGENT_SEED=your-unique-seed-phrase-here
   BACKEND_URL=http://localhost:8000
   ```

3. Start the FastAPI backend (in another terminal):
   ```bash
   cd agent-backend
   python server.py
   ```

4. Start the wrapper agent:
   ```bash
   cd fetch-agent
   python agent.py
   ```

5. On first run the agent prints a **Local Agent Inspector URL**. Open it to register the agent with Agentverse (Connect → Mailbox → Finish).

## Architecture

```
ASI:One / Agentverse Users
        ↕  Chat Protocol (via mailbox)
  fetch-agent/agent.py   (uAgent wrapper)
        ↕  HTTP POST + SSE streaming
  agent-backend/server.py (existing FastAPI backend)
```

The wrapper translates Chat Protocol messages into HTTP requests, streams status updates back as intermediate chat messages, and returns the final response when the pipeline completes.
