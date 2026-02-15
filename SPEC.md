📜 Benchwarmer.ai — Technical Specification & Hackathon Plan
1. The Vision

Benchwarmer.ai is a conversational research assistant that scientifically validates algorithmic claims. Instead of just "generating code," it behaves like a scientist: it researches algorithms, implements them, runs head-to-head benchmarks in a secure sandbox, and discusses the results with you dynamically.

The Loop:

    User: "Benchmark Dijkstra vs. A* Search on a grid." (Natural Language)

    Agent: Writes the code + Test Harness.

    Infrastructure: Executed in Modal Sandbox (User code is never run locally).

    Result: Interactive graphs + Analysis.

    User: "Now show me memory usage instead of time." (Conversational Refinement)

    Agent: Re-renders the analysis instantly.

2. Prize Targets (The "Why")

    Modal (Sandbox Challenge): Core execution engine.

    Greylock (Best Multi-turn Agent): The conversational analysis loop.

    Anthropic (Best Use of Claude): The "Coder" and "Analyst" agents.

    Perplexity (Sonar API): Finding SOTA baselines when the user asks vague questions (e.g., "Compare the fastest sorting algos").

    Decagon (Best Conversational Assistant): The natural language interface.

    Vercel (Best Use of Vercel/Best Deployed on Vercel)

3. System Architecture
A. The "Modal-First" Backend

    Why: We avoid Vercel's 10s timeout. Modal Web Endpoints support up to 15-minute executions, essential for running benchmarks. We will run all our agent code inside the Modal container.

    Entrypoint: backend/modal_app.py.

    Endpoints:

        POST /generate: Streaming endpoint. Receives chat history, yields text chunks (logs) and final JSON (data).

B. The Frontend (Next.js 15)

    Host: Vercel.

    UI Library: Tailwind. No ShadCN/component library. Let's build our own unique UI.

    Visualization: Recharts (Line/Bar charts).

    State: Uses ai/react (Vercel AI SDK) to handle the streaming chat.

    The "Lab" View: A dynamic component that renders:

        Loading State: Live terminal logs from the backend.

        Success State: Interactive Graph + Statistical Summary table.

C. The Agent Logic (LangGraph)

    Orchestrator: A state machine running inside the Modal container.

    Tools:

        search_perplexity(query): Finds algorithm logic.

        generate_code_claude(prompt): Writes Python.

        run_sandbox(code): Executes in isolation.

4. Detailed Data Flow
Step 1: User Request

    "Benchmark QuickSort vs MergeSort on random integers."

Step 2: Research & Planning (Agent)

    Intent Classification: Agent identifies [QuickSort, MergeSort] as targets and List[int] as input.

    Baseline Search: (Optional) If user asks "Compare QuickSort to SOTA", call Perplexity to find "Timsort".

Step 3: Code Generation (Claude)

    The Prompt: "Implement QuickSort and MergeSort. You must adhere to the BenchmarkHarness class interface."

    The Interface:
    Python

    class Algorithm(ABC):
        def load_data(self, size: int): ...
        def run(self): ...

Step 4: The Execution Loop (Modal)

    Draft 1: Agent sends code to modal.Sandbox.

    Smoke Test: Run with N=10.

        Scenario: Code crashes (RecursionError).

        Action: Capture stderr. Send back to Claude. "Fix this error."

    Benchmark Run: Once passing, run for N=[100, 1k, 10k, 100k].

    Metric Capture: time.perf_counter() and tracemalloc.

Step 5: Streaming Response

    The backend streams newline-delimited JSON or text:

        {"status": "log", "message": "Compiling QuickSort..."}

        {"status": "log", "message": "Running N=1000..."}

        {"status": "result", "data": [...series data...]}

5. API Contract

Request (Frontend -> Backend)
JSON

POST https://[your-modal-url].modal.run/generate
{
  "messages": [
    {"role": "user", "content": "Benchmark BFS vs DFS"}
  ],
  "config": {
    "max_n": 10000,
    "trials": 3
  }
}

Response (Streaming)
Plaintext

event: log
data: "Searching for implementation details..."

event: log
data: "Running benchmarks in sandbox..."

event: data
data: {
  "title": "BFS vs DFS Performance",
  "x_label": "Nodes (N)",
  "y_label": "Time (ms)",
  "series": [
    {"name": "BFS", "data": [{"x": 100, "y": 1.2}, {"x": 1000, "y": 15.4}]},
    {"name": "DFS", "data": [{"x": 100, "y": 0.9}, {"x": 1000, "y": 12.1}]}
  ]
}

6. Implementation Checklist (36 Hours)
Phase 1: The Skeleton (Friday Night - 4 Hours)

    [ ] Modal Setup: Create backend/modal_app.py. Get a "Hello World" endpoint running that returns JSON.

    [ ] Frontend Setup: npx create-next-app. Install lucide-react, recharts, ai.

    [ ] Connection: Make the Next.js frontend call the Modal endpoint.

Phase 2: The Muscle (Saturday Morning - 8 Hours)

    [ ] Sandbox Logic: Implement the run_in_sandbox function. It must accept a string of code, execute it, and return stdout.

    [ ] The Harness: Write the Python benchmark_runner.py template that the AI will fill in. This ensures consistent measurement.

    [ ] Claude Integration: Connect anthropic SDK to generate the code based on user prompts.

Phase 3: The Brain (Saturday Afternoon - 8 Hours)

    [ ] Perplexity: Add the search step to find algorithms.

    [ ] Retry Loop: Detect generic Python errors in the sandbox output and trigger a re-generation.

    [ ] Visuals: Connect the JSON output to the Recharts component.

Phase 4: Polish (Sunday Morning)

    [ ] Pre-canned Demos: Hardcode 3 perfect examples (Sorting, Pathfinding, Matrix Mult) for the judges.

    [ ] Streaming Logs: Ensure the user sees the "terminal" scrolling while waiting.

    [ ] Analysis: Allow the user to ask "Summarize this" after the graph loads.

7. Critical Constraints

    Timeout: Modal Web Endpoints have a configurable timeout. Set it to 600s (10 mins).

    Security: NEVER exec() code in the main process. ALWAYS use modal.Sandbox.

    Dependencies: The Modal Image must pre-install standard scientific libs:
    image = modal.Image.debian_slim().pip_install("numpy", "pandas", "scipy", "networkx", "scikit-learn")