# Benchwarmer.AI — Detailed Explanation

## What Is Benchwarmer.AI?

Benchwarmer.AI is an **AI-powered algorithm benchmarking platform** for graph optimization problems. Instead of manually writing benchmark harnesses, you describe your problem in plain English and the system:

1. **Understands** your problem using an LLM (Claude Sonnet 4)
2. **Generates** a complete benchmark configuration automatically
3. **Runs** your algorithms against diverse graph instances
4. **Visualizes** results interactively through natural language

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    run_benchmark.py                      │
│               (Full end-to-end CLI)                      │
├─────────┬──────────────┬──────────────┬─────────────────┤
│ Intake  │  Benchmark   │  Execution   │    Plot         │
│ Agent   │  Config      │  Engine      │    Agent        │
│ (NL→    │  (Pydantic   │  (Runner +   │    (NL→Code→   │
│  JSON)  │   models)    │   Metrics)   │     Plots)      │
├─────────┴──────────────┴──────────────┴─────────────────┤
│            Foundation Layer                               │
│  ┌────────────┬───────────────┬───────────────────────┐  │
│  │ Generators │ Problem       │ Algorithm             │  │
│  │ (5 types)  │ Classes (2)   │ Interface (ABC)       │  │
│  └────────────┴───────────────┴───────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Benchwarmer.AI/
├── benchwarmer/                  # Main package
│   ├── config.py                 # Pydantic data models
│   ├── agents/
│   │   ├── intake.py             # NL → BenchmarkConfig (Claude)
│   │   ├── plot.py               # NL → matplotlib code (Claude)
│   │   └── tools.py              # Deterministic tool functions
│   ├── algorithms/
│   │   └── base.py               # AlgorithmWrapper ABC
│   ├── engine/
│   │   └── runner.py             # BenchmarkRunner execution engine
│   ├── generators/
│   │   ├── base.py               # BaseGenerator ABC
│   │   ├── erdos_renyi.py        # Random graphs G(n, p)
│   │   ├── barabasi_albert.py    # Scale-free networks
│   │   ├── grid_2d.py            # 2D grid/lattice graphs
│   │   ├── planar_random.py      # Random planar graphs
│   │   └── planted_partition.py  # Community-structured graphs
│   ├── problem_classes/
│   │   ├── registry.py           # Auto-discovery registry
│   │   ├── maximum_cut.py        # Max-Cut problem definition
│   │   └── minimum_vertex_cover.py  # Min Vertex Cover definition
│   └── utils/
│       └── sandbox.py            # Sandboxed code execution
├── scripts/
│   ├── run_benchmark.py          # Full CLI entry point
│   └── demo_phase1.py            # Phase 1 standalone demo
├── tests/                        # 47 unit tests
│   ├── test_agents.py            # Tool function tests
│   ├── test_engine.py            # Runner tests
│   ├── test_generators.py        # Generator tests
│   ├── test_problem_classes.py   # Problem class tests
│   └── test_sandbox.py           # Sandbox tests
├── pyproject.toml
└── requirements.txt
```

---

## How It Works — Step by Step

### Step 1: Intake Agent (NL → Configuration)

**File:** `benchwarmer/agents/intake.py`

When you type something like `"max cut problem"`, the Intake Agent:

1. **Classifies** the problem using `classify_problem()` — a deterministic keyword-matching tool that scores your description against registered problem classes
2. **Looks up generators** using `get_generators()` — retrieves all available graph generators for the matched problem class
3. **Builds a config** — Claude assembles a `BenchmarkConfig` JSON with appropriate generators, sizes, and parameters
4. **Validates** using `validate_config()` — checks the JSON against the Pydantic schema

The agent uses **Claude Sonnet 4's tool-use** capability. The three tool functions are defined in `agents/tools.py` and are purely deterministic (no API calls), making them independently testable.

**Key design decision:** The system prompt includes an **exact JSON schema example** so Claude generates valid configs on the first try without unnecessary clarifying questions.

### Step 2: Configuration Models

**File:** `benchwarmer/config.py`

The entire benchmark is described by a `BenchmarkConfig` Pydantic model:

```python
BenchmarkConfig
├── problem_class: str              # e.g. "maximum_cut"
├── problem_description: str        # Human-readable
├── objective: "minimize"|"maximize"
├── instance_config: InstanceConfig
│   └── generators: list[GeneratorConfig]
│       ├── type: str               # e.g. "erdos_renyi"
│       ├── params: dict            # e.g. {"p": 0.3}
│       ├── sizes: list[int]        # e.g. [50, 100, 200, 500]
│       ├── count_per_size: int     # default 3
│       └── why: str                # Reasoning for this generator
├── evaluation_priorities: EvaluationPriorities
├── execution_config: ExecutionConfig
│   ├── timeout_seconds: float      # default 60
│   ├── runs_per_config: int        # default 5
│   └── memory_limit_mb: int        # default 2048
└── solution_validation: SolutionValidation
```

**Notable fix:** `GeneratorConfig.params` uses `validation_alias=AliasChoices("params", "parameters")` because Claude sometimes generates `"parameters"` instead of `"params"`. Without this, the params dict would silently be empty.

### Step 3: Instance Generation

**Files:** `benchwarmer/generators/*.py`

Five graph generators, all subclassing `BaseGenerator`:

| Generator | Description | Key Params | Use Case |
|---|---|---|---|
| **erdos_renyi** | Random graph G(n, p) | `p` (edge probability) | General benchmarks |
| **barabasi_albert** | Preferential attachment | `m` (edges per new node) | Social networks, scale-free |
| **grid_2d** | 2D lattice/grid | — | Road networks, structured |
| **planar_random** | Delaunay triangulation | `weighted` | Geographic networks |
| **planted_partition** | Community structure | `k`, `p_in`, `p_out` | Known-optimal testing |

Every generator outputs a **standardized graph dict**:
```python
{
    "nodes": [0, 1, 2, ...],
    "edges": [{"source": 0, "target": 1, "weight": 1.0}, ...],
    "metadata": {"generator": "erdos_renyi", "size": 100, "params": {"p": 0.3}}
}
```

### Step 4: Algorithm Interface

**File:** `benchwarmer/algorithms/base.py`

Users implement the `AlgorithmWrapper` abstract class:

```python
class MyAlgorithm(AlgorithmWrapper):
    name = "my_algo"

    def solve(self, instance: dict, timeout: float = 60.0) -> dict:
        # instance has "nodes", "edges", "metadata"
        # ... your algorithm ...
        return {
            "solution": {"partition": [0, 1, 0, 1, ...]},
            "metadata": {"iterations": 42}
        }
```

The CLI provides **built-in baselines** (greedy + random) for both Max-Cut and Min Vertex Cover so the benchmark runs immediately without custom code.

### Step 5: Execution Engine

**File:** `benchwarmer/engine/runner.py`

`BenchmarkRunner` orchestrates the full benchmark:

1. **Generate instances** — iterates over generators × sizes × count_per_size
2. **Run algorithms** — for each (algorithm × instance × run_index):
   - Start `tracemalloc` for memory tracking
   - Time the `solve()` call with `time.perf_counter()`
   - Validate the solution via the problem class's `evaluate()` method
   - Record a `BenchmarkResult` with objective value, time, memory, status
3. **Collect results** — returns a pandas DataFrame with columns:

| Column | Type | Description |
|---|---|---|
| `algorithm_name` | str | Algorithm identifier |
| `instance_name` | str | e.g. "erdos_renyi_n100_2" |
| `instance_generator` | str | Generator type used |
| `problem_size` | int | Number of nodes |
| `objective_value` | float | Solution quality score |
| `wall_time_seconds` | float | Execution time |
| `peak_memory_mb` | float | Peak memory usage |
| `status` | str | "success", "timeout", or "error" |
| `run_index` | int | Run number (for statistical reliability) |
| `feasible` | bool | Whether the solution is valid |

### Step 6: Problem Classes

**Files:** `benchwarmer/problem_classes/maximum_cut.py`, `minimum_vertex_cover.py`

Each problem class defines:
- **`evaluate(instance, solution)`** — computes the objective value
- **`is_feasible(instance, solution)`** — checks solution validity
- **`available_generators()`** — lists compatible generators
- **`keywords`** — for NL classification matching

The **registry** (`registry.py`) auto-discovers problem class modules at import time using `pkgutil`.

### Step 7: Interactive Plot Agent

**File:** `benchwarmer/agents/plot.py`

After benchmarks finish, you enter an interactive analysis loop. Type requests like:
- `"Bar chart comparing average objective by algorithm"`
- `"Box plot of wall time by algorithm for each graph size"`
- `"Summary table of results"`

The Plot Agent:
1. Sends the DataFrame schema + sample rows + your request to Claude
2. Claude generates matplotlib Python code
3. Code runs in the **sandbox** (`utils/sandbox.py`)
4. If it fails, the error is fed back to Claude for **self-correction** (up to 2 retries)
5. The generated code is displayed for transparency

### Step 8: Sandboxed Execution

**File:** `benchwarmer/utils/sandbox.py`

LLM-generated code runs in a **restricted namespace**:
- ✅ Pre-injected: `df`, `plt`, `pd`, `np`, `matplotlib`, `output_path`
- ✅ Safe builtins: `print`, `len`, `range`, `sorted`, `min`, `max`, etc.
- ❌ Blocked: `open()`, `exec()`, `eval()`, `__import__()`, file system access

If no explicit `plt.savefig()` is called, the sandbox auto-saves any open figures.

---

## The Full Pipeline

When you run `python scripts/run_benchmark.py`:

```
1. "Describe your problem" → user types "max cut problem"

2. Intake Agent (4 Claude API calls):
   → classify_problem("max cut problem")     → "maximum_cut" (0.9 confidence)
   → get_generators("maximum_cut")           → 5 generators available
   → validate_config({...})                  → valid ✅
   → Present config to user

3. Auto-register built-in baselines:
   → greedy_cut (adjacency-based greedy partitioning)
   → random_cut (random 50/50 partition)

4. BenchmarkRunner executes:
   → 4 generators × 4 sizes × 3 instances × 5 runs × 2 algorithms = 480 runs
   → Each run: timed, memory-tracked, solution validated
   → Results: pandas DataFrame

5. Summary table printed

6. Interactive analysis loop:
   📊 > "box plot of wall time by size"
   → Claude generates matplotlib code
   → Sandbox executes → plot saved to plots/plot_000.png
   📊 > "thanks"
   → Exit
```

---

## Testing

47 unit tests across 5 test files, all deterministic (no API key required):

```
tests/test_agents.py          — 14 tests (tool functions, dispatcher)
tests/test_engine.py           —  6 tests (runner, metrics, failures)
tests/test_generators.py       — 13 tests (all 5 generators)
tests/test_problem_classes.py  —  8 tests (Max-Cut, Min Vertex Cover)
tests/test_sandbox.py          —  6 tests (execution, errors, safety)
```

Run with: `python -m pytest tests/ -v`

---

## Dependencies

| Package | Purpose |
|---|---|
| `anthropic` | Claude Sonnet 4 API for Intake and Plot agents |
| `pydantic` | Data validation and config models |
| `pandas` | Results DataFrame and analysis |
| `networkx` | Graph generation (all generators use this) |
| `matplotlib` | Plot generation |
| `numpy` | Numerical operations in plots |
| `python-dotenv` | Load API key from `.env` file |

---

## Key Design Decisions

1. **Deterministic tools, stochastic agent** — The three Intake Agent tools are pure functions that can be tested without an API key. Only the LLM orchestration layer requires the Anthropic API.

2. **Schema-driven LLM guidance** — Instead of hoping the LLM guesses the right JSON format, the system prompt and tool schemas include explicit examples, reducing invalid configs from ~70% to ~0%.

3. **Self-correcting Plot Agent** — If generated code fails, the error trace is fed back to Claude, which fixes and retries (up to 2×). This handles edge cases in matplotlib without user intervention.

4. **Sandboxed execution** — Generated plot code runs in a restricted namespace with `__builtins__` overridden. This blocks dangerous operations like `open()`, `exec()`, and `__import__()`.

5. **Auto-discovery registries** — Both generators and problem classes are auto-discovered via `pkgutil`. Adding a new problem class is as simple as dropping a new `.py` file in `problem_classes/`.

6. **Natural language exit** — The analysis loop recognizes phrases like "thanks", "done", "that's enough" — not just explicit `exit` commands.
