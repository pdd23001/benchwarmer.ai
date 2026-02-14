# Revised Architecture: Natural Language Intake + User-Driven Plots

---

## Change 1: Natural Language Problem Description

### Before (rigid)
```yaml
problem_class: maximum_cut
instances:
  generators:
    - type: erdos_renyi
      params: {p: 0.3}
      sizes: [50, 100, 200, 500]
```

### After (natural language)
```
User: "I'm working on partitioning social networks to maximize 
       cross-group connections. My graphs are sparse, undirected, 
       with weighted edges representing interaction frequency. 
       Typical size is 500-5000 nodes. I care most about solution 
       quality but need results within 60 seconds."
```

The system figures out: this is Max-Cut on sparse weighted undirected graphs,
needs instance generators that mimic social network topology (Barabási-Albert,
planted partition), benchmark sizes should go up to 5000, and the analysis
should weight quality heavily but flag anything over 60s.

---

### Intake Agent Design

**Model:** Claude Sonnet 4 via tool_use

**Job:** Turn a freeform problem description into a structured `BenchmarkConfig`
that the rest of the pipeline can consume.

**System prompt (core logic):**

```
You are the intake agent for an algorithm benchmarking platform.

The user will describe their optimization problem in natural language. 
Your job is to:

1. UNDERSTAND the problem — ask clarifying questions if needed
2. CLASSIFY it into a known problem class (or flag it as custom)
3. INFER the right benchmarking setup:
   - What graph types match their real-world scenario?
   - What sizes to test at?
   - What matters more: speed, quality, memory, consistency?
   - Any hard constraints (e.g., "must run under 60 seconds")?
4. OUTPUT a structured BenchmarkConfig JSON

You have access to the following tools:
- classify_problem(description) → returns candidate problem classes with confidence
- get_generators(problem_class) → returns available instance generators
- validate_config(config) → checks if a config is valid and complete

IMPORTANT BEHAVIORS:
- If the problem clearly maps to a known class, don't over-ask. Confirm and move on.
- If it's ambiguous (could be Max-Cut OR graph partitioning), ask ONE clarifying question.
- Always infer instance generators from the user's domain description:
    - "social networks" → Barabási-Albert, planted partition
    - "road networks" → grid-like graphs, planar graphs
    - "molecular structures" → sparse, bounded-degree graphs
    - "internet topology" → power-law graphs
    - "random benchmarks" → Erdős-Rényi
- Extract any implicit constraints the user mentioned.
- Don't ask about things you can set sensible defaults for.
```

**Conversation flow:**

```
User: "I'm trying to find the minimum set of sensors that covers all 
       corridors in a building. Each corridor connects two junctions. 
       I want to compare my greedy approach against whatever the 
       literature recommends."

Agent (internally): 
  → classify_problem("minimum set of nodes covering all edges") 
  → returns: Minimum Vertex Cover (confidence: 0.95)
  → get_generators("minimum_vertex_cover")
  → returns: [erdos_renyi, grid_2d, planar, barabasi_albert]
  → Building corridors → grid/planar topology

Agent: "This sounds like a Minimum Vertex Cover problem — you want 
        the smallest set of junctions such that every corridor has at 
        least one sensor. Since you're modeling building layouts, I'll 
        generate test instances using grid and planar graph structures, 
        which mimic real floor plans. 
        
        Two quick questions:
        - Roughly how many junctions in your typical building? (So I 
          can set the right benchmark sizes)
        - Is there a time budget per solution, or is quality all 
          that matters?"

User: "Usually 50-200 junctions. Quality matters most but I'd like 
       each run under 30 seconds."

Agent: "Got it. Here's what I'll set up:"
  → outputs BenchmarkConfig
```

**Output: BenchmarkConfig**

```json
{
  "problem_class": "minimum_vertex_cover",
  "problem_description": "Sensor placement covering building corridors",
  "objective": "minimize",
  
  "instance_config": {
    "generators": [
      {
        "type": "grid_2d",
        "why": "Mimics building floor plan layouts",
        "sizes": [25, 50, 100, 200, 400],
        "count_per_size": 3
      },
      {
        "type": "planar_random",
        "why": "Irregular building layouts",
        "sizes": [25, 50, 100, 200, 400],
        "count_per_size": 3
      }
    ],
    "custom_instances": []
  },

  "evaluation_priorities": {
    "solution_quality": "primary",
    "runtime": "secondary — hard ceiling 30s",
    "memory": "not a concern",
    "consistency": "report but don't optimize for"
  },

  "execution_config": {
    "timeout_seconds": 30,
    "runs_per_config": 5,
    "memory_limit_mb": 2048
  },

  "solution_validation": {
    "feasibility_check": "every edge must have at least one endpoint in the solution set",
    "objective_function": "count of nodes in solution set (minimize)"
  }
}
```

**The taxonomy still exists — but it lives behind the scenes.** The user never
picks from a dropdown. The intake agent maps their description to the taxonomy
and uses it to select generators, validation logic, and evaluation criteria.

---

### Problem Class Registry (Backend)

Each problem class is a Python module with standardized functions:

```python
# problem_classes/minimum_vertex_cover.py

class ProblemClass:
    name = "minimum_vertex_cover"
    objective = "minimize"
    description = "Find the smallest subset of vertices that covers all edges"
    
    # Keywords the intake agent can use for classification
    keywords = [
        "vertex cover", "node cover", "sensor placement", 
        "cover all edges", "hitting set on edges",
        "minimum dominating", "facility coverage"
    ]

    @staticmethod
    def validate_solution(instance: dict, solution: dict) -> dict:
        """Check if a solution is feasible."""
        cover_set = set(solution["solution"]["vertices"])
        uncovered = []
        for edge in instance["edges"]:
            if edge["source"] not in cover_set and edge["target"] not in cover_set:
                uncovered.append(edge)
        return {
            "feasible": len(uncovered) == 0,
            "uncovered_edges": len(uncovered),
            "cover_size": len(cover_set),
        }

    @staticmethod
    def compute_objective(instance: dict, solution: dict) -> float:
        """Return the objective value (lower is better for this problem)."""
        return len(solution["solution"]["vertices"])

    @staticmethod
    def available_generators() -> list[str]:
        return ["erdos_renyi", "grid_2d", "planar_random", "barabasi_albert"]
```

Adding a new problem class = adding one Python file. The intake agent
discovers available classes by reading the registry.

---

## Change 2: User-Driven Plot Generation

### Before (fixed plot set)
The system always generates the same 6 plots regardless of what the user 
actually wants to see.

### After (user asks for what they want)
```
User: "Show me how solution quality degrades as graph density increases"
User: "I want a Pareto front of quality vs runtime at n=200"
User: "Compare the consistency of each algorithm — box plots of objective value"
User: "How does my algorithm's runtime scale compared to the SDP approach?"
User: "Show the results as a table, not a chart"
```

---

### Plot Agent Design

**Model:** Claude Sonnet 4

**Job:** Take a natural language visualization request + the benchmark results
DataFrame, and generate the Python plotting code to produce it.

**Why generate code instead of using a charting LLM tool?**
- Full control over aesthetics (publication-quality if needed)
- Reproducible — user can re-run the same plot code later
- No hallucinated data — the code reads directly from the results DataFrame
- User can tweak the generated code if they want

**System prompt:**

```
You are a visualization agent for an algorithm benchmarking platform.

You will receive:
1. A natural language request for a plot or table
2. The schema of the results DataFrame (columns, dtypes, sample rows)
3. The list of algorithms and instance configurations

Your job: write Python code using matplotlib and/or pandas that 
produces the requested visualization.

RULES:
- The results DataFrame is already loaded as `df` with these columns:
    algorithm_name: str
    instance_name: str
    instance_generator: str
    problem_size: int
    objective_value: float
    wall_time_seconds: float
    peak_memory_mb: float
    status: str ("success" | "timeout" | "error")
    run_index: int
    
- Use matplotlib with a clean style (seaborn-v0_8 or similar)
- Label everything: title, axes, legend
- Use colorblind-friendly palettes
- Save output to the path provided as `output_path`
- For tables, generate a styled HTML table or a formatted terminal table
- Filter out status != "success" before plotting unless the request 
  is specifically about failure rates
- When showing aggregates, always use mean with error bars (std or CI)
- If the user's request is ambiguous, make a reasonable choice and 
  add a code comment explaining the decision

OUTPUT: Only the Python code. No explanation.
```

**Example interactions:**

```
User: "Show me quality vs runtime tradeoff at the largest problem size"

Agent generates:
───────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np

# Filter to largest problem size, successful runs only
max_size = df["problem_size"].max()
data = df[(df["problem_size"] == max_size) & (df["status"] == "success")]

# Aggregate per algorithm: mean quality, mean time, with std
agg = data.groupby("algorithm_name").agg(
    mean_quality=("objective_value", "mean"),
    std_quality=("objective_value", "std"),
    mean_time=("wall_time_seconds", "mean"),
    std_time=("wall_time_seconds", "std"),
).reset_index()

fig, ax = plt.subplots(figsize=(10, 7))

colors = plt.cm.Set2(np.linspace(0, 1, len(agg)))

for i, row in agg.iterrows():
    ax.errorbar(
        row["mean_time"], row["mean_quality"],
        xerr=row["std_time"], yerr=row["std_quality"],
        fmt="o", markersize=12, capsize=5,
        color=colors[i], label=row["algorithm_name"],
        linewidth=2
    )

ax.set_xlabel("Wall-clock Time (seconds)", fontsize=13)
ax.set_ylabel("Objective Value", fontsize=13)
ax.set_title(f"Quality vs Runtime Tradeoff (n={max_size})", fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Pareto front
# Sort by time, mark non-dominated points
sorted_agg = agg.sort_values("mean_time")
pareto = []
best_quality = float("-inf")
for _, row in sorted_agg.iterrows():
    if row["mean_quality"] > best_quality:
        pareto.append(row)
        best_quality = row["mean_quality"]

if len(pareto) > 1:
    pareto_df = pd.DataFrame(pareto)
    ax.plot(
        pareto_df["mean_time"], pareto_df["mean_quality"],
        "--", color="gray", alpha=0.5, label="Pareto front"
    )

plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches="tight")
───────────────────────────────────────────────────
```

```
User: "Just give me a summary table of all results"

Agent generates:
───────────────────────────────────────────────────
import pandas as pd

success = df[df["status"] == "success"]

summary = success.groupby(["algorithm_name", "problem_size"]).agg(
    avg_quality=("objective_value", "mean"),
    std_quality=("objective_value", "std"),
    avg_time=("wall_time_seconds", "mean"),
    avg_memory=("peak_memory_mb", "mean"),
    success_rate=("status", "count"),  # all are "success" here
).reset_index()

# Pivot for readability: rows=algorithm, columns=metric@size
summary["size_label"] = "n=" + summary["problem_size"].astype(str)

quality_table = summary.pivot(
    index="algorithm_name", 
    columns="size_label", 
    values="avg_quality"
).round(2)

time_table = summary.pivot(
    index="algorithm_name",
    columns="size_label",
    values="avg_time"
).round(4)

# Save as HTML
html = "<h2>Solution Quality (mean objective value)</h2>"
html += quality_table.to_html()
html += "<br><h2>Runtime (mean seconds)</h2>"
html += time_table.to_html()

with open(output_path, "w") as f:
    f.write(html)
───────────────────────────────────────────────────
```

---

### Conversation Loop for Iterative Exploration

This is where it gets powerful. After the initial benchmark run, the user
enters an interactive analysis session:

```
System: "Benchmarking complete. 5 algorithms × 20 instances × 5 runs = 
         500 data points collected. What would you like to see?"

User: "How did my algorithm do overall?"

Agent: [generates summary table + brief narrative]
       "Your Randomized Greedy found solutions averaging 3.2% below the 
        SDP Relaxation approach on medium instances, but ran 47x faster. 
        On large instances (n=400), the gap widened to 8.1%. 
        Want me to plot the scaling comparison?"

User: "Yeah show me the scaling"

Agent: [generates runtime scaling plot, log-log, with fitted curves]

User: "Interesting. On the Barabási graphs specifically, how does quality compare?"

Agent: [generates filtered quality comparison, Barabási instances only]

User: "Can you show a heatmap — algorithms on one axis, instance types 
       on the other, colored by quality?"

Agent: [generates heatmap]

User: "Export everything as a PDF report"

Agent: [compiles all generated plots + narrative into PDF]
```

**This means the plot generation is NOT a one-shot step. It's a 
conversational loop where the user explores their results interactively.**

---

### Revised Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   User: Natural language          User: "my_algorithm.py"       │
│   problem description             + paper PDFs                   │
│          │                                │                      │
│          ▼                                ▼                      │
│   ┌──────────────┐               ┌────────────────┐             │
│   │ Intake Agent │               │ Implementation │             │
│   │ (Claude)     │               │ Agent (Claude)  │             │
│   │              │               │                 │             │
│   │ NL → config  │               │ Paper → code    │             │
│   └──────┬───────┘               └────────┬───────┘             │
│          │                                │                      │
│          ▼                                ▼                      │
│   BenchmarkConfig               Algorithm implementations       │
│          │                                │                      │
│          └──────────────┬─────────────────┘                      │
│                         ▼                                        │
│               ┌──────────────────┐                               │
│               │ Execution Engine │                               │
│               │ (No LLM)        │                               │
│               │                  │                               │
│               │ Run all algos    │                               │
│               │ Collect metrics  │                               │
│               └────────┬─────────┘                               │
│                        │                                         │
│                        ▼                                         │
│                  Results DataFrame                                │
│                        │                                         │
│                        ▼                                         │
│          ┌──────────────────────────┐                            │
│          │  Interactive Analysis    │ ◄──── User: "show me..."   │
│          │  (Claude + matplotlib)   │ ◄──── User: "compare..."   │
│          │                          │ ◄──── User: "why did..."   │
│          │  - Generates plot code   │                            │
│          │  - Executes it           │                            │
│          │  - Narrates findings     │                            │
│          │  - Responds to follow-ups│                            │
│          └──────────┬───────────────┘                            │
│                     │                                            │
│                     ▼                                            │
│           Plots + Tables + Narrative                             │
│           (live in dashboard OR exported as PDF)                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### What This Changes About the Build Plan

**Week 1 stays the same** — instance generators, execution engine, algorithm 
interface. These don't depend on whether input is NL or structured.

**Week 2 adds:**
- Intake agent with problem classification tool
- Problem class registry (Python modules)
- The mapping from domain descriptions → graph generators

**Week 3 changes:**
- Instead of fixed plotting code, build the plot agent loop:
  - Claude receives results schema + user request
  - Claude generates matplotlib code
  - System executes code in sandbox
  - Returns plot to user
  - User asks follow-up → repeat
- This is actually simpler to build than a custom dashboard with 
  pre-built chart components — you're offloading the chart design to Claude

**Week 4:**
- Polish the conversational flow
- PDF report compilation from accumulated plots + narrative
- Handle edge cases (user asks for impossible plots, empty results, etc.)

---

### Risk: LLM-Generated Plots Can Be Wrong

The plot agent could generate code that:
- Aggregates incorrectly (mean when median is appropriate)
- Filters out data it shouldn't
- Misinterprets what the user asked for
- Produces ugly or misleading visualizations

**Mitigations:**
1. Always show the generated code alongside the plot (transparency)
2. Include a "raw data" export so users can verify
3. Have a small test suite of "canonical requests" that you validate 
   during development (e.g., "summary table" should always produce 
   the same structure)
4. Let users edit the generated code and re-run (power user feature)
5. Include standard sanity checks: warn if N < 5 (low statistical 
   power), warn if all algorithms have same score (likely a bug), etc.
