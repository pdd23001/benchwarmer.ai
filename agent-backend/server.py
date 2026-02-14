
import logging
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Ensure we can import from benchwarmer package
sys.path.insert(0, ".")

from benchwarmer.algorithms.base import AlgorithmWrapper
from benchwarmer.config import BenchmarkConfig, GeneratorConfig, InstanceConfig
from benchwarmer.engine.runner import BenchmarkRunner
from benchwarmer.agents.intake import IntakeAgent
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

app = FastAPI()

# ─── Data Models ──────────────────────────────────────────────────────────────

class BenchmarkRequest(BaseModel):
    query: str

class SeriesData(BaseModel):
    name: str
    color: str
    dataKey: str

class BenchmarkResponse(BaseModel):
    title: str
    xLabel: str
    yLabel: str
    series: List[SeriesData]
    data: List[Dict[str, Any]]

# ─── Toy Algorithms (Reused from demo_phase1.py) ──────────────────────────────

class GreedyVertexCover(AlgorithmWrapper):
    """Simple greedy: repeatedly pick the endpoint of any uncovered edge."""
    name = "greedy_vc"

    def solve(self, instance: dict, timeout: float = 60.0) -> dict:
        covered: set[int] = set()
        cover: list[int] = []
        for edge in instance["edges"]:
            u, v = edge["source"], edge["target"]
            if u not in covered and v not in covered:
                cover.append(u)
                covered.add(u)
        # Double check
        for edge in instance["edges"]:
            u, v = edge["source"], edge["target"]
            if u not in covered and v not in covered:
                cover.append(u)
                covered.add(u)
        return {"solution": {"vertices": cover}, "metadata": {"strategy": "greedy"}}


class RandomVertexCover(AlgorithmWrapper):
    """Picks endpoints of edges at random until all are covered."""
    name = "random_vc"

    def solve(self, instance: dict, timeout: float = 60.0) -> dict:
        import random
        covered: set[int] = set()
        cover: list[int] = []
        edges = list(instance["edges"])
        random.shuffle(edges)
        for edge in edges:
            u, v = edge["source"], edge["target"]
            if u not in covered and v not in covered:
                chosen = random.choice([u, v])
                cover.append(chosen)
                covered.add(chosen)
        # Second pass
        for edge in instance["edges"]:
            u, v = edge["source"], edge["target"]
            if u not in covered and v not in covered:
                cover.append(u)
                covered.add(u)
        return {"solution": {"vertices": cover}, "metadata": {"strategy": "random"}}

class RandomMaxCut(AlgorithmWrapper):
    """Randomly partitions vertices into two sets (0 and 1)."""
    name = "random_max_cut"

    def solve(self, instance: dict, timeout: float = 60.0) -> dict:
        import random
        partition = []
        n_nodes = len(instance["nodes"])
        # If nodes are just a count, we assume 0..n-1.
        # If instance["nodes"] is a list of metadata, we just need the count.
        # But wait, instance["nodes"] in Benchwarmer is usually a list of dicts or just indices?
        # Let's check ErdosRenyiGenerator. usually it's list of ints.
        # Actually, let's just use the length.
        
        partition = [random.choice([0, 1]) for _ in range(n_nodes)]
        
        return {"solution": {"partition": partition}, "metadata": {"strategy": "random"}}

# ─── Helper Functions ────────────────────────────────────────────────────────—

def run_demo_benchmark(query: str) -> pd.DataFrame:
    """
    Runs a hardcoded benchmark for demonstration purposes.
    In a real scenario, 'query' would be used to generate the config.
    """
    print(f"Received query: {query}")
    
    config = BenchmarkConfig(
        problem_class="minimum_vertex_cover",
        problem_description=f"Demo: {query}",
        objective="minimize",
        instance_config=InstanceConfig(
            generators=[
                GeneratorConfig(type="erdos_renyi", sizes=[20, 50, 100], count_per_size=2, params={"p": 0.3}),
                GeneratorConfig(type="grid_2d", sizes=[25, 36, 100], count_per_size=2),
            ]
        ),
        execution_config={"timeout_seconds": 30, "runs_per_config": 2},
    )

    runner = BenchmarkRunner(config)
    runner.register_algorithm(GreedyVertexCover())
    runner.register_algorithm(RandomVertexCover())

    return runner.run()

def transform_results(df: pd.DataFrame) -> BenchmarkResponse:
    """
    Transforms the pandas DataFrame into the frontend's expected JSON format.
    """
    # Filter for successes
    success_df = df[df["status"] == "success"].copy()
    
    if success_df.empty:
        raise HTTPException(status_code=500, detail="No successful benchmark runs generated.")

    # Group by problem_size and algorithm to get mean objective value
    # We want to plot: x=problem_size, y=objective_value, series=algorithm
    
    # Calculate aggregation
    agg = success_df.groupby(["algorithm_name", "problem_size"])["objective_value"].mean().reset_index()
    
    # Pivot to get: index=problem_size, columns=algorithm_name, values=objective_value
    pivot = agg.pivot(index="problem_size", columns="algorithm_name", values="objective_value").reset_index()
    
    # Construct data list
    data = []
    for _, row in pivot.iterrows():
        item = {"x": int(row["problem_size"])}
        for col in pivot.columns:
            if col != "problem_size":
                # Handle NaN if an algo failed for a specific size
                val = row[col]
                if pd.notna(val):
                    item[col] = float(val)
        data.append(item)
    
    # Sort data by x (problem_size)
    data.sort(key=lambda d: d["x"])

    # Define series based on available algorithms
    algorithms = [c for c in pivot.columns if c != "problem_size"]
    series = []
    colors = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b"] # Red, Blue, Green, Amber
    
    for i, algo in enumerate(algorithms):
        series.append(SeriesData(
            name=algo,
            color=colors[i % len(colors)],
            dataKey=algo
        ))

    return BenchmarkResponse(
        title="Vertex Cover: Greedy vs Random",
        xLabel="Graph Size (Nodes)",
        yLabel="Average Vertex Cover Size",
        series=series,
        data=data
    )

# ─── Endpoints ────────────────────────────────────────────────────────────────

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/api/benchmark", response_model=BenchmarkResponse)
async def run_benchmark_endpoint(request: BenchmarkRequest):
    try:
        logging.info(f"Processing query: {request.query!r}")  # Use repr to see whitespace/empty
        
        # 1. Initialize Intake Agent
        #    It will automatically look for ANTHROPIC_API_KEY in os.environ
        agent = IntakeAgent()
        
        # 2. Run Intake Agent (non-interactive mode for API)
        #    This uses Claude to generate the BenchmarkConfig
        config = agent.run(request.query, interactive=False)
        logging.info(f"Generated config: {config}")

        # 3. Initialize Runner
        runner = BenchmarkRunner(config)
        
        # 4. Register Algorithms
        #    In a full version, we'd dynamically load these based on the problem class.
        #    For now, we register our "universal" baselines + specific ones if needed.
        #    But our demo algorithms (Greedy/Random Vertex Cover) only work for 
        #    Minimum Vertex Cover. 
        #    
        #    TODO: Add a proper registry or factory.
        #    For this step, we'll just register the VC algos if the problem is MVC,
        #    or generic ones if we had them.
        
        if config.problem_class == "minimum_vertex_cover":
            runner.register_algorithm(GreedyVertexCover())
            runner.register_algorithm(RandomVertexCover())
        elif config.problem_class == "maximum_cut":
            # Use the new RandomMaxCut
            runner.register_algorithm(RandomMaxCut())
        else:
             logging.warning(f"Unknown problem class {config.problem_class}, attempting with available algos.")
             runner.register_algorithm(GreedyVertexCover())
             runner.register_algorithm(RandomVertexCover())

        # 5. Run Benchmark
        df = runner.run()
        
        # 6. Transform Results
        response = transform_results(df)
        return response

    except Exception as e:
        logging.error(f"Error running benchmark: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
