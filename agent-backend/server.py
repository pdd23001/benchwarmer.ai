
import logging
import sys
import pandas as pd
import tempfile
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Ensure we can import from benchwarmer package
sys.path.insert(0, ".")

from benchwarmer.algorithms.base import AlgorithmWrapper
from benchwarmer.config import BenchmarkConfig, GeneratorConfig, InstanceConfig
from benchwarmer.engine.runner import BenchmarkRunner
from benchwarmer.agents.intake import IntakeAgent
from benchwarmer.agents.implementation import ImplementationAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def transform_results(df: pd.DataFrame, problem_class: str = "unknown") -> BenchmarkResponse:
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

    # Problem-specific labels
    labels = {
        "maximum_cut": {
            "title": "Maximum Cut Algorithm Comparison",
            "ylabel": "Cut Value (edges crossing partition)",
        },
        "minimum_vertex_cover": {
            "title": "Vertex Cover Algorithm Comparison",
            "ylabel": "Vertex Cover Size",
        },
    }

    problem_labels = labels.get(problem_class, {
        "title": f"{problem_class.replace('_', ' ').title()} Comparison",
        "ylabel": "Objective Value",
    })

    return BenchmarkResponse(
        title=problem_labels["title"],
        xLabel="Graph Size (Nodes)",
        yLabel=problem_labels["ylabel"],
        series=series,
        data=data
    )

# ─── Endpoints ────────────────────────────────────────────────────────────────

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/api/benchmark", response_model=BenchmarkResponse)
async def run_benchmark_endpoint(
    query: str = Form(...),
    algorithm_description: Optional[str] = Form(None),
    pdfs: List[UploadFile] = File(default=[]),
    py_files: List[UploadFile] = File(default=[]),
):
    """
    Run a benchmark with optional PDF papers and Python algorithm files.

    Parameters:
    - query: Natural language problem description
    - algorithm_description: Optional description for PDF-based generation
    - pdfs: Optional list of PDF files (research papers to generate algorithms from)
    - py_files: Optional list of .py files (your own algorithm implementations)
    """
    pdf_paths = []
    py_paths = []

    try:
        logging.info(f"Processing query: {query!r}")
        logging.info(f"Received {len(pdfs)} PDF files")
        logging.info(f"Received {len(py_files)} Python files")

        # Save uploaded PDFs to temporary files
        for pdf_file in pdfs:
            if pdf_file.filename:
                # Create temp file with original extension
                suffix = Path(pdf_file.filename).suffix or ".pdf"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    content = await pdf_file.read()
                    tmp.write(content)
                    pdf_paths.append(tmp.name)
                    logging.info(f"Saved PDF: {pdf_file.filename} -> {tmp.name}")

        # Save uploaded Python files to temporary files
        for py_file in py_files:
            if py_file.filename:
                suffix = Path(py_file.filename).suffix or ".py"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='w') as tmp:
                    content = await py_file.read()
                    tmp.write(content.decode('utf-8'))
                    tmp.flush()
                    py_paths.append(tmp.name)
                    logging.info(f"Saved Python file: {py_file.filename} -> {tmp.name}")

        # 1. Initialize Intake Agent
        intake_agent = IntakeAgent()

        # 2. Run Intake Agent (non-interactive mode for API)
        # Add explicit instruction to not ask questions
        enhanced_query = f"{query}\n\n[SYSTEM: This is an API request. Do NOT ask clarifying questions. Use sensible defaults for any ambiguous aspects. Proceed directly to generating the benchmark configuration.]"

        config = intake_agent.run(enhanced_query, interactive=False)
        logging.info(f"Generated config: {config}")

        # 3. Initialize Runner
        runner = BenchmarkRunner(config)

        # 4. Register Algorithms

        # 4a. If Python files provided, load user's algorithms
        if py_paths:
            from benchwarmer.utils.algorithm_sandbox import execute_algorithm_code

            for py_path in py_paths:
                logging.info(f"Loading user algorithm from {py_path}...")
                try:
                    with open(py_path, 'r') as f:
                        user_code = f.read()

                    result = execute_algorithm_code(user_code, config.problem_class)

                    if result["success"]:
                        logging.info(f"Successfully loaded user algorithm: {result['name']}")
                        runner.register_algorithm(result["algorithm"])
                    else:
                        logging.error(f"Failed to load user algorithm: {result['error']}")
                        if "traceback" in result and result["traceback"]:
                            logging.error(f"Traceback: {result['traceback']}")
                        # Continue with other algorithms
                except Exception as e:
                    logging.error(f"Error loading Python file: {e}", exc_info=True)

        # 4b. Add baseline algorithms only if no user algorithms provided
        if not py_paths:
            logging.info("No user algorithms provided, adding baseline algorithms...")
            if config.problem_class == "minimum_vertex_cover":
                runner.register_algorithm(GreedyVertexCover())
                runner.register_algorithm(RandomVertexCover())
            elif config.problem_class == "maximum_cut":
                runner.register_algorithm(RandomMaxCut())
            else:
                logging.warning(f"Unknown problem class {config.problem_class}")
                runner.register_algorithm(RandomVertexCover())

        # 4c. If PDFs provided, generate custom algorithm using Implementation Agent
        if pdf_paths:
            logging.info(f"Generating custom algorithm from {len(pdf_paths)} PDFs...")
            impl_agent = ImplementationAgent()

            algo_desc = algorithm_description or "Implement the algorithm described in the provided papers"

            result = impl_agent.generate(
                description=algo_desc,
                problem_class=config.problem_class,
                pdf_paths=pdf_paths,
                max_retries=2,
            )

            if result["success"]:
                logging.info(f"Successfully generated algorithm: {result['name']}")
                runner.register_algorithm(result["algorithm"])
            else:
                logging.error(f"Failed to generate algorithm: {result['error']}")
                if "traceback" in result and result["traceback"]:
                    logging.error(f"Traceback: {result['traceback']}")
                # Continue with baseline algorithms only

        # 5. Run Benchmark
        logging.info("Starting benchmark execution...")
        df = runner.run()
        logging.info(f"Benchmark complete. Generated {len(df)} result rows.")

        # 6. Transform Results
        logging.info("Transforming results for frontend...")
        response = transform_results(df, problem_class=config.problem_class)
        logging.info("Results transformed successfully.")
        return response

    except Exception as e:
        logging.error(f"Error running benchmark: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary PDF files
        for path in pdf_paths:
            try:
                os.unlink(path)
                logging.info(f"Cleaned up temp file: {path}")
            except Exception as e:
                logging.warning(f"Failed to delete temp file {path}: {e}")

        # Clean up temporary Python files
        for path in py_paths:
            try:
                os.unlink(path)
                logging.info(f"Cleaned up temp file: {path}")
            except Exception as e:
                logging.warning(f"Failed to delete temp file {path}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
