
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
import asyncio
import uuid
import json
import time
import subprocess
from collections import defaultdict
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from benchwarmer.algorithms.base import AlgorithmWrapper
from benchwarmer.config import BenchmarkConfig, GeneratorConfig, InstanceConfig
from benchwarmer.engine.runner import BenchmarkRunner
from benchwarmer.agents.intake import IntakeAgent
from benchwarmer.agents.implementation import ImplementationAgent
from benchwarmer.engine.modal_runner import ModalRunner
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

# ─── Session & WebSocket Management ──────────────────────────────────────────

class SessionManager:
    def __init__(self):
        # session_id -> { "files": [], "algorithms": [], "config": None, "logs": [] }
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": time.time(),
            "user_algo_path": None,
            "user_algo_name": None,
            "problem_class": None,
            "challengers": [],  # List of {id, type, status, path, name}
            "config": None,
            "base_dir": tempfile.mkdtemp(prefix=f"benchwarmer_{session_id}_")
        }
        return session_id

    def get_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return self.sessions[session_id]

    def add_challenger(self, session_id: str, type: str, path: str, name: str) -> str:
        session = self.get_session(session_id)
        challenger_id = str(uuid.uuid4())
        session["challengers"].append({
            "id": challenger_id,
            "type": type,        # "pdf", "text", "baseline"
            "status": "pending", # "analyzing", "ready", "error"
            "path": path,
            "name": name,
            "implementation": None # Will hold the AlgorithmWrapper instance later
        })
        return challenger_id

    def get_challenger(self, session_id: str, challenger_id: str):
        session = self.get_session(session_id)
        for c in session["challengers"]:
            if c["id"] == challenger_id:
                return c
        return None

session_manager = SessionManager()

class ConnectionManager:
    def __init__(self):
        # session_id -> list of WebSockets
        self.active_connections: Dict[str, List[WebSocket]] = defaultdict(list)

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id].append(websocket)

    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)

    async def broadcast(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logging.warning(f"Failed to send to websocket: {e}")

ws_manager = ConnectionManager()

# Custom Log Handler for WebSockets
class WebSocketLogHandler(logging.Handler):
    def __init__(self, session_id: str, ws_manager, loop):
        super().__init__()
        self.session_id = session_id
        self.ws_manager = ws_manager
        self.loop = loop

    def emit(self, record):
        try:
            msg = self.format(record)
            # Schedule the broadcast in the main event loop
            asyncio.run_coroutine_threadsafe(
                self.ws_manager.broadcast(self.session_id, {
                    "type": "log",
                    "source": "Runner",
                    "message": msg
                }),
                self.loop
            )
        except Exception:
            self.handleError(record)

# ─── Data Models ──────────────────────────────────────────────────────────────

class StartSessionResponse(BaseModel):
    session_id: str
    detected_class: Optional[str]
    filename: str

class ChallengerResponse(BaseModel):
    challenger_id: str
    status: str
    message: str

class ConfigureResponse(BaseModel):
    config: Dict[str, Any]
    message: str

class RunSessionRequest(BaseModel):
    execution_mode: str = "local"  # "local" or "modal"
    modal_token_id: Optional[str] = None
    modal_token_secret: Optional[str] = None

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

# ─── New API Endpoints ──────────────────────────────────────────────────────

@app.post("/api/session/start", response_model=StartSessionResponse)
async def start_session(file: UploadFile = File(...)):
    """Step 1: Upload User Algorithm"""
    session_id = session_manager.create_session()
    session = session_manager.get_session(session_id)
    
    # Save user file
    suffix = Path(file.filename).suffix or ".py"
    user_path = os.path.join(session["base_dir"], f"user_algo{suffix}")
    
    content = await file.read()
    with open(user_path, "wb") as f:
        f.write(content)
        
    session["user_algo_path"] = user_path
    session["user_algo_name"] = file.filename
    
    # Quick analysis to guess problem class (naive regex for now)
    # in the future we can use an LLM or AST
    code_str = content.decode("utf-8", errors="ignore")
    detected_class = "unknown"
    if "partition" in code_str or "cut" in code_str.lower():
        detected_class = "maximum_cut"
    elif "cover" in code_str or "vertex" in code_str.lower():
        detected_class = "minimum_vertex_cover"
        
    session["problem_class"] = detected_class
    
    logging.info(f"Session {session_id} started with {file.filename} (detected: {detected_class})")
    
    return StartSessionResponse(
        session_id=session_id,
        detected_class=detected_class,
        filename=file.filename
    )

@app.post("/api/session/{session_id}/challenger", response_model=ChallengerResponse)
async def add_challenger(
    session_id: str,
    type: str = Form(...), # "pdf" or "baseline"
    file: Optional[UploadFile] = File(None)
):
    """Step 2: Add a Challenger (PDF or Baseline)"""
    session = session_manager.get_session(session_id)
    
    if type == "pdf":
        if not file:
            raise HTTPException(status_code=400, detail="PDF file required")
            
        suffix = Path(file.filename).suffix or ".pdf"
        pdf_path = os.path.join(session["base_dir"], f"paper_{uuid.uuid4()}{suffix}")
        
        content = await file.read()
        with open(pdf_path, "wb") as f:
            f.write(content)
            
        cid = session_manager.add_challenger(session_id, "pdf", pdf_path, file.filename)
        
        # Trigger background analysis (using asyncio task to not block)
        asyncio.create_task(analyze_challenger(session_id, cid))
        
        return ChallengerResponse(challenger_id=cid, status="analyzing", message="Analyzing PDF...")
        
    elif type == "baseline":
        # Add a baseline like Random or Greedy
        name = "Baseline (Random)"
        cid = session_manager.add_challenger(session_id, "baseline", "builtin", name)
        
        # Mark ready immediately
        c = session_manager.get_challenger(session_id, cid)
        c["status"] = "ready"
        
        return ChallengerResponse(challenger_id=cid, status="ready", message="Baseline added")
        
    else:
        raise HTTPException(status_code=400, detail=f"Unknown type: {type}")

async def analyze_challenger(session_id: str, challenger_id: str):
    """Background task to analyze PDF and generate implementation plan"""
    try:
        session = session_manager.get_session(session_id)
        challenger = session_manager.get_challenger(session_id, challenger_id)
        
        await ws_manager.broadcast(session_id, {
            "type": "challenger_update",
            "challenger_id": challenger_id,
            "status": "analyzing",
            "message": "Reading PDF content..."
        })
        
        # Simulate or call actual agent (TODO: Call ImplementationAgent here)
        # For v0 demo, we might want to just set it to "ready" or mock the "thinking"
        # provided we have the code. 
        # BUT the plan is to stream LLM tokens. 
        # For this step, we just prepare it. The actual verifiable code generation happens
        # in the execution phase or we can do it now.
        # User said: "It can read PDFs and extract algorithms."
        # Let's do it now so it's ready for the race.
        
        await ws_manager.broadcast(session_id, {
            "type": "challenger_update", # Frontend should show spinning loader
            "challenger_id": challenger_id,
            "status": "ready",
            "message": "Analysis complete. Ready to implement."
        })
        
        challenger["status"] = "ready"
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        challenger["status"] = "error"
        await ws_manager.broadcast(session_id, {
            "type": "challenger_error",
            "challenger_id": challenger_id,
            "message": str(e)
        })

@app.post("/api/session/{session_id}/configure", response_model=ConfigureResponse)
async def configure_session(session_id: str, preferences: str = Form(...)):
    """Step 3: Configure Benchmark Environment (Intake Agent)"""
    session = session_manager.get_session(session_id)
    
    # Run Intake Agent
    # For now, we wrap it to be async-ish (it's sync under the hood)
    intake_agent = IntakeAgent()
    
    # We might want to inject knowledge about the problem class if we detected it
    context = ""
    if session["problem_class"]:
        context = f"The user is solving {session['problem_class']}. "
        
    full_query = f"{context}{preferences}\n\n[SYSTEM: This is an API request. Generate config immediately.]"
    
    try:
        # Run in threadpool to not block event loop
        loop = asyncio.get_event_loop()
        config = await loop.run_in_executor(None, lambda: intake_agent.run(full_query, interactive=False))
        
        # Save config to session
        session["config"] = config
        
        return ConfigureResponse(config=config.model_dump(), message="Configuration generated")
    except Exception as e:
        logging.error(f"Intake agent failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/session/{session_id}/run", response_model=BenchmarkResponse)
async def run_session(session_id: str, request: RunSessionRequest = RunSessionRequest()):
    """Step 4: Execute Benchmark (Live)"""
    session = session_manager.get_session(session_id)
    if not session["config"]:
        raise HTTPException(status_code=400, detail="Session not configured")
        
    config = session["config"]
    
    # Background this so we can return immediately? 
    # Actually, user might want to await? 
    # No, with WebSockets, we should probably return "started" and stream the rest.
    # checking the return type... BenchmarkResponse. 
    # If the frontend expects a response, we must wait. 
    # BUT user wants "Live Terminal".
    # So we should probably return a "Job Started" response and let WS handle the data.
    # However, existing frontend expects BenchmarkResponse. 
    # We are changing the frontend anyway. 
    # Let's change this endpoint to return { "status": "started" } and stream results via WS.
    
    asyncio.create_task(execute_session_benchmark(session_id, request))
    
    # Return empty/partial response or change the model. 
    # For now, let's keep the model but return empty data, frontend will ignore it 
    # if it's listening to WS.
    return BenchmarkResponse(
        title="Benchmark Running...",
        xLabel="Size",
        yLabel="Value",
        series=[],
        data=[]
    )

async def execute_session_benchmark(session_id: str, request: RunSessionRequest):
    """Orchestrates the parallel execution of all agents/algorithms"""
    session = session_manager.get_session(session_id)
    config = session["config"]
    
    # 1. Implementation Phase (Parallel)
    # For each PDF challenger, we need to generate code
    
    # 2. Register Algorithms
    if request.execution_mode == "modal":
        runner = ModalRunner(
            config,
            modal_token_id=request.modal_token_id,
            modal_token_secret=request.modal_token_secret
        )
    else:
        runner = BenchmarkRunner(config)
    
    # Attach WebSocket Logger to the runner
    runner_logger = logging.getLogger("benchwarmer.engine.runner")
    ws_handler = WebSocketLogHandler(session_id, ws_manager, asyncio.get_running_loop())
    ws_handler.setLevel(logging.INFO)
    runner_logger.addHandler(ws_handler)
    
    # User Algo
    if session["user_algo_path"]:
        from benchwarmer.utils.algorithm_sandbox import execute_algorithm_code
        try:
            with open(session["user_algo_path"], 'r') as f:
                code = f.read()
            res = execute_algorithm_code(code, config.problem_class)
            if res["success"]:
                runner.register_algorithm(res["algorithm"])
                await ws_manager.broadcast(session_id, {
                    "type": "log",
                    "source": "System",
                    "message": f"Loaded user algorithm: {res['name']}"
                })
            else:
                await ws_manager.broadcast(session_id, {
                    "type": "error",
                    "message": f"Failed to load user algo: {res.get('error')} {res.get('traceback', '')}"
                })
        except Exception as e:
            await ws_manager.broadcast(session_id, {
                "type": "error",
                "message": f"Failed to load user algo: {e}"
            })
            
    # Challengers
    for c in session["challengers"]:
        if c["type"] == "baseline":
            # Add baseline
            if config.problem_class == "minimum_vertex_cover":
                runner.register_algorithm(GreedyVertexCover())
            else:
                 runner.register_algorithm(RandomMaxCut())
            await ws_manager.broadcast(session_id, {
                "type": "log", 
                "source": "System",
                "message": f"Added baseline: {c['name']}"
            })
            
        elif c["type"] == "pdf":
            # Generate code using ImplementationAgent
            await ws_manager.broadcast(session_id, {
                "type": "challenger_update",
                "challenger_id": c["id"],
                "status": "implementing",
                "message": f"Generating implementation for {c['name']}..."
            })
            
            try:
                impl_agent = ImplementationAgent()
                # Run in threadpool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: impl_agent.generate(
                    description="Implement the algorithm described in the provided papers",
                    problem_class=config.problem_class,
                    pdf_paths=[c["path"]],
                    max_retries=2
                ))
                
                if result["success"]:
                    c["implementation"] = result["algorithm"]
                    runner.register_algorithm(c["implementation"])
                    
                    await ws_manager.broadcast(session_id, {
                        "type": "challenger_update",
                        "challenger_id": c["id"],
                        "status": "ready",
                        "message": f"Implementation successful: {result['name']}"
                    })
                    
                    await ws_manager.broadcast(session_id, {
                        "type": "log",
                        "source": "ImplementationAgent",
                        "challenger_id": c["id"],
                        "message": f"Generated code for {c['name']}:\n{result.get('code', '')[:200]}..."
                    })
                else:
                    await ws_manager.broadcast(session_id, {
                        "type": "challenger_error",
                        "challenger_id": c["id"],
                        "message": f"Generation failed: {result['error']}"
                    })
            except Exception as e:
                logging.error(f"Implementation failed: {e}")
                await ws_manager.broadcast(session_id, {
                    "type": "challenger_error",
                    "challenger_id": c["id"],
                    "message": f"Agent error: {str(e)}"
                })

    # 3. Run Benchmark (Custom Loop to stream events)
    # We need to monkey-patch or subclass Runner to emit events?
    # Or just iterate manually.
    
    await ws_manager.broadcast(session_id, {"type": "status", "status": "running"})
    
    # ... execution logic ...
    # For v0, let's just run it synchronously and emit the final result
    # We will refine 'live' streaming in the next iteration.
    
    try:
        if request.execution_mode == "modal":
            # ModalRunner is Async
            df = await runner.run()
        else:
            # BenchmarkRunner is Sync
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, runner.run)
    finally:
        # Cleanup logger
        runner_logger.removeHandler(ws_handler)
    
    # Transform and send results
    response = transform_results(df, config.problem_class)
    
    await ws_manager.broadcast(session_id, {
        "type": "result",
        "data": response.model_dump()
    })
    
    await ws_manager.broadcast(session_id, {"type": "status", "status": "complete"})


@app.websocket("/api/session/{session_id}/live")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket for live terminal streaming"""
    await ws_manager.connect(websocket, session_id)
    try:
        while True:
            # Keep connection alive, maybe receive commands (like "stop")
            data = await websocket.receive_text()
            # process client commands if any
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, session_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
