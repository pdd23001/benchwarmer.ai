
import logging
import sys
import pandas as pd
import tempfile
import os
import re
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
import httpx

from benchwarmer.algorithms.base import AlgorithmWrapper
from benchwarmer.config import BenchmarkConfig, GeneratorConfig, InstanceConfig
from benchwarmer.engine.runner import BenchmarkRunner
from benchwarmer.agents.intake import IntakeAgent
from benchwarmer.agents.implementation import ImplementationAgent
from benchwarmer.agents.backends import OpenAIBackend
from benchwarmer.engine.modal_runner import ModalRunner
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

app = FastAPI()

# Edge Nemotron defaults (DGX Spark / Ollama-compatible API).
NEMOTRON_BASE_URL = os.environ.get("NEMOTRON_BASE_URL", "http://10.19.177.52:11434/api")
NEMOTRON_MODEL = os.environ.get(
    "NEMOTRON_MODEL",
    "hf.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q4_K_M",
)
NEMOTRON_API_KEY = (
    os.environ.get("NEMOTRON_API_KEY")
    or os.environ.get("NVIDIA_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_PY_UPLOAD_BYTES = 512 * 1024        # 512 KB
MAX_PDF_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

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
            stale_connections: list[WebSocket] = []
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logging.warning(f"Failed to send to websocket: {e}")
                    stale_connections.append(connection)
            for connection in stale_connections:
                self.disconnect(connection, session_id)

ws_manager = ConnectionManager()


def _validate_upload(
    file_name: str,
    content: bytes,
    allowed_suffixes: set[str],
    max_bytes: int,
    label: str,
) -> None:
    if not file_name:
        raise HTTPException(status_code=400, detail=f"{label} filename is required")
    suffix = Path(file_name).suffix.lower()
    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid {label} file extension '{suffix or '(none)'}'. "
                f"Allowed: {', '.join(sorted(allowed_suffixes))}"
            ),
        )
    if not content:
        raise HTTPException(status_code=400, detail=f"{label} file is empty")
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"{label} file too large ({len(content)} bytes). "
                f"Max allowed: {max_bytes} bytes"
            ),
        )


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from plain text or fenced code."""
    if not text:
        return {}
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence_match.group(1) if fence_match else None
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}
    return {}


def _extract_pdf_text(pdf_path: str, max_chars: int = 24000) -> str:
    """
    Best-effort PDF text extraction for ingestion prompts.
    Falls back to byte-decoding if PDF parsing libraries are unavailable.
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        chunks: list[str] = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                chunks.append(txt)
            if sum(len(c) for c in chunks) >= max_chars:
                break
        joined = "\n\n".join(chunks)
        if joined.strip():
            return joined[:max_chars]
    except Exception as e:
        logging.warning("PDF text extraction fallback for %s: %s", pdf_path, e)

    # Fallback: decode raw bytes (noisy but better than empty input).
    with open(pdf_path, "rb") as f:
        raw = f.read()
    return raw.decode("latin-1", errors="ignore")[:max_chars]


async def _broadcast_code_preview(
    session_id: str,
    challenger_id: str,
    challenger_name: str,
    code: str,
) -> None:
    """
    Send a readable code preview to the challenger terminal without flooding WS.
    """
    if not code:
        return
    trimmed = code[:6000]
    chunk_size = 1500
    chunks = [trimmed[i:i + chunk_size] for i in range(0, len(trimmed), chunk_size)]
    total = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        await ws_manager.broadcast(session_id, {
            "type": "log",
            "source": "ImplementationAgent",
            "challenger_id": challenger_id,
            "message": (
                f"Generated code for {challenger_name} "
                f"(part {idx}/{total}):\n{chunk}"
            ),
        })


def _run_nemotron_ingestion(
    pdf_text: str,
    challenger_name: str,
    problem_hint: str,
) -> dict[str, Any]:
    """Use Nemotron (OpenAI-compatible API) to extract implementation-ready summary."""
    # Path 1: DGX Spark / Ollama-style endpoint (no API key required).
    if "11434" in NEMOTRON_BASE_URL or NEMOTRON_BASE_URL.rstrip("/").endswith("/api"):
        base = NEMOTRON_BASE_URL.rstrip("/")
        chat_url = f"{base}/chat" if base.endswith("/api") else f"{base}/api/chat"
        prompt = (
            f"Paper name: {challenger_name}\n"
            f"Problem hint: {problem_hint or 'unknown'}\n\n"
            "From this paper text, extract:\n"
            "- algorithm_name\n"
            "- objective\n"
            "- key_steps (ordered list)\n"
            "- data_structures\n"
            "- parameters\n"
            "- complexity\n"
            "- implementation_notes\n"
            "- assumptions\n\n"
            "Respond as strict JSON object with exactly those keys.\n\n"
            "Paper text:\n"
            f"{pdf_text}"
        )
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                chat_url,
                json={
                    "model": NEMOTRON_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an ingestion agent for algorithm papers. "
                                "Extract actionable implementation details for another coding model. "
                                "Return JSON only."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "format": "json",
                },
            )
            resp.raise_for_status()
            payload = resp.json()
        text = (payload.get("message", {}) or {}).get("content", "")
        parsed = _extract_json_object(text)
        if not parsed:
            raise ValueError("Nemotron edge ingestion returned non-JSON output")
        return parsed

    # Path 2: OpenAI-compatible hosted endpoint.
    backend = OpenAIBackend(
        base_url=NEMOTRON_BASE_URL,
        model=NEMOTRON_MODEL,
        api_key=NEMOTRON_API_KEY,
    )
    system = (
        "You are an ingestion agent for algorithm papers. "
        "Extract actionable implementation details for another coding model. "
        "Return JSON only."
    )
    prompt = (
        f"Paper name: {challenger_name}\n"
        f"Problem hint: {problem_hint or 'unknown'}\n\n"
        "From this paper text, extract:\n"
        "- algorithm_name\n"
        "- objective\n"
        "- key_steps (ordered list)\n"
        "- data_structures\n"
        "- parameters\n"
        "- complexity\n"
        "- implementation_notes\n"
        "- assumptions\n\n"
        "Respond as strict JSON object with exactly those keys.\n\n"
        "Paper text:\n"
        f"{pdf_text}"
    )
    resp = backend.generate(
        messages=[{"role": "user", "content": prompt}],
        system=system,
        tools=None,
        max_tokens=1400,
    )
    text = "\n".join(
        block.text for block in resp.content if getattr(block, "type", "") == "text"
    )
    parsed = _extract_json_object(text)
    if not parsed:
        raise ValueError("Nemotron ingestion returned non-JSON output")
    return parsed


def _estimate_total_jobs(config: BenchmarkConfig, algorithm_count: int) -> int:
    generated_instances = sum(
        len(gen.sizes) * gen.count_per_size
        for gen in config.instance_config.generators
    )
    total_instances = generated_instances + len(config.instance_config.custom_instances)
    return max(1, algorithm_count) * total_instances * config.execution_config.runs_per_config


def _apply_runtime_budget(
    config: BenchmarkConfig,
    execution_mode: str,
    expected_algorithms: int,
) -> tuple[BenchmarkConfig, str | None]:
    """
    Keep demo runs responsive by downscaling oversized benchmark configs.
    This avoids the long post-implementation bottleneck before charts appear.
    """
    # Keep local runs snappy for live demo UX; modal can handle more work.
    budget = 180 if execution_mode == "modal" else 48
    current_jobs = _estimate_total_jobs(config, expected_algorithms)
    if current_jobs <= budget:
        return config, None

    optimized = config.model_copy(deep=True)
    # Tighten dimensions in order: runs_per_config -> count_per_size -> sizes.
    while _estimate_total_jobs(optimized, expected_algorithms) > budget:
        changed = False
        ec = optimized.execution_config
        if ec.runs_per_config > 1:
            ec.runs_per_config -= 1
            changed = True
        else:
            for gen in optimized.instance_config.generators:
                if gen.count_per_size > 1:
                    gen.count_per_size -= 1
                    changed = True
                    break
            if not changed:
                for gen in optimized.instance_config.generators:
                    if len(gen.sizes) > 1:
                        gen.sizes = gen.sizes[: max(1, len(gen.sizes) - 1)]
                        changed = True
                        break
        if not changed:
            break

    new_jobs = _estimate_total_jobs(optimized, expected_algorithms)
    message = (
        f"Runtime budget applied: estimated jobs reduced from {current_jobs} to {new_jobs} "
        f"for faster live demo feedback."
    )
    return optimized, message

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

    content = await file.read()
    _validate_upload(
        file_name=file.filename or "",
        content=content,
        allowed_suffixes={".py"},
        max_bytes=MAX_PY_UPLOAD_BYTES,
        label="Python",
    )

    # Save user file
    suffix = Path(file.filename).suffix or ".py"
    user_path = os.path.join(session["base_dir"], f"user_algo{suffix}")

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

        content = await file.read()
        _validate_upload(
            file_name=file.filename or "",
            content=content,
            allowed_suffixes={".pdf"},
            max_bytes=MAX_PDF_UPLOAD_BYTES,
            label="PDF",
        )

        suffix = Path(file.filename).suffix or ".pdf"
        pdf_path = os.path.join(session["base_dir"], f"paper_{uuid.uuid4()}{suffix}")

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
    """Background task to ingest PDF and prepare implementation context."""
    challenger = None
    try:
        session = session_manager.get_session(session_id)
        challenger = session_manager.get_challenger(session_id, challenger_id)
        if not challenger:
            raise ValueError("Challenger not found")
        
        await ws_manager.broadcast(session_id, {
            "type": "challenger_update",
            "challenger_id": challenger_id,
            "status": "analyzing",
            "message": "Reading PDF content..."
        })

        loop = asyncio.get_event_loop()
        pdf_text = await loop.run_in_executor(None, lambda: _extract_pdf_text(challenger["path"]))
        problem_hint = session.get("problem_class") or "unknown"

        ingestion_source = "none"
        ingestion_payload: dict[str, Any] = {}

        # Primary ingestion path: Nemotron (OpenAI-compatible hosted or edge server).
        if NEMOTRON_BASE_URL:
            await ws_manager.broadcast(session_id, {
                "type": "log",
                "source": "IngestionAgent",
                "challenger_id": challenger_id,
                "message": (
                    f"Nemotron ingestion starting for {challenger['name']} "
                    f"(endpoint: {NEMOTRON_BASE_URL})..."
                ),
            })
            try:
                ingestion_payload = await loop.run_in_executor(
                    None,
                    lambda: _run_nemotron_ingestion(
                        pdf_text=pdf_text,
                        challenger_name=challenger["name"],
                        problem_hint=problem_hint,
                    ),
                )
                ingestion_source = "nemotron"
            except Exception as e:
                await ws_manager.broadcast(session_id, {
                    "type": "log",
                    "source": "IngestionAgent",
                    "challenger_id": challenger_id,
                    "message": f"Nemotron ingestion failed, using fallback context: {e}",
                })

        if not ingestion_payload:
            ingestion_source = "fallback"
            ingestion_payload = {
                "algorithm_name": challenger["name"],
                "objective": f"Optimize {problem_hint}",
                "key_steps": ["Extract core heuristic from paper text", "Implement solve() for problem class"],
                "data_structures": [],
                "parameters": {},
                "complexity": "unknown",
                "implementation_notes": pdf_text[:3000],
                "assumptions": ["Paper text may be incomplete"],
            }

        challenger["ingestion"] = {
            "source": ingestion_source,
            "summary": ingestion_payload,
        }

        summary_text = json.dumps(ingestion_payload, indent=2)
        challenger["ingestion_context"] = (
            f"Ingestion source: {ingestion_source}\n"
            f"Structured summary:\n{summary_text}"
        )

        await ws_manager.broadcast(session_id, {
            "type": "challenger_update",
            "challenger_id": challenger_id,
            "status": "ready",
            "message": f"Analysis complete via {ingestion_source}. Ready to implement."
        })
        
        challenger["status"] = "ready"
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        if challenger is not None:
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
    
    asyncio.create_task(safe_execute_session_benchmark(session_id, request))
    
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


async def safe_execute_session_benchmark(session_id: str, request: RunSessionRequest):
    """Guarded background runner so websocket clients always receive terminal error status."""
    try:
        await execute_session_benchmark(session_id, request)
    except Exception as e:
        logging.error("Benchmark task failed: %s", e, exc_info=True)
        await ws_manager.broadcast(session_id, {
            "type": "error",
            "message": f"Benchmark failed: {e}",
        })
        await ws_manager.broadcast(session_id, {"type": "status", "status": "error"})

async def execute_session_benchmark(session_id: str, request: RunSessionRequest):
    """Orchestrates the parallel execution of all agents/algorithms"""
    session = session_manager.get_session(session_id)
    base_config = session["config"]
    expected_algorithms = (1 if session.get("user_algo_path") else 0) + len(session["challengers"])
    config, budget_note = _apply_runtime_budget(
        base_config,
        request.execution_mode,
        expected_algorithms=max(1, expected_algorithms),
    )
    if budget_note:
        await ws_manager.broadcast(session_id, {
            "type": "log",
            "source": "System",
            "message": budget_note,
        })
    
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
    modal_runner_logger = logging.getLogger("benchwarmer.engine.modal_runner")
    ws_handler = WebSocketLogHandler(session_id, ws_manager, asyncio.get_running_loop())
    ws_handler.setLevel(logging.INFO)
    runner_logger.addHandler(ws_handler)
    modal_runner_logger.addHandler(ws_handler)
    
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
            
    # Challengers: baselines first (cheap), then PDF implementations in parallel.
    pdf_challengers: list[dict[str, Any]] = []
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
            pdf_challengers.append(c)

    async def implement_pdf_challenger(challenger: dict[str, Any]) -> None:
        await ws_manager.broadcast(session_id, {
            "type": "challenger_update",
            "challenger_id": challenger["id"],
            "status": "implementing",
            "message": f"Generating implementation for {challenger['name']}...",
        })
        try:
            impl_agent = ImplementationAgent()
            loop = asyncio.get_event_loop()
            nemotron_context = challenger.get("ingestion_context")
            result = await loop.run_in_executor(
                None,
                lambda: impl_agent.generate(
                    description=(
                        "Implement the algorithm described in the provided paper. "
                        "Use the ingestion summary as the primary guide."
                    ),
                    problem_class=config.problem_class,
                    additional_context=nemotron_context,
                    pdf_paths=[challenger["path"]],
                    max_retries=2,
                ),
            )

            if result["success"]:
                challenger["implementation"] = result["algorithm"]
                runner.register_algorithm(challenger["implementation"])
                await ws_manager.broadcast(session_id, {
                    "type": "challenger_update",
                    "challenger_id": challenger["id"],
                    "status": "testing",
                    "message": "Running smoke validation on generated implementation...",
                })
                smoke_result = result.get("smoke_result", {})
                if smoke_result:
                    await ws_manager.broadcast(session_id, {
                        "type": "log",
                        "source": "ImplementationAgent",
                        "challenger_id": challenger["id"],
                        "message": f"Smoke test result: {json.dumps(smoke_result)}",
                    })
                await ws_manager.broadcast(session_id, {
                    "type": "challenger_update",
                    "challenger_id": challenger["id"],
                    "status": "ready",
                    "message": f"Implementation successful: {result['name']}",
                })
                await _broadcast_code_preview(
                    session_id=session_id,
                    challenger_id=challenger["id"],
                    challenger_name=challenger["name"],
                    code=result.get("code", ""),
                )
            else:
                await ws_manager.broadcast(session_id, {
                    "type": "challenger_error",
                    "challenger_id": challenger["id"],
                    "message": f"Generation failed: {result['error']}",
                })
        except Exception as e:
            logging.error("Implementation failed for %s: %s", challenger.get("name"), e, exc_info=True)
            await ws_manager.broadcast(session_id, {
                "type": "challenger_error",
                "challenger_id": challenger["id"],
                "message": f"Agent error: {str(e)}",
            })

    if pdf_challengers:
        await asyncio.gather(
            *(implement_pdf_challenger(c) for c in pdf_challengers),
            return_exceptions=True,
        )

    # 3. Run Benchmark (Custom Loop to stream events)
    # We need to monkey-patch or subclass Runner to emit events?
    # Or just iterate manually.
    
    await ws_manager.broadcast(session_id, {"type": "status", "status": "benchmarking"})
    
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
        modal_runner_logger.removeHandler(ws_handler)
    
    await ws_manager.broadcast(session_id, {"type": "status", "status": "aggregating"})

    # Surface per-algorithm failures without failing entire run.
    try:
        error_rows = df[df["status"] != "success"]
        if not error_rows.empty:
            grouped = (
                error_rows.groupby(["algorithm_name", "status"])
                .size()
                .reset_index(name="count")
            )
            for _, row in grouped.iterrows():
                await ws_manager.broadcast(session_id, {
                    "type": "log",
                    "source": "Runner",
                    "message": (
                        f"Algorithm '{row['algorithm_name']}' had {int(row['count'])} "
                        f"{row['status']} run(s)"
                    ),
                })
    except Exception:
        logging.exception("Failed to summarize algorithm errors for websocket output")

    # Transform and send results (or explicit error status).
    try:
        response = transform_results(df, config.problem_class)
        await ws_manager.broadcast(session_id, {
            "type": "result",
            "data": response.model_dump()
        })
        await ws_manager.broadcast(session_id, {"type": "status", "status": "completed"})
    except HTTPException as e:
        await ws_manager.broadcast(session_id, {
            "type": "error",
            "message": e.detail if hasattr(e, "detail") else str(e),
        })
        await ws_manager.broadcast(session_id, {"type": "status", "status": "error"})


@app.websocket("/api/session/{session_id}/live")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket for live terminal streaming"""
    await ws_manager.connect(websocket, session_id)
    await ws_manager.broadcast(session_id, {"type": "status", "status": "connected"})
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
