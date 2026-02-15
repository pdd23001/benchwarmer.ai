"""
Benchwarmer.AI — FastAPI backend.

Endpoints:
    POST /api/chat        SSE-streaming multi-turn orchestrator (primary)
    POST /api/benchmark   Legacy one-shot benchmark
"""

import asyncio
import base64
import json
import logging
import os
import sys
import tempfile
import threading
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Ensure we can import from benchwarmer package
sys.path.insert(0, ".")

from benchwarmer.algorithms.base import AlgorithmWrapper
from benchwarmer.config import BenchmarkConfig, GeneratorConfig, InstanceConfig
from benchwarmer.engine.runner import BenchmarkRunner
from benchwarmer.agents.intake import IntakeAgent
from benchwarmer.agents.orchestrator import OrchestratorAgent
from benchwarmer.utils.loader import load_algorithm_from_file
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Data Models ──────────────────────────────────────────────────────────────

class FileAttachment(BaseModel):
    filename: str
    content_base64: str  # base64-encoded bytes


class BenchmarkRequest(BaseModel):
    query: str
    execution_mode: str = "local"
    modal_token_id: Optional[str] = None
    modal_token_secret: Optional[str] = None


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


# Chat models (multi-turn orchestrator)
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    execution_mode: str = "local"  # local | modal (first turn)
    llm_backend: str = "claude"  # claude | nemotron (first turn)
    pdfs: Optional[List[FileAttachment]] = None
    custom_algorithm_file: Optional[FileAttachment] = None  # .py file


# ─── Toy Algorithms (Legacy) ─────────────────────────────────────────────────

class GreedyVertexCover(AlgorithmWrapper):
    name = "greedy_vc"

    def solve(self, instance: dict, timeout: float = 60.0) -> dict:
        covered: set[int] = set()
        cover: list[int] = []
        for edge in instance["edges"]:
            u, v = edge["source"], edge["target"]
            if u not in covered and v not in covered:
                cover.append(u)
                covered.add(u)
        for edge in instance["edges"]:
            u, v = edge["source"], edge["target"]
            if u not in covered and v not in covered:
                cover.append(u)
                covered.add(u)
        return {"solution": {"vertices": cover}, "metadata": {"strategy": "greedy"}}


class RandomVertexCover(AlgorithmWrapper):
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
        for edge in instance["edges"]:
            u, v = edge["source"], edge["target"]
            if u not in covered and v not in covered:
                cover.append(u)
                covered.add(u)
        return {"solution": {"vertices": cover}, "metadata": {"strategy": "random"}}


class RandomMaxCut(AlgorithmWrapper):
    name = "random_max_cut"

    def solve(self, instance: dict, timeout: float = 60.0) -> dict:
        import random
        n_nodes = len(instance["nodes"])
        partition = [random.choice([0, 1]) for _ in range(n_nodes)]
        return {"solution": {"partition": partition}, "metadata": {"strategy": "random"}}


# ─── Helper Functions ────────────────────────────────────────────────────────

def _json_default(obj: Any) -> Any:
    """Custom JSON serializer for numpy/pandas types."""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _safe_json(obj: Any) -> str:
    """JSON-encode with numpy/pandas safety."""
    return json.dumps(obj, default=_json_default, ensure_ascii=False)


def transform_results(df: pd.DataFrame) -> BenchmarkResponse:
    """Transforms the pandas DataFrame into the frontend's expected JSON format."""
    success_df = df[df["status"] == "success"].copy()

    if success_df.empty:
        raise HTTPException(status_code=500, detail="No successful benchmark runs generated.")

    agg = success_df.groupby(["algorithm_name", "problem_size"])["objective_value"].mean().reset_index()
    pivot = agg.pivot(index="problem_size", columns="algorithm_name", values="objective_value").reset_index()

    data = []
    for _, row in pivot.iterrows():
        item: Dict[str, Any] = {"x": int(row["problem_size"])}
        for col in pivot.columns:
            if col != "problem_size":
                val = row[col]
                if pd.notna(val):
                    item[col] = float(val)
        data.append(item)

    data.sort(key=lambda d: d["x"])

    algorithms = [c for c in pivot.columns if c != "problem_size"]
    colors = ["#22c55e", "#3b82f6", "#ef4444", "#f59e0b", "#a855f7", "#ec4899"]
    series = [
        SeriesData(name=algo, color=colors[i % len(colors)], dataKey=algo)
        for i, algo in enumerate(algorithms)
    ]

    return BenchmarkResponse(
        title="Benchmark Results",
        xLabel="Problem Size",
        yLabel="Objective Value",
        series=series,
        data=data,
    )


def _build_chart(df: pd.DataFrame) -> Optional[BenchmarkResponse]:
    """Build chart from results dataframe, or None on failure."""
    try:
        return transform_results(df)
    except Exception:
        return None


# ─── Session store for multi-turn chat ────────────────────────────────────────

_SESSIONS: Dict[str, Dict[str, Any]] = {}  # session_id -> {agent, messages}

from benchwarmer.database import init_db, save_message, get_messages

# Initialize DB on startup
@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/api/chat/{session_id}")
async def get_chat_history(session_id: str):
    """Retrieve chat history for a given session."""
    messages = get_messages(session_id)
    if not messages:
        return [] 
    return messages

from benchwarmer.database import get_all_sessions, delete_session, rename_session

@app.get("/api/sessions")
async def get_sessions():
    """Retrieve all chat sessions."""
    return get_all_sessions()

@app.delete("/api/sessions/{session_id}")
async def remove_session(session_id: str):
    """Delete a chat session and all its messages."""
    delete_session(session_id)
    _SESSIONS.pop(session_id, None)
    return {"ok": True}

@app.patch("/api/sessions/{session_id}")
async def update_session(session_id: str, body: dict):
    """Rename a chat session."""
    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title required")
    rename_session(session_id, title)
    return {"ok": True}

from benchwarmer.database import get_all_algorithms, get_algorithm_code

@app.get("/api/algorithms")
async def list_algorithms():
    """Retrieve all generated algorithms."""
    return get_all_algorithms()

@app.get("/api/algorithms/{name}")
async def get_algorithm(name: str):
    """Retrieve code for a specific algorithm."""
    code = get_algorithm_code(name)
    if not code:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    return {"name": name, "code": code}



# ─── Session processing status ────────────────────────────────────────────────

@app.get("/api/chat/{session_id}/status")
async def get_chat_status(session_id: str):
    """Check if the backend is still processing a turn for this session."""
    sess = _SESSIONS.get(session_id)
    if sess:
        return {"processing": sess.get("processing", False)}
    return {"processing": False}


# ─── SSE Chat Endpoint ───────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat_turn(request: ChatRequest):
    """
    Multi-turn chat endpoint — mirrors the CLI orchestrator flow.

    The orchestrator runs in a **background thread** that always saves
    its result to the DB — even if the SSE client disconnects mid-stream
    (e.g. browser refresh).  The SSE generator simply reads from the
    shared event queue and forwards events to the client.

    Returns a **Server-Sent Events** stream with real-time progress:
        data: {"type": "thinking"}
        data: {"type": "tool_start", "tool": "run_intake", ...}
        data: {"type": "tool_end",   "tool": "run_intake", "result": "..."}
        data: {"type": "done", "session_id": "...", "reply": "...", "chart": ...}
        data: {"type": "error", "error": "..."}
    """
    session_id = request.session_id
    agent: Optional[OrchestratorAgent] = None
    messages: list = []

    # ── Session setup (synchronous, before streaming) ──────────────────────
    if session_id and session_id in _SESSIONS:
        sess = _SESSIONS[session_id]
        # If session is already processing a turn, reject the new request
        if sess.get("processing", False):
            async def busy_stream():
                yield f"data: {_safe_json({'type': 'error', 'session_id': session_id, 'error': 'A turn is still being processed. Please wait.'})}\n\n"
            return StreamingResponse(
                busy_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        agent = sess["agent"]
        messages = sess["messages"]
    elif session_id and session_id not in _SESSIONS:
        # Session was lost (e.g. backend restarted). Tell the frontend.
        async def expired_stream():
            yield f"data: {_safe_json({'type': 'error', 'session_id': session_id, 'error': 'Session expired — the backend was restarted. Please start a new conversation.'})}\n\n"
        return StreamingResponse(
            expired_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        pdf_paths: list[str] = []
        custom_algo_path: Optional[str] = None

        if request.pdfs:
            tmpdir = tempfile.mkdtemp(prefix="bw_pdfs_")
            for pdf in request.pdfs:
                raw = base64.b64decode(pdf.content_base64)
                name = os.path.basename(pdf.filename) or "paper.pdf"
                path = os.path.join(tmpdir, name)
                with open(path, "wb") as f:
                    f.write(raw)
                pdf_paths.append(path)

        if request.custom_algorithm_file:
            raw = base64.b64decode(request.custom_algorithm_file.content_base64)
            tmpdir = tempfile.mkdtemp(prefix="bw_algo_")
            name = os.path.basename(request.custom_algorithm_file.filename) or "algo.py"
            if not name.endswith(".py"):
                name += ".py"
            custom_algo_path = os.path.join(tmpdir, name)
            with open(custom_algo_path, "wb") as f:
                f.write(raw)

        # Resolve LLM backend — "claude" or "nemotron"
        backend_choice = (request.llm_backend or "claude").lower()
        nemotron_url = os.environ.get("NEMOTRON_URL", "http://10.19.177.52:11434/v1")
        nemotron_model = os.environ.get(
            "NEMOTRON_MODEL",
            "hf.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q4_K_M",
        )

        agent = OrchestratorAgent(
            execution_mode=request.execution_mode or "local",
            intake_backend=backend_choice,
            pdf_paths=pdf_paths or None,
            custom_algo_path=custom_algo_path,
            nemotron_url=nemotron_url if backend_choice == "nemotron" else None,
            nemotron_model=nemotron_model if backend_choice == "nemotron" else None,
        )
        logger.info("Session created with LLM backend: %s", backend_choice)

        # Seed conversation so the orchestrator knows about attachments
        if pdf_paths or custom_algo_path:
            pdf_names = [os.path.basename(p) for p in pdf_paths] if pdf_paths else []
            algo_name = os.path.basename(custom_algo_path) if custom_algo_path else None
            parts = []
            if pdf_names:
                parts.append(f"I have uploaded PDF papers: {', '.join(pdf_names)}.")
            if algo_name:
                parts.append(f"I have also uploaded my custom algorithm implementation: {algo_name}.")
            parts.append("I'll describe my problem now.")
            messages.append({"role": "user", "content": " ".join(parts)})

            reply_parts = []
            if pdf_names:
                reply_parts.append(
                    "I see your papers. I'll extract the algorithm specifications from them during intake "
                    "so you won't need to specify algorithms separately."
                )
            if algo_name:
                reply_parts.append(
                    f"Your custom algorithm ({algo_name}) is loaded and will be included in the benchmark automatically."
                )
            reply_parts.append("Go ahead — describe your optimization problem.")
            messages.append({"role": "assistant", "content": " ".join(reply_parts)})

        session_id = str(uuid.uuid4())
        _SESSIONS[session_id] = {"agent": agent, "messages": messages}

    # Save user message to DB
    save_message(session_id, "user", request.message)

    # ── Launch orchestrator in a background thread ─────────────────────────
    # The thread always saves the result to DB when it finishes,
    # regardless of whether the SSE client is still connected.

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    _tools_called: list[str] = []

    _last_tool_input: dict = {}

    def progress_cb(event: dict):
        """Called from the worker thread — enqueue event to the async queue."""
        if event.get("type") == "tool_start":
            _tools_called.append(event.get("tool", ""))
            _last_tool_input.clear()
            _last_tool_input.update(event.get("input", {}))
        try:
            loop.call_soon_threadsafe(q.put_nowait, event)
        except RuntimeError:
            pass  # loop closed — SSE client gone, that's fine

        # After intake completes, emit structured algorithm choices for the UI
        if (
            event.get("type") == "tool_end"
            and event.get("tool") == "run_intake"
            and agent is not None
            and agent.state.algo_specs
            and agent.state.preferred_algo_spec_indices is None
        ):
            algos = [
                {
                    "index": i,
                    "name": s.name,
                    "approach": s.approach,
                    "source": s.source,
                }
                for i, s in enumerate(agent.state.algo_specs)
            ]
            try:
                loop.call_soon_threadsafe(q.put_nowait, {
                    "type": "algorithm_select",
                    "algorithms": algos,
                })
            except RuntimeError:
                pass

        # After code_algorithm completes, emit algorithm_coded events (AI-coded only)
        if (
            event.get("type") == "tool_end"
            and event.get("tool") == "code_algorithm"
            and agent is not None
            and agent.state.algorithms
        ):
            custom_name = agent.state.custom_algo_name
            for algo_wrapper in agent.state.algorithms:
                name = getattr(algo_wrapper, "name", None)
                if name and name != custom_name:
                    try:
                        loop.call_soon_threadsafe(q.put_nowait, {
                            "type": "algorithm_coded",
                            "name": name,
                        })
                    except RuntimeError:
                        pass

        # After load_suite with list_suites=true, emit suite choices
        if (
            event.get("type") == "tool_end"
            and event.get("tool") == "load_suite"
            and agent is not None
        ):
            result_text = event.get("result", "")
            if "Available benchmark suites" in result_text:
                # Parse suite info from the tool result
                import re as _re
                suite_opts = []
                for m in _re.finditer(
                    r"•\s+(\w+):\s+(.+?)\((\d+)\s+instances?\)\s*\n\s+(.+)",
                    result_text,
                ):
                    suite_opts.append({
                        "value": m.group(1),
                        "label": m.group(2).strip(),
                        "description": f"{m.group(3)} instances — {m.group(4).strip()}",
                    })
                if suite_opts:
                    try:
                        loop.call_soon_threadsafe(q.put_nowait, {
                            "type": "choice_prompt",
                            "id": "benchmark_suite",
                            "title": "Select a benchmark suite",
                            "options": suite_opts,
                            "multi_select": False,
                        })
                    except RuntimeError:
                        pass
            elif "Instances in" in result_text:
                # Parse instance list from the tool result
                import re as _re
                inst_opts = []
                for m in _re.finditer(
                    r"•\s+([\w._-]+)\s+\((\d+)\s+nodes?\)",
                    result_text,
                ):
                    inst_opts.append({
                        "value": m.group(1),
                        "label": m.group(1),
                        "description": f"{m.group(2)} nodes",
                    })
                if inst_opts:
                    try:
                        loop.call_soon_threadsafe(q.put_nowait, {
                            "type": "choice_prompt",
                            "id": "benchmark_instances",
                            "title": "Select instances to load",
                            "options": inst_opts,
                            "multi_select": True,
                        })
                    except RuntimeError:
                        pass

    # Mark session as actively processing
    _SESSIONS[session_id]["processing"] = True
    _SESSIONS[session_id]["event_queue"] = q

    _agent = agent
    _messages = messages
    _sid = session_id
    _user_msg = request.message

    def _run_orchestrator():
        """Blocking orchestrator work — runs in a daemon thread."""
        try:
            msgs_out, reply = _agent.run_one_turn(
                _messages, _user_msg, progress_cb=progress_cb
            )
            _SESSIONS[_sid]["messages"] = msgs_out

            # Handle plot image
            plot_image = None
            if "analyze_results" in _tools_called and _agent.state.last_plot_path:
                plot_path = _agent.state.last_plot_path
                if os.path.exists(plot_path):
                    import base64 as _b64
                    with open(plot_path, "rb") as f:
                        encoded = _b64.b64encode(f.read()).decode("ascii")
                    plot_image = f"data:image/png;base64,{encoded}"
                _agent.state.last_plot_path = None

            # Always persist to DB (even if the SSE client disconnected)
            save_message(
                session_id=_sid, role="assistant",
                content=reply, tools=None, plot_image=plot_image,
            )

            # Emit interactive choice prompts based on pipeline state
            # Only show instance source if code_algorithm was called this turn
            # AND user didn't already specify an instance source preference
            if (
                "code_algorithm" in _tools_called
                and _agent.state.instance_source is None
                and _agent.state.preferred_instance_source is None
                and _agent.state.algorithms
                and len(_agent.state.algorithms) > 0
            ):
                gens = (
                    _agent.state.config.instance_config.generators
                    if _agent.state.config else []
                )
                gen_desc = ""
                if gens:
                    gen_names = [g.type for g in gens]
                    gen_desc = f" ({', '.join(gen_names)})"
                try:
                    loop.call_soon_threadsafe(q.put_nowait, {
                        "type": "choice_prompt",
                        "id": "instance_source",
                        "title": "How would you like to provide instances?",
                        "options": [
                            {
                                "value": "generators",
                                "label": "Generators",
                                "description": f"Use proposed graph generators{gen_desc}",
                            },
                            {
                                "value": "custom",
                                "label": "Custom JSON",
                                "description": "Load from your own instance file",
                            },
                            {
                                "value": "benchmark suite",
                                "label": "Benchmark Suite",
                                "description": "Use standard benchmark instances (DIMACS, BiqMac, etc.)",
                            },
                        ],
                        "multi_select": False,
                    })
                except RuntimeError:
                    pass

            # Notify queue — the SSE generator will pick this up if connected
            try:
                loop.call_soon_threadsafe(q.put_nowait, {
                    "type": "done", "session_id": _sid,
                    "reply": reply, "plot_image": plot_image,
                })
            except RuntimeError:
                pass

        except Exception as e:
            logger.exception("Orchestrator error: %s", e)
            try:
                loop.call_soon_threadsafe(q.put_nowait, {
                    "type": "error", "session_id": _sid, "error": str(e),
                })
            except RuntimeError:
                pass
        finally:
            _SESSIONS[_sid]["processing"] = False

    thread = threading.Thread(target=_run_orchestrator, daemon=True)
    thread.start()

    # ── SSE stream — just reads from the queue ────────────────────────────
    async def event_stream():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=2.0)
                    yield f"data: {_safe_json(event)}\n\n"
                    # Terminal events — stop streaming
                    if event.get("type") in ("done", "error"):
                        break
                except asyncio.TimeoutError:
                    if not _SESSIONS.get(_sid, {}).get("processing", False):
                        # Orchestrator finished while we weren't reading — drain
                        while not q.empty():
                            evt = q.get_nowait()
                            yield f"data: {_safe_json(evt)}\n\n"
                        break
                    # Keep-alive heartbeat so proxies don't kill the connection
                    yield f"data: {_safe_json({'type': 'heartbeat'})}\n\n"
        except Exception as e:
            logger.exception("Stream error: %s", e)
            yield f"data: {_safe_json({'type': 'error', 'session_id': _sid, 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Legacy one-shot endpoint ────────────────────────────────────────────────

@app.post("/api/benchmark", response_model=BenchmarkResponse)
async def run_benchmark_endpoint(request: BenchmarkRequest):
    try:
        logging.info(f"Processing query: {request.query!r}")

        agent = IntakeAgent()
        config = agent.run(request.query, interactive=False)
        logging.info(f"Generated config: {config}")

        runner = BenchmarkRunner(config)

        if config.problem_class == "minimum_vertex_cover":
            runner.register_algorithm(GreedyVertexCover())
            runner.register_algorithm(RandomVertexCover())
        elif config.problem_class == "maximum_cut":
            runner.register_algorithm(RandomMaxCut())
        else:
            logging.warning(f"Unknown problem class {config.problem_class}")
            runner.register_algorithm(GreedyVertexCover())
            runner.register_algorithm(RandomVertexCover())

        df = runner.run(
            execution_mode=request.execution_mode,
            modal_token_id=request.modal_token_id,
            modal_token_secret=request.modal_token_secret,
        )

        return transform_results(df)

    except Exception as e:
        logging.error(f"Error running benchmark: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
