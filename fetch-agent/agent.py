"""
Benchwarmer.AI — Fetch.ai Agentverse wrapper agent.

Bridges the Fetch.ai Chat Protocol to the existing FastAPI backend
so the service is discoverable on ASI:One.
"""

import json
import os
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    TextContent,
    chat_protocol_spec,
)

load_dotenv()

AGENT_SEED = os.getenv("AGENT_SEED", "benchwarmer-ai-default-seed")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

agent = Agent(
    name="benchwarmer-ai",
    seed=AGENT_SEED,
    port=8001,
    mailbox=True,
    publish_agent_details=True,
)

chat_proto = Protocol(spec=chat_protocol_spec)


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

async def parse_sse_stream(response: httpx.Response):
    """Yield parsed JSON objects from an SSE text/event-stream response."""
    buffer = ""
    async for chunk in response.aiter_text():
        buffer += chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            # Lines may be prefixed with "data: " per SSE spec
            if line.startswith("data: "):
                line = line[6:]
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


TOOL_FRIENDLY_NAMES = {
    "run_intake": "Running intake analysis",
    "modify_generators": "Modifying generators",
    "modify_execution_config": "Updating execution config",
    "code_algorithm": "Coding algorithm",
    "remove_algorithm": "Removing algorithm",
    "show_status": "Fetching pipeline status",
    "run_benchmark": "Running benchmark",
    "analyze_results": "Analyzing results",
    "go_back": "Reverting step",
    "export_results": "Exporting results",
    "set_execution_mode": "Switching execution mode",
    "use_generators": "Setting up generators",
    "load_custom_instances": "Loading custom instances",
    "load_suite": "Loading benchmark suite",
}


# ---------------------------------------------------------------------------
# Chat protocol handlers
# ---------------------------------------------------------------------------

@chat_proto.on_message(ChatMessage)
async def handle_chat(ctx: Context, sender: str, msg: ChatMessage):
    ctx.logger.info(f"ChatMessage from {sender}")

    # 1. Acknowledge immediately
    await ctx.send(
        sender,
        ChatAcknowledgement(acknowledged_msg_id=msg.msg_id),
    )

    # 2. Extract text content
    text_parts = [item.text for item in msg.content if isinstance(item, TextContent)]
    user_text = " ".join(text_parts).strip()
    if not user_text:
        await ctx.send(
            sender,
            ChatMessage(
                msg_id=uuid4().hex,
                content=[
                    TextContent(text="Please send a text message."),
                    EndSessionContent(),
                ],
            ),
        )
        return

    # 3. Look up existing session for this sender
    session_id = ctx.storage.get(sender)

    # 4. POST to FastAPI backend
    payload = {"message": user_text, "session_id": session_id}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            async with client.stream(
                "POST",
                f"{BACKEND_URL}/api/chat",
                json=payload,
            ) as response:
                async for event in parse_sse_stream(response):
                    etype = event.get("type")

                    if etype == "tool_start":
                        tool = event.get("tool", "")
                        label = TOOL_FRIENDLY_NAMES.get(tool, f"Running {tool}")
                        await ctx.send(
                            sender,
                            ChatMessage(
                                msg_id=uuid4().hex,
                                content=[TextContent(text=f"⏳ {label}...")],
                            ),
                        )

                    elif etype == "tool_end":
                        tool = event.get("tool", "")
                        label = TOOL_FRIENDLY_NAMES.get(tool, tool)
                        await ctx.send(
                            sender,
                            ChatMessage(
                                msg_id=uuid4().hex,
                                content=[TextContent(text=f"✅ {label} complete.")],
                            ),
                        )

                    elif etype == "error":
                        error_text = event.get("error", "Unknown error")
                        # Clear session if it expired
                        if "expired" in error_text.lower():
                            ctx.storage.remove(sender)
                        await ctx.send(
                            sender,
                            ChatMessage(
                                msg_id=uuid4().hex,
                                content=[
                                    TextContent(text=f"❌ {error_text}"),
                                    EndSessionContent(),
                                ],
                            ),
                        )
                        return

                    elif etype == "done":
                        # Persist session for multi-turn
                        new_session_id = event.get("session_id")
                        if new_session_id:
                            ctx.storage.set(sender, new_session_id)

                        reply = event.get("reply", "")
                        await ctx.send(
                            sender,
                            ChatMessage(
                                msg_id=uuid4().hex,
                                content=[
                                    TextContent(text=reply),
                                    EndSessionContent(),
                                ],
                            ),
                        )
                        return

                    # Ignore "thinking", "heartbeat", and unknown events

    except httpx.HTTPError as exc:
        ctx.logger.error(f"Backend request failed: {exc}")
        await ctx.send(
            sender,
            ChatMessage(
                msg_id=uuid4().hex,
                content=[
                    TextContent(
                        text="Sorry, the Benchwarmer backend is currently unavailable. Please try again later."
                    ),
                    EndSessionContent(),
                ],
            ),
        )


@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Ack from {sender} for {msg.acknowledged_msg_id}")


agent.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
