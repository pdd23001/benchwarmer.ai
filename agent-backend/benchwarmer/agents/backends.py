"""
LLM Backends — minimal abstraction to support interchangeable models
(e.g. Claude via Anthropic API vs Nemotron via OpenAI-compatible endpoint).
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Fallback parser for local models (Ollama/Nemotron) that emit
# tool calls as text markup instead of structured API responses.
# ──────────────────────────────────────────────────────────────

def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output."""
    # Remove complete think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove unclosed think blocks (model started thinking but didn't close)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    # Remove orphaned </think> tags
    text = re.sub(r"</think>\s*", "", text)
    return text.strip()


def _parse_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    """
    Parse tool calls embedded as text markup in model output.

    Supports formats commonly emitted by Ollama/Nemotron:

    Format 1 (XML-style):
        <tool_call>
        <function=run_intake>
        <parameter=description>Max cut</parameter>
        </function>
        </tool_call>

    Format 2 (JSON-style):
        <tool_call>
        {"name": "run_intake", "arguments": {"description": "Max cut"}}
        </tool_call>

    Returns list of dicts with keys: name, arguments (dict).
    """
    tool_calls = []

    # Find all <tool_call>...</tool_call> blocks
    tc_blocks = re.findall(
        r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL
    )

    for block in tc_blocks:
        block = block.strip()

        # ── Format 1: <function=name> ... <parameter=key>value</parameter> ──
        func_match = re.search(r"<function=(\w+)>", block)
        if func_match:
            func_name = func_match.group(1)
            params = {}
            for pm in re.finditer(
                r"<parameter=(\w+)>(.*?)</parameter>", block, flags=re.DOTALL
            ):
                key = pm.group(1)
                value = pm.group(2).strip()
                # Try to parse as JSON value (number, bool, list, dict)
                try:
                    params[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    params[key] = value
            tool_calls.append({"name": func_name, "arguments": params})
            continue

        # ── Format 2: JSON object with name + arguments ──
        try:
            data = json.loads(block)
            if isinstance(data, dict) and "name" in data:
                args = data.get("arguments", data.get("parameters", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append({"name": data["name"], "arguments": args})
                continue
        except json.JSONDecodeError:
            pass

        # ── Format 3: {"function": {"name": ..., "arguments": ...}} ──
        try:
            data = json.loads(block)
            if isinstance(data, dict) and "function" in data:
                func = data["function"]
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append({"name": func["name"], "arguments": args})
                continue
        except json.JSONDecodeError:
            pass

        logger.warning("Could not parse tool_call block: %s", block[:200])

    return tool_calls


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"


@dataclass
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass
class LLMResponse:
    content: list[TextBlock | ToolUseBlock]
    stop_reason: str | None  # "end_turn", "tool_use", "max_tokens", etc.


class AbstractLLMBackend(ABC):
    """Interface for LLM backends."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass


class ClaudeBackend(AbstractLLMBackend):
    """
    Backend using the Anthropic API (default).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required for ClaudeBackend. "
                "Install it with: pip install anthropic"
            ) from e

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No API key provided. Pass api_key= or set ANTHROPIC_API_KEY."
            )

        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model

    def generate(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )

        # Convert Anthropic response to unified format
        content_blocks = []
        for block in response.content:
            if block.type == "text":
                content_blocks.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                content_blocks.append(
                    ToolUseBlock(
                        id=block.id,
                        name=block.name,
                        input=block.input,
                    )
                )

        return LLMResponse(
            content=content_blocks,
            stop_reason=response.stop_reason,
        )


class OpenAIBackend(AbstractLLMBackend):
    """
    Backend using an OpenAI-compatible API (e.g. vLLM, Nemotron).
    """

    def __init__(
        self,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        api_key: str | None = None,
        model: str = "nvidia/nemotron-3-nano-30b-a3b",
    ) -> None:
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for OpenAIBackend. "
                "Install it with: pip install openai"
            ) from e

        resolved_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No NVIDIA API key provided. Pass api_key= or set NVIDIA_API_KEY."
            )

        self.client = openai.OpenAI(base_url=base_url, api_key=resolved_key)
        self.model = model

    def generate(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        # Convert messages from Anthropic format to OpenAI format
        openai_messages = self._convert_messages(messages, system)
        
        # Convert tools from Anthropic format to OpenAI format
        openai_tools = None
        if tools:
            openai_tools = [self._convert_tool(t) for t in tools]

        # Build kwargs — omit tools entirely if not provided
        create_kwargs = dict(
            model=self.model,
            messages=openai_messages,
            max_tokens=max_tokens,
        )
        if openai_tools:
            create_kwargs["tools"] = openai_tools

        response = self.client.chat.completions.create(**create_kwargs)

        choice = response.choices[0]
        message = choice.message
        
        content_blocks = []
        has_structured_tool_calls = False
        
        # Tool calls (structured API response — preferred path)
        if message.tool_calls:
            has_structured_tool_calls = True
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                
                content_blocks.append(
                    ToolUseBlock(
                        id=tc.id,
                        name=tc.function.name,
                        input=args,
                    )
                )

        # Text content — may contain embedded tool calls from local models
        raw_text = message.content or ""
        
        if raw_text and not has_structured_tool_calls:
            # ── Fallback: parse tool calls from text markup ──
            # Local models (Ollama/Nemotron) often emit <tool_call> XML
            # and <think> blocks as plain text instead of using the API.
            parsed_calls = _parse_tool_calls_from_text(raw_text)
            
            if parsed_calls:
                logger.info(
                    "Parsed %d tool call(s) from text markup (fallback)",
                    len(parsed_calls),
                )
                for tc in parsed_calls:
                    content_blocks.append(
                        ToolUseBlock(
                            id=f"call_{uuid.uuid4().hex[:12]}",
                            name=tc["name"],
                            input=tc["arguments"],
                        )
                    )
                has_structured_tool_calls = True
                
                # Extract any non-tool-call text (strip think blocks + tool_call blocks)
                remaining = _strip_think_blocks(raw_text)
                remaining = re.sub(
                    r"<tool_call>.*?</tool_call>", "", remaining, flags=re.DOTALL
                ).strip()
                if remaining:
                    content_blocks.insert(0, TextBlock(text=remaining))
            else:
                # No tool calls found — just clean up think blocks from text
                cleaned = _strip_think_blocks(raw_text)
                if cleaned:
                    content_blocks.append(TextBlock(text=cleaned))
        elif raw_text and has_structured_tool_calls:
            # Structured tool calls exist but there's also text — clean it
            cleaned = _strip_think_blocks(raw_text)
            if cleaned:
                content_blocks.insert(0, TextBlock(text=cleaned))

        # Stop reason mapping
        stop_reason = "end_turn"
        if has_structured_tool_calls or choice.finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif choice.finish_reason == "length":
            stop_reason = "max_tokens"
        
        return LLMResponse(
            content=content_blocks,
            stop_reason=stop_reason,
        )

    def _convert_messages(
        self,
        messages: list[dict[str, Any]],
        system: str,
    ) -> list[dict[str, Any]]:
        """Convert Anthropic message list to OpenAI format."""
        openai_msgs = [{"role": "system", "content": system}]

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                # Handle tool results (Anthropic sends them as 'user' role blocks)
                if isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_result" for b in content
                ):
                    for block in content:
                        if block["type"] == "tool_result":
                            openai_msgs.append({
                                "role": "tool",
                                "tool_call_id": block["tool_use_id"],
                                "content": str(block["content"]),
                            })
                        elif block["type"] == "text":
                            openai_msgs.append({"role": "user", "content": block["text"]})
                else:
                    # Standard user message
                    text_content = content
                    if isinstance(content, list):
                        # Extract text from blocks
                        text_content = "\n".join(
                            b["text"] for b in content if b.get("type") == "text"
                        )
                    openai_msgs.append({"role": "user", "content": text_content})

            elif role == "assistant":
                # Handle tool use (Anthropic sends them as 'assistant' role blocks)
                tool_calls = []
                text_parts = []
                
                if isinstance(content, list):
                    for block in content:
                        if hasattr(block, "type"):
                            # It's an object from previous turn (Anthropic block object)
                            b_type = block.type
                        else:
                            # It's a dict
                            b_type = block.get("type")

                        if b_type == "tool_use":
                            # Extract tool use info
                            if hasattr(block, "id"):
                                tid = block.id
                                name = block.name
                                inp = block.input
                            else:
                                tid = block["id"]
                                name = block["name"]
                                inp = block["input"]

                            tool_calls.append({
                                "id": tid,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(inp),
                                },
                            })
                        elif b_type == "text":
                            if hasattr(block, "text"):
                                text_parts.append(block.text)
                            else:
                                text_parts.append(block["text"])
                elif isinstance(content, str):
                    text_parts.append(content)

                openai_msg = {"role": "assistant"}
                if text_parts:
                    openai_msg["content"] = "\n".join(text_parts)
                if tool_calls:
                    openai_msg["tool_calls"] = tool_calls
                
                openai_msgs.append(openai_msg)

        return openai_msgs

    def _convert_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Convert Anthropic tool definition to OpenAI format."""
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
