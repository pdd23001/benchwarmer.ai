"""Agents: PDF pipeline orchestration, GitHub search, and Claude code generation."""

from benchwarmer.agents.search import process_pdf
from benchwarmer.agents.code_generator import generate_code_from_paper
from benchwarmer.agents.tools import build_sandbox_payload_from_result

__all__ = ["process_pdf", "generate_code_from_paper", "build_sandbox_payload_from_result"]
