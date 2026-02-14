"""
PDF-driven pipeline orchestration.

Flow:
  1. User uploads a PDF of their algorithm / research paper.
  2. Parse the PDF → extract title, abstract, algorithm keywords, and GitHub URLs.
  3. If the paper contains GitHub links → give them to Perplexity to find and
     extract the implementation code from those specific repos.
  4. If Perplexity finds real code → return it.
  5. If no GitHub links in paper, or no code found → Claude generates the implementation.
  6. If run_code is True, execute the code in Modal (or local fallback) and attach result.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

from benchwarmer.agents.code_generator import generate_code_from_paper
from benchwarmer.agents.tools import (
    extract_code_blocks_from_markdown,
    sonar_extract_code_from_repos,
)
from benchwarmer.config import SearchConfig, get_search_config
from benchwarmer.engine.modal_runner import run_code_remotely
from benchwarmer.pipeline.pdf_parser import parse_pdf
from benchwarmer.schemas.search import (
    ExecutionResult,
    ImplementationCandidate,
    PDFContent,
    PipelineResult,
)


def _log(msg: str) -> None:
    """Print a status message to stderr."""
    print(msg, file=sys.stderr)


def _get_runnable_code(
    implementations: list,
    generated_code: Optional[object],
) -> Optional[Tuple[str, list]]:
    """Extract (code_string, pip_packages) from pipeline result. Returns None if nothing to run."""
    if generated_code is not None and hasattr(generated_code, "code"):
        code = getattr(generated_code, "code", "")
        deps = list(getattr(generated_code, "dependency_hints", []) or [])
        if code.strip():
            return (code.strip(), deps)
    if implementations:
        impl = implementations[0]
        snippets = getattr(impl, "snippets", []) or []
        if not snippets:
            return None
        # Prefer Python snippets; concatenate if multiple
        parts = [s.content for s in snippets if getattr(s, "language", "").lower() == "python"]
        if not parts:
            parts = [s.content for s in snippets]
        code = "\n\n".join(parts)
        deps = list(getattr(impl, "dependency_hints", []) or [])
        if code.strip():
            return (code.strip(), deps)
    return None


def process_pdf(
    pdf_path: str | Path,
    *,
    config: Optional[SearchConfig] = None,
    run_code: bool = True,
) -> PipelineResult:
    """
    Main entry point: process an uploaded research paper PDF.

    1. Parse PDF → extract content + GitHub URLs found in the paper.
    2. If GitHub links exist → send them to Perplexity to extract the code.
    3. If Perplexity returns real code → return it.
    4. Otherwise → Claude generates the implementation from the paper.
    5. If run_code is True, run the code in Modal and attach execution_result.
    """
    config = config or get_search_config()

    # ── Step 1: Parse the PDF ────────────────────────────────────────
    _log("Step 1: Parsing paper...")
    pdf_content = parse_pdf(pdf_path)
    _log(f"  Title: {pdf_content.title}")
    if pdf_content.algorithm_keywords:
        _log(f"  Keywords: {', '.join(pdf_content.algorithm_keywords[:8])}")
    if pdf_content.github_urls:
        _log(f"  GitHub links found in paper: {pdf_content.github_urls}")
    else:
        _log("  No GitHub links found in paper.")

    provenance: dict = {
        "pdf_path": str(pdf_path),
        "title": pdf_content.title,
        "algorithm_keywords": pdf_content.algorithm_keywords,
        "github_urls_in_paper": pdf_content.github_urls,
    }

    # ── Step 2: If paper has GitHub links → Perplexity extracts code ──
    if pdf_content.github_urls:
        _log("Step 2: Sending GitHub links to Perplexity to extract code...")

        response_content, code_snippets = sonar_extract_code_from_repos(
            repo_urls=pdf_content.github_urls,
            paper_title=pdf_content.title,
            algorithm_keywords=pdf_content.algorithm_keywords,
            config=config,
        )

        if code_snippets:
            _log(f"  Perplexity found {len(code_snippets)} code block(s) from the repo(s)")

            # Build an implementation candidate from the Perplexity-extracted code
            implementation = ImplementationCandidate(
                source_repo_url=pdf_content.github_urls[0],
                language=code_snippets[0].language if code_snippets else "python",
                snippets=code_snippets,
                confidence=1.0,
            )

            result = PipelineResult(
                source="github",
                pdf_content=pdf_content,
                implementations=[implementation],
                generated_code=None,
                provenance={
                    **provenance,
                    "stage": "perplexity_extracted_from_repo",
                    "repos_queried": pdf_content.github_urls,
                    "snippets_found": len(code_snippets),
                    "perplexity_response_preview": response_content[:500],
                },
            )
            if run_code:
                runnable = _get_runnable_code([implementation], None)
                if runnable:
                    code_str, deps = runnable
                    _log("Step 3: Running code (Modal)...")
                    exec_result = run_code_remotely(code_str, pip_packages=deps or None, timeout_seconds=120)
                    result.execution_result = exec_result
                    _log(f"  Exit code {exec_result.exit_code}, runtime {exec_result.runtime_seconds}s")
            return result
        else:
            _log("  Perplexity could not find code in the linked repo(s).")

    # ── Step 3: No GitHub links or no code found → Claude generates ──
    _log("Step 3: Claude is generating the implementation from the paper...")
    generated = generate_code_from_paper(pdf_content, config)

    result = PipelineResult(
        source="generated",
        pdf_content=pdf_content,
        implementations=[],
        generated_code=generated,
        provenance={
            **provenance,
            "stage": "claude_generated",
            "reason": "no_github_links" if not pdf_content.github_urls else "repo_had_no_code",
        },
    )
    if run_code:
        runnable = _get_runnable_code([], generated)
        if runnable:
            code_str, deps = runnable
            _log("Step 4: Running code (Modal)...")
            exec_result = run_code_remotely(code_str, pip_packages=deps or None, timeout_seconds=120)
            result.execution_result = exec_result
            _log(f"  Exit code {exec_result.exit_code}, runtime {exec_result.runtime_seconds}s")
    return result
