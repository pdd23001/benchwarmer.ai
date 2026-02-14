#!/usr/bin/env python3
"""
Run the PDF pipeline: parse paper → get code → run in Modal.

Usage:
  # Basic: pass a PDF path (code is run in Modal by default)
  python scripts/run_pdf_pipeline.py path/to/paper.pdf

  # Skip execution, only show code
  python scripts/run_pdf_pipeline.py paper.pdf --no-run

  # JSON output for piping
  python scripts/run_pdf_pipeline.py paper.pdf --json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path when run as script
sys.path.insert(0, ".")

from benchwarmer.agents import process_pdf


def _to_dict(obj):
    """Recursively convert dataclasses to dicts for JSON serialization."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, list):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="Process a research paper PDF: find GitHub code or generate with Claude."
    )
    parser.add_argument(
        "pdf",
        type=str,
        help="Path to the research paper PDF file",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Do not run the code (only show generated/fetched code)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw result as JSON",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    if not pdf_path.suffix.lower() == ".pdf":
        print(f"Error: Expected a .pdf file, got: {pdf_path.suffix}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing: {pdf_path}", file=sys.stderr)

    result = process_pdf(pdf_path, run_code=not args.no_run)

    if args.json:
        print(json.dumps(_to_dict(result), indent=2, default=str))
        return

    # Pretty print
    print()
    print("=" * 70)
    print("PIPELINE RESULT")
    print("=" * 70)

    pdf = result.pdf_content
    if pdf:
        print(f"Paper title:  {pdf.title}")
        if pdf.abstract:
            abstract_preview = (pdf.abstract[:200] + "...") if len(pdf.abstract) > 200 else pdf.abstract
            print(f"Abstract:     {abstract_preview}")
        if pdf.algorithm_keywords:
            print(f"Keywords:     {', '.join(pdf.algorithm_keywords)}")
        if pdf.github_urls:
            print(f"GitHub URLs:  {', '.join(pdf.github_urls)}")
    print()

    print(f"Source:       {result.source}")
    print(f"Provenance:   {result.provenance}")
    print()

    if result.source == "github" and result.implementations:
        print("--- GitHub Code Found ---")
        for i, impl in enumerate(result.implementations[:5], 1):
            print(f"  Repo {i}: {impl.source_repo_url}")
            print(f"  Files:  {len(impl.snippets)} code block(s)")
            print()
            for j, sn in enumerate(impl.snippets, 1):
                file_label = sn.file_path or f"block_{j}"
                lines = sn.content.split("\n")
                print(f"    [{j}] {file_label} ({sn.language}, {len(lines)} lines)")
                # Show first 15 lines of each file
                for line in lines[:15]:
                    try:
                        print(f"      {line}")
                    except UnicodeEncodeError:
                        print(f"      {line.encode('ascii', 'replace').decode()}")
                if len(lines) > 15:
                    print(f"      ... ({len(lines)} lines total, {len(sn.content)} chars)")
                print()
            print("-" * 50)

    elif result.source == "generated" and result.generated_code:
        gen = result.generated_code
        print("--- Claude-Generated Implementation ---")
        print(f"Algorithm:    {gen.algorithm_name}")
        print(f"Language:     {gen.language}")
        if gen.dependency_hints:
            print(f"Dependencies: {', '.join(gen.dependency_hints)}")
        print()
        print("Code:")
        print("-" * 50)
        # Show first 60 lines
        lines = gen.code.split("\n")
        for line in lines[:60]:
            print(f"  {line}")
        if len(lines) > 60:
            print(f"  ... ({len(lines)} lines total)")
        print("-" * 50)
        if gen.explanation:
            print()
            print(f"Explanation: {gen.explanation}")

    else:
        print("No implementations found or generated.")

    # Execution output (if code was run)
    if result.execution_result:
        ex = result.execution_result
        print("--- Execution Output ---")
        print(f"  Exit code: {ex.exit_code}  |  Runtime: {ex.runtime_seconds}s")
        if ex.error:
            print(f"  Error: {ex.error}")
        if ex.stdout:
            print("  Stdout:")
            for line in ex.stdout.rstrip().split("\n"):
                print(f"    {line}")
        if ex.stderr:
            print("  Stderr:")
            for line in ex.stderr.rstrip().split("\n"):
                print(f"    {line}")
        print("-" * 50)

    print()


if __name__ == "__main__":
    main()
