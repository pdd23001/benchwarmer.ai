#!/usr/bin/env python3
"""
Run the search agent with a prompt and print results.

Usage:
  # With prompt as argument (set PERPLEXITY_API_KEY first)
  python scripts/run_search.py "maximum cut algorithm Python implementation"

  # Optional: problem class and algorithm names
  python scripts/run_search.py "Goemans-Williamson max cut" --problem-class maximum_cut --algorithms "Goemans-Williamson"

  # Interactive: prompt read from stdin
  python scripts/run_search.py
  > minimum vertex cover greedy algorithm
"""

import argparse
import json
import sys

# Add project root to path when run as script
sys.path.insert(0, ".")

from benchwarmer.agents import search_algorithms, build_sandbox_payload_from_result
from benchwarmer.schemas import SearchConstraints


def main():
    parser = argparse.ArgumentParser(description="Test the Perplexity search agent with a prompt.")
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Search prompt (e.g. 'max cut algorithm GitHub')",
    )
    parser.add_argument("--problem-class", "-p", default=None, help="Problem class filter (e.g. maximum_cut)")
    parser.add_argument(
        "--algorithms", "-a",
        default=None,
        help="Comma-separated algorithm names (e.g. 'Goemans-Williamson,SDP')",
    )
    parser.add_argument("--min-confidence", "-c", type=float, default=None, help="Min implementation confidence 0–1")
    parser.add_argument("--json", action="store_true", help="Output raw result as JSON (no pretty print)")
    args = parser.parse_args()

    query = args.query
    if not query:
        print("Enter your search prompt (one line):", file=sys.stderr)
        try:
            query = input().strip()
        except EOFError:
            query = ""
        if not query:
            print("No query provided.", file=sys.stderr)
            sys.exit(1)

    constraints = SearchConstraints(
        problem_class=args.problem_class,
        algorithm_names=[s.strip() for s in args.algorithms.split(",")] if args.algorithms else None,
        min_confidence=args.min_confidence,
    )

    print(f"Query: {query}", file=sys.stderr)
    if constraints.problem_class or constraints.algorithm_names:
        print(f"Constraints: {constraints}", file=sys.stderr)
    print("Searching (GitHub-first, then papers)...", file=sys.stderr)

    result = search_algorithms(query, constraints)

    if args.json:
        # Minimal JSON-friendly dump (dataclasses)
        def _to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
            if isinstance(obj, list):
                return [_to_dict(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            return obj
        print(json.dumps(_to_dict(result), indent=2, default=str))
        return

    # Pretty print
    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Next action:    {result.next_action}")
    print(f"Found implementation: {result.found_implementation}")
    print(f"Provenance:    {result.provenance}")
    print()

    if result.next_action == "run_code" and result.implementations:
        print("--- Implementations ---")
        for i, impl in enumerate(result.implementations[:5], 1):
            print(f"  {i}. {impl.source_repo_url or '(snippet only)'} (confidence={impl.confidence:.2f})")
            for j, sn in enumerate(impl.snippets[:2], 1):
                preview = (sn.content[:200] + "…") if len(sn.content) > 200 else sn.content
                print(f"     Snippet {j} [{sn.language}]: {preview!r}")
        payload = build_sandbox_payload_from_result(result)
        print(f"\nSandbox payload: {len(payload.code_units)} code unit(s), deps: {payload.dependency_hints}")
    elif result.next_action == "ocr_pipeline" and result.papers:
        print("--- Papers (OCR fallback) ---")
        for i, p in enumerate(result.papers[:5], 1):
            print(f"  {i}. {p.title}")
            print(f"     {p.url} (relevance={p.relevance_score:.2f})")
            if p.abstract_snippet:
                print(f"     {p.abstract_snippet[:150]}…")
    else:
        print("No implementations or papers returned.")
    print()


if __name__ == "__main__":
    main()
