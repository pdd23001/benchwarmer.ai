"""Tests for search agent: decision branches and payload schema validation."""

import pytest
from unittest.mock import patch

from benchwarmer.agents.search import search_algorithms
from benchwarmer.agents.tools import (
    build_sandbox_payload_from_result,
    extract_code_blocks_from_markdown,
    extract_github_repo_urls,
    parse_sonar_response,
    score_implementation_candidate,
    build_implementation_candidates,
    build_paper_candidates_from_search_results,
)
from benchwarmer.schemas.search import (
    CodeSnippet,
    ImplementationCandidate,
    PaperCandidate,
    SearchConstraints,
    SearchResult,
    SandboxPayload,
    SandboxCodeUnit,
)
from benchwarmer.pipeline.ocr_handoff import create_ocr_handoff_payload, OCRHandoffPayload


# ---- Unit: URL and code extraction ----

def test_extract_github_repo_urls_from_text():
    text = "See https://github.com/user/repo and https://github.com/other/proj/tree/main/src"
    urls = extract_github_repo_urls(text)
    assert "https://github.com/user/repo" in urls
    assert "https://github.com/other/proj" in urls
    assert len(urls) == 2


def test_extract_github_repo_urls_dedupe():
    text = "https://github.com/a/b https://github.com/A/B"
    urls = extract_github_repo_urls(text)
    assert len(urls) == 1
    assert urls[0].lower() == "https://github.com/a/b"


def test_extract_code_blocks_from_markdown():
    content = "Text\n```python\nx = 1\n```\nMore\n```\nraw\n```"
    snippets = extract_code_blocks_from_markdown(content)
    assert len(snippets) == 2
    assert snippets[0].language == "python"
    assert snippets[0].content == "x = 1"
    assert snippets[1].content == "raw"


def test_parse_sonar_response():
    r = {
        "choices": [{"message": {"content": "Hello", "role": "assistant"}}],
        "citations": ["https://a.com"],
        "search_results": [{"title": "T", "url": "https://b.com", "snippet": "S"}],
    }
    content, citations, search_results = parse_sonar_response(r)
    assert content == "Hello"
    assert citations == ["https://a.com"]
    assert len(search_results) == 1
    assert search_results[0]["url"] == "https://b.com"


# ---- Unit: scoring ----

def test_score_implementation_candidate_high():
    constraints = SearchConstraints(problem_class="max_cut", algorithm_names=["Goemans-Williamson"])
    content = "Goemans-Williamson max cut implementation in Python."
    snippets = [CodeSnippet(content="def solve(): pass", language="python")]
    score = score_implementation_candidate(
        "https://github.com/x/y", "max cut", constraints, content, snippets, None
    )
    assert score >= 0.6


def test_score_implementation_candidate_low_no_code():
    constraints = SearchConstraints()
    score = score_implementation_candidate(
        "https://github.com/x/y", "random query", constraints, "some text", [], None
    )
    assert score < 0.5


# ---- Branch: GitHub hit -> run_code ----

@patch("benchwarmer.agents.search.sonar_search_github_implementations")
@patch("benchwarmer.agents.search.sonar_search_papers")
def test_search_algorithms_returns_run_code_when_implementation_found(mock_papers, mock_github):
    mock_github.return_value = (
        "Here is a **max cut** implementation: https://github.com/algo/maxcut. ```python\ndef max_cut(g): pass\n```",
        ["https://github.com/algo/maxcut"],
        [CodeSnippet(content="def max_cut(g): pass", language="python")],
        [{"url": "https://github.com/algo/maxcut", "snippet": "max cut algorithm"}],
    )
    mock_papers.return_value = ("", [])

    result = search_algorithms("max cut algorithm", SearchConstraints(problem_class="maximum_cut", min_confidence=0.3))

    assert result.found_implementation is True
    assert result.next_action == "run_code"
    assert len(result.implementations) > 0
    assert any(len(impl.snippets) > 0 for impl in result.implementations)
    assert result.papers == []
    mock_papers.assert_not_called()


# ---- Branch: No hit -> ocr_pipeline ----

@patch("benchwarmer.agents.search.sonar_search_github_implementations")
@patch("benchwarmer.agents.search.sonar_search_papers")
def test_search_algorithms_returns_ocr_pipeline_when_no_implementation(mock_papers, mock_github):
    mock_github.return_value = (
        "No code found.",
        [],
        [],
        [],
    )
    mock_papers.return_value = (
        "",
        [
            {"title": "Max-Cut Paper", "url": "https://arxiv.org/abs/1234.5678", "snippet": "We present an algorithm."},
            {"title": "Another", "url": "https://doi.org/10.1234/xyz", "snippet": "Related work."},
        ],
    )

    result = search_algorithms("obscure algorithm xyz", SearchConstraints(min_confidence=0.9))

    assert result.found_implementation is False
    assert result.next_action == "ocr_pipeline"
    assert len(result.papers) >= 1
    assert result.implementations == []
    mock_papers.assert_called_once()


# ---- Contract: Sandbox payload ----

def test_build_sandbox_payload_has_code_units_and_metadata():
    impl = ImplementationCandidate(
        source_repo_url="https://github.com/a/b",
        snippets=[CodeSnippet(content="print(1)", language="python")],
        confidence=0.8,
    )
    result = SearchResult(
        found_implementation=True,
        implementations=[impl],
        next_action="run_code",
        query="test query",
    )
    payload = build_sandbox_payload_from_result(result)
    assert isinstance(payload, SandboxPayload)
    assert payload.query == "test query"
    assert len(payload.code_units) > 0
    assert payload.code_units[0].content == "print(1)"
    assert payload.code_units[0].source_repo_url == "https://github.com/a/b"
    assert payload.provenance is not None


def test_build_sandbox_payload_empty_when_no_implementation():
    result = SearchResult(found_implementation=False, papers=[], next_action="ocr_pipeline", query="q")
    payload = build_sandbox_payload_from_result(result)
    assert payload.code_units == []
    assert payload.query == "q"


# ---- OCR handoff contract ----

def test_create_ocr_handoff_payload():
    papers = [
        PaperCandidate(title="P1", url="https://arxiv.org/abs/1", candidate_algorithm_names=["AlgoA"]),
        PaperCandidate(title="P2", url="https://doi.org/2", candidate_algorithm_names=["AlgoB", "AlgoA"]),
    ]
    handoff = create_ocr_handoff_payload("query", papers, ranked_algorithm_names=["AlgoA"])
    assert isinstance(handoff, OCRHandoffPayload)
    assert handoff.query == "query"
    assert len(handoff.papers) == 2
    assert "AlgoA" in handoff.ranked_algorithm_names
    assert "AlgoB" in handoff.ranked_algorithm_names


# ---- Schema: build_implementation_candidates ----

def test_build_implementation_candidates_filters_by_min_confidence():
    constraints = SearchConstraints(problem_class="max_cut")
    candidates = build_implementation_candidates(
        repo_urls=["https://github.com/x/y"],
        code_snippets=[CodeSnippet(content="code", language="python")],
        content="max cut implementation",
        search_results=[],
        query="max cut",
        constraints=constraints,
        min_confidence=0.99,
    )
    # High threshold may yield 0 or 1 depending on scoring
    assert all(c.confidence >= 0.99 for c in candidates)


def test_build_paper_candidates_ranking():
    search_results = [
        {"title": "Relevant", "url": "https://arxiv.org/abs/1111.1111", "snippet": "max cut exact algorithm"},
        {"title": "Less", "url": "https://example.com/other", "snippet": "other topic"},
    ]
    papers = build_paper_candidates_from_search_results(search_results, "", "max cut", max_results=10)
    assert len(papers) == 2
    assert papers[0].relevance_score >= papers[1].relevance_score
    assert papers[0].arxiv_id == "1111.1111"
