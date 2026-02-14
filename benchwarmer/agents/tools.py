"""Tool adapters: Perplexity Sonar, GitHub repo fetching, code extraction, and scoring."""

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple
import httpx

from benchwarmer.config import SearchConfig, get_search_config
from benchwarmer.schemas.search import (
    CodeSnippet,
    ImplementationCandidate,
    PaperCandidate,
    SandboxCodeUnit,
    SandboxPayload,
    SearchConstraints,
    SearchResult,
)


# ---------------------------------------------------------------------------
# GitHub API: fetch actual code from repos
# ---------------------------------------------------------------------------

# File extensions we consider "source code" worth fetching
_CODE_EXTENSIONS = {
    ".py", ".pyx", ".pyi",      # Python
    ".java",                     # Java
    ".cpp", ".c", ".cc", ".h",  # C/C++
    ".rs",                       # Rust
    ".go",                       # Go
    ".js", ".ts",               # JS/TS
    ".jl",                       # Julia
    ".m",                        # MATLAB/Octave
    ".r",                        # R
}

# Skip these directories when scanning repos
_SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".tox", "venv", "env", ".venv", "dist", "build", "egg-info"}

# Maximum total bytes of code to fetch per repo (avoid downloading massive repos)
_MAX_FETCH_BYTES = 200_000


def _github_api_get(url: str, *, timeout: float = 15.0) -> Optional[dict | list]:
    """Make a GET request to the GitHub API (unauthenticated, 60 req/hr limit)."""
    import sys
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "benchwarmer-bot/0.1",
    }
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code == 200:
                return resp.json()
            # Log non-200 for debugging
            print(f"    [GitHub API] {resp.status_code} for {url}", file=sys.stderr)
            return None
    except httpx.HTTPError as exc:
        print(f"    [GitHub API] Error: {exc}", file=sys.stderr)
        return None


def _fetch_raw_file(owner: str, repo: str, file_path: str, ref: str = "HEAD") -> Optional[str]:
    """Fetch a single file's raw content from GitHub via raw.githubusercontent.com."""
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{file_path}"
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                return resp.text
            return None
    except httpx.HTTPError:
        return None


@dataclass
class RepoInfo:
    """Metadata about a GitHub repo fetched from the API."""
    owner: str
    repo: str
    description: str = ""
    language: str = ""
    topics: List[str] = field(default_factory=list)
    default_branch: str = "main"
    file_count: int = 0


def fetch_repo_info(repo_url: str) -> Optional[RepoInfo]:
    """Fetch basic metadata about a GitHub repo (description, language, topics)."""
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)", repo_url)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)

    data = _github_api_get(f"https://api.github.com/repos/{owner}/{repo}")
    if not data or not isinstance(data, dict):
        return None

    return RepoInfo(
        owner=owner,
        repo=repo,
        description=data.get("description") or "",
        language=data.get("language") or "",
        topics=data.get("topics") or [],
        default_branch=data.get("default_branch", "main"),
    )


def fetch_repo_code_files(
    repo_url: str,
    repo_info: Optional[RepoInfo] = None,
    *,
    max_files: int = 10,
) -> List[CodeSnippet]:
    """
    Fetch actual source code files from a GitHub repo.

    Uses the GitHub Trees API to get the file listing, then downloads the most
    relevant source files via raw.githubusercontent.com.

    Returns a list of CodeSnippet with real code from the repo.
    """
    import sys

    # Parse owner/repo from URL
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)", repo_url)
    if not m:
        return []
    owner, repo_name = m.group(1), m.group(2)

    # Use provided repo_info or fetch it
    if not repo_info:
        data = _github_api_get(f"https://api.github.com/repos/{owner}/{repo_name}")
        if not data or not isinstance(data, dict):
            return []
        default_branch = data.get("default_branch", "main")
    else:
        default_branch = repo_info.default_branch

    # Get the full file tree (recursive)
    tree_url = f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/{default_branch}?recursive=1"
    tree_data = _github_api_get(tree_url)
    if not tree_data or not isinstance(tree_data, dict):
        return []

    tree = tree_data.get("tree") or []

    # Filter to source code files, skip junk dirs
    code_files: List[dict] = []
    for item in tree:
        if item.get("type") != "blob":
            continue
        path = item.get("path", "")
        # Skip files in ignored directories
        parts = path.split("/")
        if any(p.lower() in _SKIP_DIRS or p.lower().endswith(".egg-info") for p in parts):
            continue
        # Check extension
        ext = ""
        if "." in path.split("/")[-1]:
            ext = "." + path.split("/")[-1].rsplit(".", 1)[-1]
        if ext.lower() not in _CODE_EXTENSIONS:
            continue
        size = item.get("size", 0)
        # Skip very large files (>50KB) and empty files
        if size == 0 or size > 50_000:
            continue
        code_files.append({"path": path, "size": size, "ext": ext.lower()})

    if not code_files:
        # Log what file types ARE in the repo for debugging
        all_exts: set = set()
        for item in tree:
            if item.get("type") == "blob":
                p = item.get("path", "")
                if "." in p.split("/")[-1]:
                    all_exts.add("." + p.split("/")[-1].rsplit(".", 1)[-1].lower())
        if all_exts:
            print(f"    [debug] Repo has these file types: {sorted(all_exts)}", file=sys.stderr)
        return []

    # Prioritize: Python first, then by path depth (shallower = more important), then by size
    def _sort_key(f: dict) -> tuple:
        is_python = 0 if f["ext"] == ".py" else 1
        depth = f["path"].count("/")
        # Prefer files with algorithm-sounding names
        name = f["path"].split("/")[-1].lower()
        is_main = 0 if any(kw in name for kw in ("main", "algorithm", "solve", "run", "core")) else 1
        is_test = 1 if any(kw in name for kw in ("test", "spec", "conftest")) else 0
        return (is_test, is_python, is_main, depth, f["size"])

    code_files.sort(key=_sort_key)

    # Fetch the top N files, respecting max bytes
    snippets: List[CodeSnippet] = []
    total_bytes = 0
    for cf in code_files[:max_files]:
        if total_bytes >= _MAX_FETCH_BYTES:
            break
        content = _fetch_raw_file(owner, repo_name, cf["path"], ref=default_branch)
        if content and content.strip():
            # Detect language from extension
            lang_map = {
                ".py": "python", ".pyx": "python", ".pyi": "python",
                ".java": "java", ".cpp": "cpp", ".c": "c", ".cc": "cpp", ".h": "c",
                ".rs": "rust", ".go": "go", ".js": "javascript", ".ts": "typescript",
                ".jl": "julia", ".m": "matlab", ".r": "r",
            }
            lang = lang_map.get(cf["ext"], "text")
            snippets.append(CodeSnippet(
                content=content,
                language=lang,
                file_path=cf["path"],
            ))
            total_bytes += len(content)

    return snippets


# ---------------------------------------------------------------------------
# Perplexity Sonar API
# ---------------------------------------------------------------------------

def perplexity_chat_completions(
    messages: List[dict],
    config: Optional[SearchConfig] = None,
    *,
    search_domain_filter: Optional[List[str]] = None,
    search_mode: str = "web",
    max_tokens: int = 4096,
) -> dict:
    """
    Call Perplexity chat completions (Sonar). Returns raw API response with
    choices[0].message.content, citations, and search_results when present.
    """
    config = config or get_search_config()
    if not config.perplexity_api_key:
        return {
            "choices": [{"message": {"content": "", "role": "assistant"}, "index": 0, "finish_reason": "stop"}],
            "citations": [],
            "search_results": [],
        }
    url = f"{config.perplexity_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": config.sonar_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    if search_domain_filter is not None:
        payload["search_domain_filter"] = search_domain_filter
    if search_mode:
        payload["search_mode"] = search_mode
    headers = {
        "Authorization": f"Bearer {config.perplexity_api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    return data


def parse_sonar_response(response: dict) -> Tuple[str, List[str], List[dict]]:
    """Extract content, citations, and search_results from a Perplexity completion response."""
    content = ""
    citations: List[str] = []
    search_results: List[dict] = []
    choices = response.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
    citations = list(response.get("citations") or [])
    search_results = list(response.get("search_results") or [])
    return content, citations, search_results


# ---------------------------------------------------------------------------
# GitHub URL extraction and cleanup
# ---------------------------------------------------------------------------

# Match GitHub repo URLs (user/repo or host/user/repo) and blob/tree paths for normalization
GITHUB_REPO_PATTERN = re.compile(
    r"https?://(?:www\.)?github\.com/([^/\s]+)/([^/\s#]+)(?:/(?:tree|blob)/([^\s#]+))?(?:/[^\s#]*)?"
)


def extract_github_repo_urls(text: str, search_results: Optional[List[dict]] = None) -> List[str]:
    """
    Extract unique GitHub repo root URLs from text and optional search_results.
    Normalizes to https://github.com/owner/repo (no trailing path).
    """
    seen: set = set()
    urls: List[str] = []

    def _strip_citation_artifacts(s: str) -> str:
        """Strip Perplexity citation artifacts from captured GitHub URL parts."""
        s = s.rstrip("/")
        s = re.sub(r"[\.\*]+\[\d*\]?$", "", s)
        s = re.sub(r"\[\d*\]?$", "", s)
        s = re.sub(r"[^a-zA-Z0-9\-_.]+$", "", s)
        return s

    def add_repo(owner: str, repo: str) -> None:
        owner = _strip_citation_artifacts(owner)
        repo = _strip_citation_artifacts(repo)
        if "/" in repo:
            repo = repo.split("/")[0]
        if not owner or not repo:
            return
        key = (owner.lower(), repo.lower())
        if key in seen:
            return
        seen.add(key)
        urls.append(f"https://github.com/{owner}/{repo}")

    for m in GITHUB_REPO_PATTERN.finditer(text):
        add_repo(m.group(1), m.group(2))
    if search_results:
        for r in search_results:
            u = (r.get("url") or "").strip() if isinstance(r, dict) else str(r).strip()
            if "github.com" in u:
                mm = GITHUB_REPO_PATTERN.search(u)
                if mm:
                    add_repo(mm.group(1), mm.group(2))
    return urls


# ---------------------------------------------------------------------------
# Code block extraction
# ---------------------------------------------------------------------------

CODE_BLOCK_PATTERN = re.compile(r"```(\w*)\s*\n(.*?)```", re.DOTALL)


def extract_code_blocks_from_markdown(content: str) -> List[CodeSnippet]:
    """Extract fenced code blocks from markdown content."""
    snippets: List[CodeSnippet] = []
    for m in CODE_BLOCK_PATTERN.finditer(content):
        lang = (m.group(1) or "text").strip().lower()
        if lang == "text":
            lang = "python"
        snippets.append(CodeSnippet(content=m.group(2).strip(), language=lang))
    return snippets


# ---------------------------------------------------------------------------
# Perplexity: extract code from specific repo URLs
# ---------------------------------------------------------------------------

def sonar_extract_code_from_repos(
    repo_urls: List[str],
    paper_title: str,
    algorithm_keywords: List[str],
    config: Optional[SearchConfig] = None,
) -> Tuple[str, List[CodeSnippet]]:
    """
    Give Perplexity specific GitHub repo URLs (found in a research paper) and ask it
    to find and extract the algorithm implementation code from those repos.

    Returns (full_response_content, code_snippets).
    """
    config = config or get_search_config()

    urls_str = "\n".join(f"  - {url}" for url in repo_urls)
    keywords_str = ", ".join(algorithm_keywords[:8]) if algorithm_keywords else paper_title

    prompt = (
        f"I have a research paper titled: \"{paper_title}\"\n"
        f"The paper references these GitHub repositories:\n{urls_str}\n\n"
        f"The key algorithms/methods are: {keywords_str}\n\n"
        "Please look through these repositories and extract the COMPLETE implementation code "
        "for the core algorithm. Include:\n"
        "1. The main algorithm implementation (full source code, not snippets)\n"
        "2. Any helper functions or classes needed to run it\n"
        "3. The file path(s) where the code lives in the repo\n\n"
        "Return the code in fenced code blocks (```python ... ``` or the appropriate language). "
        "If a repo has no code or is just a README, say so explicitly."
    )

    messages = [{"role": "user", "content": prompt}]
    response = perplexity_chat_completions(
        messages,
        config,
        search_domain_filter=["github.com"],
        search_mode="web",
        max_tokens=8192,
    )

    content, _citations, _search_results = parse_sonar_response(response)
    snippets = extract_code_blocks_from_markdown(content)
    return content, snippets


# ---------------------------------------------------------------------------
# Scoring and candidate building (legacy, still used)
# ---------------------------------------------------------------------------

def score_implementation_candidate(
    repo_url: str,
    query: str,
    constraints: SearchConstraints,
    content: str,
    code_snippets: List[CodeSnippet],
    search_result_snippet: Optional[str] = None,
) -> float:
    """
    Compute confidence in [0, 1] for a candidate: algorithm name match, problem class match, code presence.
    """
    text = (content or "") + " " + (search_result_snippet or "")
    text_lower = text.lower()
    query_lower = query.lower()
    score = 0.0
    if code_snippets:
        score += 0.4
    query_terms = [t for t in query_lower.split() if len(t) > 2]
    if query_terms and any(t in text_lower for t in query_terms):
        score += 0.3
    if constraints.problem_class and constraints.problem_class.lower() in text_lower:
        score += 0.2
    if constraints.algorithm_names:
        for name in constraints.algorithm_names:
            if name and name.lower() in text_lower:
                score += 0.2
                break
    return min(1.0, score)


def build_implementation_candidates(
    repo_urls: List[str],
    code_snippets: List[CodeSnippet],
    content: str,
    search_results: List[dict],
    query: str,
    constraints: SearchConstraints,
    min_confidence: float,
) -> List[ImplementationCandidate]:
    """Build and filter ImplementationCandidate list with confidence scoring."""
    candidates: List[ImplementationCandidate] = []
    for i, repo_url in enumerate(repo_urls[:5]):
        snippet_subset = code_snippets[:3] if code_snippets else []
        sr_snippet = None
        for sr in search_results:
            if sr.get("url") and repo_url in (sr.get("url") or ""):
                sr_snippet = sr.get("snippet")
                break
        conf = score_implementation_candidate(
            repo_url, query, constraints, content, snippet_subset, sr_snippet
        )
        if conf < min_confidence:
            continue
        candidates.append(
            ImplementationCandidate(
                source_repo_url=repo_url,
                commit_or_ref=None,
                language="python",
                entrypoint_guess=None,
                dependency_hints=[],
                license_text_or_url=None,
                snippets=snippet_subset,
                confidence=conf,
                algorithm_name_match=constraints.algorithm_names[0] if constraints.algorithm_names else None,
                problem_class_match=constraints.problem_class,
            )
        )
    if not candidates and code_snippets:
        conf = score_implementation_candidate("", query, constraints, content, code_snippets, None)
        if conf >= min_confidence:
            candidates.append(
                ImplementationCandidate(
                    source_repo_url="",
                    language="python",
                    snippets=code_snippets[:5],
                    confidence=conf,
                )
            )
    return sorted(candidates, key=lambda c: c.confidence, reverse=True)


def build_sandbox_payload_from_result(result: SearchResult) -> SandboxPayload:
    """
    Build Claude/Modal sandbox payload from a SearchResult when found_implementation is True.
    """
    if not result.found_implementation or not result.implementations:
        return SandboxPayload(
            query=result.query,
            code_units=[],
            provenance={**(result.provenance or {}), "query": result.query, "note": "no_implementation"},
        )
    dep_hints: List[str] = []
    license_url: Optional[str] = None
    code_units: List[SandboxCodeUnit] = []
    for impl in result.implementations:
        for sn in impl.snippets:
            code_units.append(
                SandboxCodeUnit(
                    content=sn.content,
                    language=sn.language,
                    file_path=sn.file_path,
                    source_repo_url=impl.source_repo_url or None,
                    commit_or_ref=impl.commit_or_ref,
                    entrypoint_guess=impl.entrypoint_guess,
                )
            )
        dep_hints.extend(impl.dependency_hints or [])
        if impl.license_text_or_url and not license_url:
            license_url = impl.license_text_or_url
    return SandboxPayload(
        query=result.query,
        code_units=code_units,
        dependency_hints=list(dict.fromkeys(dep_hints)),
        license_text_or_url=license_url,
        provenance=result.provenance if result.provenance else {"query": result.query},
    )
