"""PDF parsing: extract text, title, abstract, GitHub URLs, and algorithm details from uploaded research papers."""

import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from benchwarmer.schemas.search import PDFContent


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract full text from a PDF file using PyMuPDF."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.suffix.lower() == ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path.suffix}")

    doc = fitz.open(str(pdf_path))
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)


def _extract_title(full_text: str) -> str:
    """Heuristic: first non-empty line(s) before the abstract are likely the title."""
    lines = [ln.strip() for ln in full_text.split("\n") if ln.strip()]
    if not lines:
        return "Unknown Title"

    title_lines: list[str] = []
    for line in lines:
        low = line.lower()
        # Stop when we hit abstract, introduction, keywords, author affiliations, etc.
        if any(marker in low for marker in ["abstract", "introduction", "keywords", "1."]):
            break
        # Skip very short lines that look like page numbers or dates
        if len(line) < 4 or re.match(r"^\d+$", line):
            continue
        title_lines.append(line)
        # Title rarely exceeds 3 lines
        if len(title_lines) >= 3:
            break

    return " ".join(title_lines).strip() if title_lines else lines[0]


def _extract_abstract(full_text: str) -> Optional[str]:
    """Extract abstract section from paper text."""
    text_lower = full_text.lower()

    # Try to find explicit "abstract" section
    abstract_start = text_lower.find("abstract")
    if abstract_start == -1:
        return None

    # Move past the word "abstract" and any following whitespace/punctuation
    start = abstract_start + len("abstract")
    remainder = full_text[start:].lstrip(" \n\t.:-\u2013\u2014")

    # End at introduction, keywords, or next major section
    end_markers = [
        r"\bintroduction\b",
        r"\bkeywords?\b",
        r"\b1[\.\)]\s",
        r"\bI\.\s+Introduction",
    ]
    end_pos = len(remainder)
    for marker in end_markers:
        m = re.search(marker, remainder, re.IGNORECASE)
        if m and m.start() < end_pos:
            end_pos = m.start()

    abstract = remainder[:end_pos].strip()
    # Clean up: collapse whitespace
    abstract = re.sub(r"\s+", " ", abstract)
    return abstract if len(abstract) > 20 else None


_STOPWORDS = {
    # Common English stopwords
    "the", "a", "an", "our", "this", "new", "novel", "and", "or", "for", "of",
    "in", "on", "to", "with", "by", "is", "are", "was", "were", "be", "been",
    "from", "that", "which", "as", "at", "it", "its", "they", "their", "we",
    "has", "have", "had", "not", "but", "if", "can", "will", "do", "does",
    "also", "more", "than", "each", "all", "any", "some", "many", "most",
    "other", "two", "one", "three", "these", "those", "every", "both",
    "end", "use", "used", "using", "based", "set", "may", "such", "then",
    "only", "into", "over", "very", "just", "between", "through", "after",
    "before", "where", "when", "how", "what", "so", "about", "up", "out",
    "no", "yes", "per", "via",
    # Academic/filler words that aren't algorithm names
    "proposed", "existing", "state-of-the-art", "various", "numerous", "several",
    "comparable", "compared", "overall", "innovative", "fundamental", "scalable",
    "search", "optimization", "function", "reference", "distribution", "learning",
    "exact", "colony", "swarm", "evolutionary", "metaheuristic", "meta-heuristics",
    "partitioning", "approximation",
}


def _is_good_keyword(kw: str) -> bool:
    """Check if a keyword is likely a meaningful algorithm/method name."""
    key = kw.lower().strip()
    if key in _STOPWORDS:
        return False
    if len(key) < 3:
        return False
    # Skip partial words (e.g. "tion", "zation")
    if key.isalpha() and len(key) < 5 and key.islower():
        return False
    # Skip strings that look like broken OCR fragments
    if not re.match(r"^[A-Za-z0-9]", kw):
        return False
    # Must contain at least one letter
    if not any(c.isalpha() for c in kw):
        return False
    return True


def _extract_algorithm_keywords(full_text: str) -> list[str]:
    """Extract potential algorithm names and key technical terms from the paper."""
    keywords: list[str] = []

    # 1. Look for explicit "keywords" section (highest quality)
    kw_match = re.search(
        r"keywords?\s*[:\-\u2013\u2014]\s*(.+?)(?:\n\n|\n[A-Z1-9]|\bintroduction\b)",
        full_text,
        re.IGNORECASE | re.DOTALL,
    )
    if kw_match:
        raw = kw_match.group(1).strip()
        for kw in re.split(r"[;,]", raw):
            kw = kw.strip().rstrip(".")
            if 2 < len(kw) < 80:
                keywords.append(kw)

    # 2. Named algorithm patterns (e.g. "Harris Hawk Optimization", "Kernighan-Lin algorithm")
    algo_patterns = [
        r"(?:algorithm|method|approach|heuristic)\s+(?:called|named|dubbed|termed)\s+[\"']?([A-Z][\w\s\-]+?)(?:[\"'\.\,])",
        r"(?:we\s+(?:propose|present|introduce|develop))\s+(?:a\s+)?(?:novel\s+)?([A-Z][\w\s\-]+?)(?:\s+(?:algorithm|method|approach|heuristic))",
        r"\b([A-Z][\w]*(?:[\-\s][A-Z][\w]*)*)\s+(?:algorithm|optimizer|heuristic)\b",
        r"\(([A-Z][A-Z0-9\-]{1,15})\)",
    ]
    for pat in algo_patterns:
        for m in re.finditer(pat, full_text):
            name = m.group(1).strip()
            if 2 < len(name) < 60:
                keywords.append(name)

    # Deduplicate and filter
    seen: set[str] = set()
    unique: list[str] = []
    for kw in keywords:
        if not _is_good_keyword(kw):
            continue
        key = kw.lower()
        if key not in seen:
            seen.add(key)
            unique.append(kw)
    return unique


def _extract_github_urls(full_text: str) -> list[str]:
    """Extract unique GitHub repository URLs found directly in the paper text."""
    pattern = re.compile(
        r"https?://(?:www\.)?github\.com/([a-zA-Z0-9\-_.]+)/([a-zA-Z0-9\-_.]+)"
    )
    seen: set[str] = set()
    urls: list[str] = []
    for m in pattern.finditer(full_text):
        owner = m.group(1)
        repo = m.group(2)
        # Clean up trailing punctuation from PDF extraction artifacts
        repo = re.sub(r"[.\s]+$", "", repo)
        key = f"{owner.lower()}/{repo.lower()}"
        if key not in seen:
            seen.add(key)
            urls.append(f"https://github.com/{owner}/{repo}")
    return urls


def parse_pdf(pdf_path: str | Path) -> PDFContent:
    """
    Parse a research paper PDF and extract structured content.
    Returns a PDFContent with title, abstract, full text, algorithm keywords,
    and any GitHub URLs found in the paper.
    """
    full_text = extract_text_from_pdf(pdf_path)
    title = _extract_title(full_text)
    abstract = _extract_abstract(full_text)
    algorithm_keywords = _extract_algorithm_keywords(full_text)
    github_urls = _extract_github_urls(full_text)

    # Build a summary: title + abstract + keywords
    summary_parts = [f"Title: {title}"]
    if abstract:
        summary_parts.append(f"Abstract: {abstract}")
    if algorithm_keywords:
        summary_parts.append(f"Algorithms/Keywords: {', '.join(algorithm_keywords)}")
    if github_urls:
        summary_parts.append(f"GitHub: {', '.join(github_urls)}")
    summary = "\n".join(summary_parts)

    return PDFContent(
        file_path=str(pdf_path),
        title=title,
        abstract=abstract,
        full_text=full_text,
        algorithm_keywords=algorithm_keywords,
        github_urls=github_urls,
        summary=summary,
    )
