"""Claude code generator: produces algorithm implementations from parsed research paper content."""

import re
from typing import Optional

import anthropic

from benchwarmer.config import SearchConfig, get_search_config
from benchwarmer.schemas.search import GeneratedCode, PDFContent


# Maximum characters of paper text to include in the prompt (avoid blowing context window)
_MAX_PAPER_TEXT = 12_000


def _build_generation_prompt(pdf_content: PDFContent) -> str:
    """Build a detailed prompt for Claude to implement the algorithm from the paper."""
    # Truncate full text to fit context
    paper_text = pdf_content.full_text[:_MAX_PAPER_TEXT]
    if len(pdf_content.full_text) > _MAX_PAPER_TEXT:
        paper_text += "\n[... paper text truncated ...]"

    prompt = f"""You are an expert algorithm engineer. A user has uploaded a research paper and needs a working Python implementation of the algorithm(s) described in it.

## Paper Information
**Title:** {pdf_content.title}
"""
    if pdf_content.abstract:
        prompt += f"\n**Abstract:** {pdf_content.abstract}\n"

    if pdf_content.algorithm_keywords:
        prompt += f"\n**Identified algorithms/keywords:** {', '.join(pdf_content.algorithm_keywords)}\n"

    prompt += f"""
## Paper Text (excerpt)
{paper_text}

## Instructions
1. Implement the **core algorithm(s)** described in this paper as clean, well-documented Python code.
2. Include type hints, docstrings, and inline comments explaining key steps.
3. Make the implementation self-contained and runnable — include any helper functions needed.
4. If the paper describes multiple algorithms or variants, implement the main/primary one.
5. Include a simple usage example or `if __name__ == "__main__"` block at the bottom.
6. List any required pip packages at the top as comments (e.g. `# pip install numpy scipy`).

Return ONLY the Python implementation code in a single fenced code block (```python ... ```).
After the code block, provide a brief (2-3 sentence) explanation of what the code does.
"""
    return prompt


def _extract_code_and_explanation(response_text: str) -> tuple[str, str]:
    """Extract the Python code block and explanation from Claude's response."""
    # Extract code block
    code_match = re.search(r"```python\s*\n(.*?)```", response_text, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        # Everything after the code block is the explanation
        explanation = response_text[code_match.end():].strip()
    else:
        # Fallback: treat entire response as code if no fence found
        code = response_text.strip()
        explanation = ""

    return code, explanation


def _extract_pip_dependencies(code: str) -> list[str]:
    """Extract pip dependency hints from code comments like '# pip install numpy scipy'."""
    deps: list[str] = []
    for m in re.finditer(r"#\s*pip\s+install\s+(.+)", code, re.IGNORECASE):
        for pkg in m.group(1).split():
            pkg = pkg.strip().rstrip(",;")
            if pkg and pkg not in deps:
                deps.append(pkg)
    return deps


def generate_code_from_paper(
    pdf_content: PDFContent,
    config: Optional[SearchConfig] = None,
) -> GeneratedCode:
    """
    Use Claude to generate a Python implementation of the algorithm described
    in the parsed research paper.
    """
    config = config or get_search_config()

    if not config.anthropic_api_key:
        return GeneratedCode(
            code="# Error: ANTHROPIC_API_KEY not set",
            language="python",
            explanation="Cannot generate code: Anthropic API key is missing.",
            paper_title=pdf_content.title,
        )

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    prompt = _build_generation_prompt(pdf_content)

    message = client.messages.create(
        model=config.anthropic_model,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text if message.content else ""
    code, explanation = _extract_code_and_explanation(response_text)
    deps = _extract_pip_dependencies(code)

    # Try to identify the primary algorithm name
    algo_name = pdf_content.algorithm_keywords[0] if pdf_content.algorithm_keywords else pdf_content.title

    return GeneratedCode(
        code=code,
        language="python",
        explanation=explanation,
        dependency_hints=deps,
        paper_title=pdf_content.title,
        algorithm_name=algo_name,
    )
