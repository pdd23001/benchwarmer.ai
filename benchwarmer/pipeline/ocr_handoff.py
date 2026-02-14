"""OCR handoff contract and queue item shape when no GitHub implementation is found."""

from dataclasses import dataclass, field
from typing import List, Optional

from benchwarmer.schemas.search import PaperCandidate


@dataclass
class OCRHandoffPayload:
    """Payload passed to OCR extraction + implementation pipeline."""

    query: str
    papers: List[PaperCandidate]
    ranked_algorithm_names: List[str] = field(default_factory=list)
    provenance: Optional[dict] = None


def create_ocr_handoff_payload(
    query: str,
    papers: List[PaperCandidate],
    ranked_algorithm_names: Optional[List[str]] = None,
    provenance: Optional[dict] = None,
) -> OCRHandoffPayload:
    """Build the OCR handoff payload for the pipeline."""
    names = ranked_algorithm_names or []
    for p in papers:
        for algo in p.candidate_algorithm_names:
            if algo and algo not in names:
                names.append(algo)
    return OCRHandoffPayload(
        query=query,
        papers=papers,
        ranked_algorithm_names=names,
        provenance=provenance,
    )
