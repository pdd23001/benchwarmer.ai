"""Pipeline stages: PDF parsing, OCR handoff, execution."""

from benchwarmer.pipeline.pdf_parser import parse_pdf
from benchwarmer.pipeline.ocr_handoff import OCRHandoffPayload, create_ocr_handoff_payload

__all__ = ["parse_pdf", "OCRHandoffPayload", "create_ocr_handoff_payload"]
