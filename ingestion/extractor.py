"""
extractor.py
────────────
Extracts raw text from the Constitution of India PDF.

Responsibilities:
  1. Open the PDF with PyMuPDF
  2. Extract text page by page
  3. Remove the Table of Contents section so its article numbers
     don't interfere with the chunker
  4. Basic text cleaning (fix hyphenated line breaks, normalise whitespace)
"""

import re
import fitz                          # PyMuPDF
from ingestion.logger import get_logger

log = get_logger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_body_text(pdf_path: str) -> str:
    """
    Full pipeline:
        PDF  →  raw text  →  TOC removed  →  cleaned body text

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        Cleaned body text starting from PART I of the Constitution.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        RuntimeError:      If the PDF cannot be opened or has no pages.
    """
    raw_text = _extract_raw_text(pdf_path)
    body_text = _remove_toc(raw_text)
    clean_text = _clean_text(body_text)
    return clean_text


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_raw_text(pdf_path: str) -> str:
    """Open the PDF and concatenate text from every page."""
    import os
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Cannot open PDF: {exc}") from exc

    if len(doc) == 0:
        raise RuntimeError("PDF has no pages.")

    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append(text)
        log.debug("Page %d extracted (%d chars)", i + 1, len(text))

    doc.close()
    full_text = "\n".join(pages)
    log.info("Extracted %d pages | %d total characters", len(pages), len(full_text))
    return full_text


def _remove_toc(text: str) -> str:
    """
    The PDF begins with a multi-page Table of Contents that lists:
        364.   Special provisions as to major ports...

    If we don't strip this section the chunker will create false
    article chunks from the TOC entries.

    Strategy: find the first occurrence of the PART I heading that is
    immediately followed by 'THE UNION AND ITS TERRITORY' — that marks
    the start of the actual constitutional body text.
    """
    # Primary pattern: PART I header as it appears in the body
    patterns = [
        r'PART\s+I\s*\n\s*THE UNION AND ITS TERRITORY',
        r'PART\s+I\b.*?THE UNION AND ITS TERRITORY',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            body = text[match.start():]
            log.info(
                "TOC removed — body starts at char %d "
                "(matched pattern: %r)",
                match.start(), pattern
            )
            return body

    # Fallback: if we cannot find the marker, return full text with a warning
    log.warning(
        "Could not find PART I / THE UNION AND ITS TERRITORY marker. "
        "Returning full text — TOC entries may create spurious chunks."
    )
    return text


def _clean_text(text: str) -> str:
    """
    Light cleaning that preserves article structure:

    1. Re-join words broken across lines with a hyphen
       e.g. "terri-\ntory"  →  "territory"
    2. Collapse runs of blank lines to a single blank line
    3. Strip trailing whitespace from each line
    4. Normalise non-breaking spaces and other Unicode spaces to plain space
    """
    # 1. Re-join hyphenated line-breaks
    text = re.sub(r'-\n(\S)', r'\1', text)

    # 2. Normalise various Unicode whitespace to plain space
    text = re.sub(r'[\u00a0\u2009\u202f\u3000]', ' ', text)

    # 3. Strip trailing spaces per line
    lines = [line.rstrip() for line in text.splitlines()]

    # 4. Collapse 3+ consecutive blank lines to 2
    cleaned_lines = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 2:
                cleaned_lines.append(line)
        else:
            blank_run = 0
            cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)
    log.info("Text cleaned — final length: %d chars", len(result))
    return result