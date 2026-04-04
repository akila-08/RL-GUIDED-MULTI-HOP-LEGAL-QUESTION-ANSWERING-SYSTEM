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
    """
    Open the PDF and extract text page by page.

    Critically, footnotes and page headers are removed AT THE PAGE LEVEL
    before pages are joined. This is the only reliable approach because:

    - Footnotes always sit at the BOTTOM of a page, below a separator line.
      Removing them per-page is unambiguous.

    - Page headers always sit at the TOP of a page.
      Removing them per-page is unambiguous.

    - After joining, footnotes from page N appear BEFORE the continuation
      text of page N+1, making them impossible to separate cleanly from
      article body text using a single-pass regex.

    WHY the old approach failed:
      Page 32 (Preamble) ends with:
          ______________________________________________
          2. Subs. by s. 2, ibid., for "Unity of the Nation"...

      After joining, _remove_toc() cut at "PART I / THE UNION AND ITS
      TERRITORY" which is on page 33. So the preamble footnote survived
      the TOC cut and landed before Article 1 — the chunker saw it as
      Article 2 (matched regex + em-dash from next page header).
    """
    import os
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Cannot open PDF: {exc}") from exc

    if len(doc) == 0:
        raise RuntimeError("PDF has no pages.")

    # Separator line that precedes footnotes — 10+ underscores or dashes
    SEPARATOR    = re.compile(r'[_\-]{10,}')

    # Page header block that appears at the top of continuation pages:
    #   "THE CONSTITUTION OF INDIA\n(Part I.—Union and its territory)\n3\n"
    PAGE_HEADER  = re.compile(
        r'^THE CONSTITUTION OF INDIA\s*\n'   # title line
        r'\([^)]*\)\s*\n'                    # running header: "(Part X.—...)"
        r'\d+\s*\n',                         # page number on its own line
        re.MULTILINE
    )

    # Standalone page number that starts certain pages, e.g. "2\nPART I\n"
    # Only strip it when it is the very first thing on the page
    LONE_PAGE_NUM = re.compile(r'^\d{1,3}\s*\n')

    pages_cleaned = []

    for i, page in enumerate(doc):
        text = page.get_text()

        # ── Step A: Remove footnote block ─────────────────────────────────
        # Everything from the separator line to the end of the page is
        # footnote content. Trim it off cleanly.
        sep_match = SEPARATOR.search(text)
        if sep_match:
            text = text[:sep_match.start()].rstrip()

        # ── Step B: Remove page header block ──────────────────────────────
        # "THE CONSTITUTION OF INDIA / (Part X.—...) / PAGE_NUM"
        # appears at the top of continuation pages (not part-opener pages)
        text = PAGE_HEADER.sub('', text)

        # ── Step C: Remove lone page number at very start ──────────────────
        # e.g. "2\nPART I\n..." — the "2" is a page number, not content
        text = LONE_PAGE_NUM.sub('', text)

        pages_cleaned.append(text.strip())
        log.debug("Page %d cleaned (%d chars)", i + 1, len(text))

    doc.close()

    # Join with a single newline — articles now flow continuously
    full_text = "\n".join(pages_cleaned)
    log.info(
        "Extracted %d pages | %d total characters after per-page cleaning",
        len(pages_cleaned), len(full_text)
    )
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

