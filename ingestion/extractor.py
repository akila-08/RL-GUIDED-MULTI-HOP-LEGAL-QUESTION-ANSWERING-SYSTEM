import re
import fitz                          # PyMuPDF
from ingestion.logger import get_logger

log = get_logger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_body_text(pdf_path: str) -> str:
    raw_text = _extract_raw_text(pdf_path)
    body_text = _remove_toc(raw_text)
    
    clean_text = _clean_text(body_text)
    return clean_text


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_raw_text(pdf_path: str) -> str:
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

