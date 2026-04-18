import re
from typing import List, Dict
from ingestion.logger import get_logger
from core.config import Config

log = get_logger(__name__)


# ── Patterns ──────────────────────────────────────────────────────────────────

# Strict article pattern:
#   ^               — start of line (MULTILINE)
#   (\d+[A-Z]{0,2}) — article number: 1, 2A, 51A, 371B, 371AA
#   \.              — literal dot
#   \s+             — one or more spaces
#   [A-Z]           — title MUST start with a capital letter
#                     (footnotes start with lowercase: "Subs.", "Ins.", "Rep.")
#
# After finding candidates we do a secondary check:
#   the em-dash — must appear within the first 200 chars of the chunk
#   (all real articles have "Title.—content" structure)

ARTICLE_PATTERN = re.compile(
    r'^(\d+[A-Z]{0,2})\.\s+([A-Z])',
    re.MULTILINE
)

# Part headers
PART_PATTERN = re.compile(
    r'^(PART\s+[IVXLC]+)\s*\n\s*([A-Z][A-Z ,\-]+)',
    re.MULTILINE
)

# Em-dash variants used in the Constitution
EM_DASH_RE = re.compile(r'[—\u2014\u2013]')

def remove_toc(text: str) -> str:
    log.info("Attempting to remove Table of Contents...")

    # Strategy 1: Find the Preamble — Constitution body always starts here
    preamble = re.search(r'WE,\s+THE\s+PEOPLE\s+OF\s+INDIA', text, re.IGNORECASE)
    if preamble:
        log.info("TOC removed — body starts at Preamble (char %d)", preamble.start())
        return text[preamble.start():]

    # Strategy 2: Find the real PART I body (after TOC's PART I listing)
    # The TOC just lists article names; the real body has full sentence content
    part1_body = re.search(
        r'PART\s+I\s*\n\s*THE UNION AND ITS TERRITORY\s*\n+\s*\d+\.\s+Name and territory of the Union\.—',
        text,
        re.IGNORECASE
    )
    if part1_body:
        log.info("TOC removed — body starts at PART I real content (char %d)", part1_body.start())
        return text[part1_body.start():]

    # Strategy 3: Find first em-dash occurrence — TOC lines never have em-dashes
    first_emdash = re.search(r'[—\u2014]', text)
    if first_emdash:
        # Walk back to the start of that line
        line_start = text.rfind('\n', 0, first_emdash.start()) + 1
        log.info("TOC removed via em-dash fallback (char %d)", line_start)
        return text[line_start:]

    log.warning("Could not detect TOC — returning text as-is")
    return text
import re

def clean_pdf_text(text: str) -> str:
    # ── Step 1: Remove page header blocks ─────────────────────────────────
    # Pattern:
    #   "THE CONSTITUTION OF INDIA"  (page header)
    #   "(Part XYZ.—...)"            (running header, optional)
    #   one or more digits           (page number)
    text = re.sub(
        r'THE CONSTITUTION OF INDIA\s*\n'
        r'(?:\([^)]*[—\-][^)]*\)\s*\n)?'   # optional running header line
        r'\d+\s*\n',
        '\n',
        text
    )

    # ── Step 2: Remove footnote blocks ────────────────────────────────────
    # Everything from the dashed separator line to end of that page's footnotes
    # Footnotes always follow a line of underscores/dashes
    text = re.sub(
        r'[_\-]{10,}\s*\n'                  # separator line (________)
        r'(?:[\d]+\[?\..*\n)*'             # one or more footnote lines
        r'(?:\s*\n)*',                      # trailing blank lines
        '\n',
        text
    )

    # ── Step 3: Remove standalone page numbers ────────────────────────────
    # A lone number on its own line at the start of content
    # (appears as first token on pages like "2\nPART I\n...")
    text = re.sub(
        r'(?m)^\d{1,3}\s*\n(?=PART\s+[IVXLC]+|[A-Z]{2,})',
        '',
        text
    )

    # ── Step 4: Collapse excessive blank lines ────────────────────────────
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()
# ── Public API ────────────────────────────────────────────────────────────────


def chunk_by_article(text: str) -> List[Dict]:
    text = clean_pdf_text(text)
    text = remove_toc(text)
    # Step 1: find all candidate matches
    candidates = list(ARTICLE_PATTERN.finditer(text))
    log.info("Found %d candidate article boundaries (pre-filter)", len(candidates))

    # Step 2: filter — keep only those followed by an em-dash within 200 chars
    matches = []
    for m in candidates:
        lookahead = text[m.start(): m.start() + 200]
        if EM_DASH_RE.search(lookahead):
            matches.append(m)

    log.info("After em-dash filter: %d real article boundaries", len(matches))

    if not matches:
        log.error("No article boundaries found — check PDF extraction and TOC removal.")
        return []

    # Step 3: build part map
    part_map = _build_part_map(text)

    # Step 4: slice into chunks, deduplicate IDs
    chunks    = []
    seen_ids  = {}          # id -> count, for deduplication
    skipped   = 0

    for i, match in enumerate(matches):
        start       = match.start()
        end         = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        article_num = match.group(1)
        chunk_text  = text[start:end].strip()

        # Skip very short chunks (likely noise)
        if len(chunk_text) < Config.MIN_CHUNK_CHARS:
            log.debug("Skipping article %s — too short (%d chars)", article_num, len(chunk_text))
            skipped += 1
            continue

        # Deduplicate IDs — if same article number appears twice,
        # suffix with _b, _c, etc. (handles omitted+reinstated articles)
        base_id = f"article_{article_num}"
        if base_id in seen_ids:
            seen_ids[base_id] += 1
            chunk_id = f"{base_id}_{chr(97 + seen_ids[base_id])}"   # _b, _c ...
            log.debug("Duplicate article number %s — stored as %s", article_num, chunk_id)
        else:
            seen_ids[base_id] = 0
            chunk_id = base_id

        title = _extract_title(chunk_text, article_num)
        part  = _get_part(start, part_map)

        chunks.append({
            "id":          chunk_id,
            "article_num": article_num,
            "title":       title,
            "part":        part,
            "text":        chunk_text,
            "char_count":  len(chunk_text),
        })

    log.info(
        "Chunking complete — %d chunks created, %d skipped (too short)",
        len(chunks), skipped
    )
    _log_sample(chunks)
    return chunks


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_title(chunk_text: str, article_num: str) -> str:
    """
    Extract the article title.
    Format:  371B. Special provision for Assam.—...
    We want: Special provision for Assam
    """
    title_match = re.match(
        r'\d+[A-Z]{0,2}\.\s+(.+?)(?:\.—|—|\.?\n)',
        chunk_text
    )
    if title_match:
        title = title_match.group(1).strip()
        title = re.sub(r'[.\-]+$', '', title).strip()
        return title

    first_line = chunk_text.splitlines()[0].strip()
    first_line = re.sub(r'^\d+[A-Z]{0,2}\.\s+', '', first_line)
    return first_line[:120] if first_line else f"Article {article_num}"


def _build_part_map(text: str) -> List[tuple]:
    part_map = []
    for m in PART_PATTERN.finditer(text):
        part_label = m.group(1).strip()
        part_title = m.group(2).strip()[:60]
        part_map.append((m.start(), f"{part_label} — {part_title}"))
    log.debug("Found %d PART headers", len(part_map))
    return part_map


def _get_part(char_pos: int, part_map: List[tuple]) -> str:
    part = "PREAMBLE"
    for pos, name in part_map:
        if pos <= char_pos:
            part = name
        else:
            break
    return part


def _log_sample(chunks: List[Dict], n: int = 5) -> None:
    log.info("--- First %d chunks (sample) ---", min(n, len(chunks)))
    for c in chunks[:n]:
        log.info(
            "  Article %-6s | %-50s | %s | %d chars",
            c["article_num"], c["title"][:50], c["part"][:30], c["char_count"]
        )