import re

STOPWORDS = {
    "how", "does", "do", "is", "are", "the", "of", "and", "or", "to", "in", "for",
    "what", "when", "if", "a", "an", "their", "they", "together", "under", "with",
    "from", "by", "on", "at", "this", "that", "these", "those", "it", "as"
}

LEGAL_PHRASE_PATTERNS = [
    r"Council of States", r"House of the People", r"public employment",
    r"Indian citizenship", r"foreign State", r"foreign citizenship",
    r"High Court", r"Supreme Court", r"Legislative Assembly",
    r"Legislative Council", r"State Election Commissioner",
    r"Comptroller and Auditor-General", r"Chief Justice of India",
    r"Fundamental Rights", r"village panchayats", r"secretarial staff",
    r"Consolidated Fund of a State", r"Governor", r"President",
    r"Speaker", r"Deputy Speaker",
    r"original jurisdiction", r"exclusive jurisdiction", r"appellate jurisdiction",
    r"legal dispute", r"inter-state dispute", r"writ jurisdiction",
    r"Parliament", r"Union of India",
]

ARTICLE_HINTS = {
    "14": "equality before the law",
    "15": "prohibition of discrimination",
    "16": "equality of opportunity in public employment",
    "16(2)": "prohibition of discrimination in public employment",
    "21": "protection of life and personal liberty",
    "83": "duration of Houses of Parliament",
    "67": "term of office of the Vice-President",
    "131": "original jurisdiction of the Supreme Court in inter-state and Union-State disputes",
    "132": "appellate jurisdiction of the Supreme Court in constitutional matters",
    "133": "appellate jurisdiction of the Supreme Court in civil matters",
    "153": "Governor for each State",
    "180(2)": "presiding officer of Legislative Assembly in absence of Speaker and Deputy Speaker",
    "225": "rule-making powers of the High Court",
    "243K(2)": "conditions of service and tenure of the State Election Commissioner",
    "148(1)": "removal process of the Comptroller and Auditor-General",
    "148(2)": "oath requirement of the Comptroller and Auditor-General",
    "100(1)": "voting rights and casting vote in Parliament",
    "98": "secretarial staff of Parliament",
    "187(2)": "recruitment and conditions of service of secretarial staff of State Legislature",
    "9": "effect of acquiring foreign citizenship on Indian citizenship"
}

def extract_articles(question):
    pattern = r'Article[s]?\s+((?:\d+[A-Z]?(?:\([^)]+\))?)(?:\s*(?:,|and)\s*\d+[A-Z]?(?:\([^)]+\))?)*)'
    matches = re.findall(pattern, question, flags=re.IGNORECASE)
    articles = []
    for match in matches:
        nums = re.findall(r'\d+[A-Z]?(?:\([^)]+\))?', match)
        articles.extend(nums)
    return articles

def infer_question_type(question):
    q = question.lower()
    if any(word in q for word in ["difference", "compare", "differ", "distinction", "versus", "as opposed to"]):
        return "comparative"
    if any(word in q for word in ["if", "when", "in the absence", "what happens", "under what condition", "unless"]):
        return "conditional"
    return "analytical"

# ── Legal vocabulary for entity-aware extraction ──────────────────────────────

# Noun heads that signal a legal concept even without an article number
_LEGAL_ROLE_HEADS = {
    # Jurisdiction / power
    "jurisdiction", "power", "authority", "competence",
    # Rights / duties
    "right", "rights", "duty", "duties", "obligation", "privilege", "immunity",
    # Procedural
    "procedure", "process", "mechanism", "remedy", "appeal", "petition",
    "election", "appointment", "removal", "disqualification", "dissolution",
    # Constitutional bodies / offices
    "court", "tribunal", "commission", "parliament", "legislature",
    "president", "governor", "speaker", "minister", "council",
    "committee", "assembly", "house",
    # Legal relationships
    "dispute", "conflict", "violation", "breach", "enforcement",
    "interpretation", "validity", "immunity",
    # Scope qualifiers used as noun heads
    "scope", "extent", "nature", "basis", "ground", "condition",
}

# Adjectives that, when paired with a noun head, form a meaningful legal phrase
_LEGAL_QUALIFIERS = {
    "original", "exclusive", "appellate", "writ", "inherent", "supervisory",
    "fundamental", "constitutional", "statutory", "civil", "criminal",
    "absolute", "qualified", "special", "general", "concurrent",
    "independent", "inter-state", "federal", "union", "state",
    "public", "private", "parliamentary", "legislative", "executive",
    "judicial", "administrative",
}

# Stop-words for the phrase scorer (broader than the global set)
_SCORE_STOPWORDS = {
    "how", "does", "do", "is", "are", "was", "were", "the", "of", "and",
    "or", "to", "in", "for", "what", "when", "if", "a", "an", "their",
    "they", "together", "under", "with", "from", "by", "on", "at", "this",
    "that", "these", "those", "it", "as", "be", "which", "have", "has",
    "had", "been", "would", "could", "should", "explain", "describe",
    "discuss", "why", "where", "who", "can", "will", "shall", "may",
    "applies", "apply", "give", "state", "states", "used", "using",
}


def _score_phrase(phrase: str) -> int:
    """
    Higher = more informative.
    Multi-word phrases > single words.
    Legal role heads score +2. Legal qualifiers score +1 each.
    """
    words = phrase.lower().split()
    score = len(words)  # multi-word bonus
    for w in words:
        if w in _LEGAL_ROLE_HEADS:
            score += 2
        if w in _LEGAL_QUALIFIERS:
            score += 1
    return score


def extract_key_phrases_v2(question: str):
    """
    Extract informative legal concepts / entities from a question WITHOUT
    relying on explicit article numbers.

    Uses 5 complementary layers — all regex-based, zero external deps:

    Layer 1 — Known legal institution / phrase patterns (LEGAL_PHRASE_PATTERNS)
      e.g. "Supreme Court", "original jurisdiction", "Council of States"

    Layer 2 — Multi-word Noun Phrase chunking
      Captures consecutive Capitalised words and
      Adjective + Noun compounds (title-case or legal-qualifier + noun-head).
      e.g. "original jurisdiction", "exclusive power", "State Legislature"

    Layer 3 — Legal role vocabulary scan
      Any single word in _LEGAL_ROLE_HEADS that appears in the question
      and is NOT already covered by a longer phrase.
      e.g. "jurisdiction", "dispute", "removal"

    Layer 4 — Verb-Object extraction
      Captures "verb + noun/phrase" pairs that encode the KEY ACTION of the
      question: "explain how … applies", "why it is exclusive".
      e.g. "applies", "exclusive" become seeds for focused sub-questions.

    Layer 5 — Adjective-Noun compound detection (lower-case)
      Regex scan for (legal-qualifier) + (1-2 words) to catch multi-word
      concepts that are fully lower-case in the question.
      e.g. "exclusive jurisdiction", "inter-state dispute"

    Returns: deduplicated list of up to 5 phrases, ranked by informativeness.
    """
    q = question.strip()
    q_lower = q.lower()
    candidates: list[tuple[str, int]] = []  # (phrase, score)
    seen_lower: set[str] = set()

    def _add(phrase: str) -> None:
        """Add a candidate, deduplicating by lowercase and sub-string."""
        phrase = phrase.strip()
        if not phrase or len(phrase) < 3:
            return
        pl = phrase.lower()
        # Skip if already covered by a longer candidate
        if any(pl in s for s in seen_lower):
            return
        # Remove shorter candidates that are sub-strings of this one
        nonlocal candidates
        candidates = [(p, sc) for p, sc in candidates if p.lower() not in pl]
        seen_lower.add(pl)
        candidates.append((phrase, _score_phrase(phrase)))

    # ── Layer 1: Known legal institution / phrase patterns ────────────────────
    for pattern in LEGAL_PHRASE_PATTERNS:
        if re.search(pattern, q, flags=re.IGNORECASE):
            # Use the matched text (preserves original casing)
            m = re.search(pattern, q, flags=re.IGNORECASE)
            if m:
                _add(m.group(0))

    # ── Layer 2: Multi-word noun phrase chunking ──────────────────────────────
    # 2a. Consecutive title-case words (proper noun phrases)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', q):
        _add(m.group(1))

    # 2b. legal-qualifier (lower or title-case) + noun head (1-2 words)
    qual_pat = r'\b(' + '|'.join(_LEGAL_QUALIFIERS) + r')' \
               r'\s+([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)?)\b'
    for m in re.finditer(qual_pat, q_lower):
        candidate = m.group(0)
        # Only keep if the last word is a meaningful noun (not a stopword)
        last_word = candidate.split()[-1]
        if last_word not in _SCORE_STOPWORDS:
            _add(candidate)

    # ── Layer 3: Legal role vocabulary scan ───────────────────────────────────
    # Add any role-head word NOT already embedded in a longer phrase
    for head in _LEGAL_ROLE_HEADS:
        # Match as whole word
        if re.search(r'\b' + re.escape(head) + r'\b', q_lower):
            _add(head)  # _add will skip if covered by longer phrase

    # ── Layer 4: Verb-Object extraction ───────────────────────────────────────
    # Pattern: (action verb) followed by a noun phrase up to 4 words
    # Focuses on what the question is ASKING the agent to reason about.
    verb_obj_pat = (
        r'\b(?:applies?|grants?|confers?|vests?|limits?|bars?|prevents?|'
        r'ensures?|guarantees?|prohibits?|allows?|restricts?|excludes?|'
        r'empowers?)\s+([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+){0,3})\b'
    )
    for m in re.finditer(verb_obj_pat, q_lower):
        obj = m.group(1).strip()
        if obj not in _SCORE_STOPWORDS and len(obj.split()) >= 1:
            _add(obj)

    # Also capture the complement of "is exclusive / is original / is appellate"
    for m in re.finditer(
        r'\bis\s+(' + '|'.join(_LEGAL_QUALIFIERS) + r')\b', q_lower
    ):
        _add(m.group(1))

    # ── Layer 5: Adjective-Noun compounds (fully lower-case in question) ──────
    for m in re.finditer(
        r'\b(' + '|'.join(_LEGAL_QUALIFIERS) + r')\s+'
        r'([a-z][a-z]+(?:\s+[a-z][a-z]+)?)\b',
        q_lower,
    ):
        last = m.group(0).split()[-1]
        if last not in _SCORE_STOPWORDS:
            _add(m.group(0))

    # ── Fallback: top informative single words if nothing extracted ───────────
    if not candidates:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', q)
        for w in words:
            if w.lower() not in _SCORE_STOPWORDS:
                _add(w)

    # ── Rank and return top 5 ─────────────────────────────────────────────────
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [phrase for phrase, _ in candidates[:5]]




def detect_focus(question):
    q = question.lower()
    focus_keywords = [
        "original jurisdiction", "exclusive jurisdiction", "appellate jurisdiction",
        "duration", "citizenship", "fairness", "equality", "discrimination",
        "appointment", "removal", "oath", "tenure", "salary", "allowances",
        "power", "rule-making powers", "executive power", "secretarial staff",
        "presiding officer", "absence", "office", "conditions of service",
        "validity", "disqualification", "special address", "messages",
        "language", "rights", "duties", "procedure", "expenditure", "grants",
        "legal dispute", "inter-state", "jurisdiction",
    ]
    for kw in sorted(focus_keywords, key=len, reverse=True):
        if kw in q:
            return kw
    return None

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r',\s*([A-Za-z ]+)\s*,\s*\1$', r', \1', text, flags=re.IGNORECASE)
    words = text.split()
    cleaned = []
    for w in words:
        if not cleaned or cleaned[-1].lower() != w.lower():
            cleaned.append(w)
    text = " ".join(cleaned)
    text = re.sub(r'\s+([,?.])', r'\1', text)
    return text

def get_article_hint(article):
    return ARTICLE_HINTS.get(article, None)

def dataset_style_decompose_v3(question, question_type=None):
    if question_type is None:
        question_type = infer_question_type(question)
    articles = extract_articles(question)
    concepts = extract_key_phrases_v2(question)
    focus = detect_focus(question)
    sub_questions = []

    if question_type == "comparative":
        if len(articles) >= 2:
            if focus:
                sub_questions.append(f"What is the scope of {focus} under Article {articles[0]}?")
                sub_questions.append(f"What is the scope of {focus} under Article {articles[1]}?")
                sub_questions.append(f"Based on their treatment of {focus}, how do Article {articles[0]} and Article {articles[1]} differ?")
            else:
                sub_questions.append(f"What does Article {articles[0]} provide?")
                sub_questions.append(f"What does Article {articles[1]} provide?")
                sub_questions.append(f"How do the provisions of Article {articles[0]} and Article {articles[1]} differ?")
        elif len(concepts) >= 2:
            c1, c2 = concepts[0], concepts[1]
            if focus == "duration":
                sub_questions.append(f"What is the prescribed duration for the {c1}?")
                sub_questions.append(f"What is the prescribed duration for the {c2}?")
                sub_questions.append(f"Based on their respective durations, what are the key differences between the {c1} and the {c2}?")
            else:
                sub_questions.append(f"What is the constitutional position of {c1} in relation to {focus or 'the issue raised'}?")
                sub_questions.append(f"What is the constitutional position of {c2} in relation to {focus or 'the issue raised'}?")
                sub_questions.append(f"How do {c1} and {c2} differ in the context of {focus or 'the original question'}?")
        else:
            sub_questions.append("What is the first conceptual entity in the question?")
            sub_questions.append("What is the second conceptual entity in the question?")
            sub_questions.append("How do these differ?")

    elif question_type == "conditional":
        if len(articles) >= 1:
            art = articles[0]
            hint = get_article_hint(art)
            if hint:
                sub_questions.append(f"What does Article {art} state regarding {hint}?")
                sub_questions.append(f"What condition, exception, or event is relevant under Article {art}?")
                sub_questions.append(f"Based on Article {art}, what happens when that condition or event occurs?")
            elif focus:
                sub_questions.append(f"What does Article {art} state regarding {focus}?")
                sub_questions.append(f"What condition, exception, or event concerning {focus} is relevant under Article {art}?")
                sub_questions.append(f"Based on Article {art}, what happens when that condition or event occurs?")
            else:
                sub_questions.append(f"What rule is stated in Article {art}?")
                sub_questions.append(f"What condition or exception is relevant under Article {art}?")
                sub_questions.append(f"Based on Article {art}, what is the legal outcome when that condition occurs?")
        elif len(concepts) >= 2:
            c1, c2 = concepts[0], concepts[1]
            if focus == "citizenship":
                sub_questions.append(f"What is the constitutional rule regarding {c1}?")
                sub_questions.append(f"What is the effect of voluntarily acquiring citizenship of a {c2}?")
                sub_questions.append(f"Based on the constitutional provisions, what happens to {c1} when that condition occurs?")
            elif "speaker" in c1.lower():
                sub_questions.append(f"What constitutional provision governs the role of the {c1} during a sitting?")
                sub_questions.append(f"What condition or event involving {c2} is described in the question?")
                sub_questions.append("When that condition occurs, what arrangement is made for the presiding officer?")
            else:
                sub_questions.append(f"What legal rule applies to {c1}?")
                sub_questions.append(f"What condition or triggering event involving {c2} is described in the question?")
                sub_questions.append("Based on the constitutional provisions, what is the legal outcome when that condition occurs?")
        else:
            # Fallback: ground the sub-questions in the actual question text
            # instead of producing generic placeholders.
            q_snippet = question.strip().rstrip('?').strip()
            sub_questions.append(f"What constitutional provision or rule is relevant to: {q_snippet}?")
            sub_questions.append(f"What condition or situation triggers that provision in the context of: {q_snippet}?")
            sub_questions.append(f"What is the legal outcome or consequence under that provision for: {q_snippet}?")

    else:
        if len(articles) >= 2:
            for art in articles[:3]:
                hint = get_article_hint(art)
                if hint:
                    sub_questions.append(f"What does Article {art} provide regarding {hint}?")
                elif focus:
                    sub_questions.append(f"What does Article {art} provide regarding {focus}?")
                else:
                    sub_questions.append(f"What does Article {art} provide?")
            if focus and concepts:
                unique_concepts = [c for c in concepts if c.lower() != focus.lower()]
                if unique_concepts:
                    sub_questions.append(f"How do these provisions together address {focus} in the context of {', '.join(unique_concepts)}?")
                else:
                    sub_questions.append(f"How do these provisions together address {focus} in the original question?")
            else:
                sub_questions.append("How do these constitutional provisions work together to answer the original question?")
        elif len(concepts) >= 2:
            c1, c2 = concepts[0], concepts[1]
            if "public employment" in [c.lower() for c in concepts] and focus == "fairness":
                sub_questions.append("What constitutional principle ensures equality in public employment?")
                sub_questions.append("What constitutional principle prohibits discrimination in public employment?")
                sub_questions.append("How do these constitutional principles together ensure fairness in public employment?")
            else:
                sub_questions.append(f"What constitutional principle or provision is related to {c1}?")
                sub_questions.append(f"What constitutional principle or provision is related to {c2}?")
                if len(concepts) >= 3:
                    sub_questions.append(f"What role does {concepts[2]} play in the constitutional context of the question?")
                sub_questions.append(f"How do these principles or provisions collectively address {focus or 'the original question'}?")
        else:
            # Fallback: ground in question instead of generic placeholders.
            q_snippet = question.strip().rstrip('?').strip()
            sub_questions.append(f"What is the primary constitutional concept or provision relevant to: {q_snippet}?")
            sub_questions.append(f"What is the secondary constitutional concept or provision relevant to: {q_snippet}?")
            sub_questions.append(f"How do these constitutional concepts or provisions together answer: {q_snippet}?")

    sub_questions = [clean_text(sq) for sq in sub_questions]
    return {"sub_questions": sub_questions}


# ── Decomposition Quality Checker ─────────────────────────────────────────────

def check_decomposition(question: str, question_type: str = None) -> None:
    """
    Diagnostic helper — call this to inspect why decomposition produces
    certain sub-questions for a given input.

    Prints:
      • Detected question type
      • Articles extracted
      • Key concepts extracted
      • Detected focus keyword
      • Generated sub-questions
      • ROUGE-L recall of joined sub-qs vs original question
      • Keyword coverage score
      • Pass/fail verdict for each quality threshold

    Usage
    -----
    from pipeline.baseline_rules import check_decomposition
    check_decomposition("If two states have a legal dispute, explain how the "
                        "original jurisdiction of the Supreme Court applies "
                        "and why it is exclusive.")
    """
    import re as _re

    # ── Step 1: feature extraction ────────────────────────────────────────────
    detected_type = question_type or infer_question_type(question)
    articles      = extract_articles(question)
    concepts      = extract_key_phrases_v2(question)
    focus         = detect_focus(question)

    print("=" * 70)
    print("DECOMPOSITION QUALITY REPORT")
    print("=" * 70)
    print(f"Question      : {question}")
    print(f"Question type : {detected_type}")
    print(f"Articles found: {articles or '(none)'}")
    print(f"Concepts found: {concepts or '(none)'}")
    print(f"Focus keyword : {focus or '(none)'}")
    print()

    # ── Step 2: run decomposer ────────────────────────────────────────────────
    result    = dataset_style_decompose_v3(question, question_type)
    sub_qs    = result["sub_questions"]

    print(f"Sub-questions ({len(sub_qs)} produced):")
    for i, sq in enumerate(sub_qs, 1):
        print(f"  [{i}] {sq}")
    print()

    # ── Step 3: ROUGE-L recall ────────────────────────────────────────────────
    rouge_score = 0.0
    try:
        from rouge_score import rouge_scorer as _rs
        scorer = _rs.RougeScorer(["rougeL"], use_stemmer=True)
        joined = " ".join(sub_qs)
        result_r = scorer.score(question, joined)
        rouge_score = float(result_r["rougeL"].recall)
    except Exception as e:
        print(f"  [WARN] ROUGE computation failed: {e}")

    # ── Step 4: keyword coverage ──────────────────────────────────────────────
    STOPWORDS_LOCAL = {
        "how", "does", "do", "is", "are", "the", "of", "and", "or", "to",
        "in", "for", "what", "when", "if", "a", "an", "their", "they",
        "together", "under", "with", "from", "by", "on", "at", "this",
        "that", "these", "those", "it", "as", "be", "which", "have",
        "has", "had", "been", "would", "could", "should",
    }
    q_tokens = {
        w.lower() for w in _re.findall(r'\b[a-zA-Z]{3,}\b', question)
        if w.lower() not in STOPWORDS_LOCAL
    }
    sub_blob = " ".join(sub_qs).lower()
    if q_tokens:
        found    = sum(1 for t in q_tokens if t in sub_blob)
        coverage = found / len(q_tokens)
        missing  = [t for t in q_tokens if t not in sub_blob]
    else:
        coverage = 1.0
        missing  = []

    # ── Step 5: atomicity ─────────────────────────────────────────────────────
    atomic_issues = []
    for i, sq in enumerate(sub_qs[:-1], 1):
        if sq.count('?') > 1:
            atomic_issues.append(f"  Sub-q [{i}] contains multiple '?' — not atomic.")
        if len(sq.split()) > 25:
            atomic_issues.append(f"  Sub-q [{i}] is {len(sq.split())} words — exceeds 25-word limit.")

    # ── Step 6: count check ───────────────────────────────────────────────────
    DECOMP_MIN = 2
    DECOMP_MAX = 5
    ROUGE_THRESH    = 0.35
    COVERAGE_THRESH = 0.50
    count_ok    = DECOMP_MIN <= len(sub_qs) <= DECOMP_MAX
    rouge_ok    = rouge_score  >= ROUGE_THRESH
    coverage_ok = coverage     >= COVERAGE_THRESH
    atomic_ok   = len(atomic_issues) == 0
    overall_ok  = count_ok and rouge_ok and coverage_ok and atomic_ok

    print("Quality Metrics:")
    tick = lambda ok: "PASS" if ok else "FAIL"
    print(f"  Count     : {len(sub_qs)} sub-questions  [{tick(count_ok)}]  (expected {DECOMP_MIN}-{DECOMP_MAX})")
    print(f"  ROUGE-L   : {rouge_score:.3f}            [{tick(rouge_ok)}]  (threshold {ROUGE_THRESH})")
    print(f"  Coverage  : {coverage:.3f}               [{tick(coverage_ok)}]  (threshold {COVERAGE_THRESH})")
    print(f"  Atomic    : {'yes' if atomic_ok else 'no'}                   [{tick(atomic_ok)}]")
    if missing:
        print(f"  Missing keywords from sub-qs: {missing}")
    if atomic_issues:
        for issue in atomic_issues:
            print(issue)
    print()
    print(f"Overall verdict: {'ACCEPTED (good decomposition)' if overall_ok else 'REJECTED (fallback will trigger)'}")
    print("=" * 70)
