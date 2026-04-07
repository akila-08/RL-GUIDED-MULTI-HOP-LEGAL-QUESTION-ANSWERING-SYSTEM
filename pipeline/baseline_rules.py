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
    r"Speaker", r"Deputy Speaker"
]

ARTICLE_HINTS = {
    "14": "equality before the law",
    "15": "prohibition of discrimination",
    "16": "equality of opportunity in public employment",
    "16(2)": "prohibition of discrimination in public employment",
    "21": "protection of life and personal liberty",
    "83": "duration of Houses of Parliament",
    "67": "term of office of the Vice-President",
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

def extract_key_phrases_v2(question):
    q = question.strip()
    q_lower = q.lower()
    found_phrases = []
    for pattern in LEGAL_PHRASE_PATTERNS:
        if re.search(pattern, q, flags=re.IGNORECASE):
            found_phrases.append(pattern)
    domain_keywords = [
        "duration", "citizenship", "equality", "discrimination",
        "fairness", "absence", "presiding officer", "executive power",
        "appointment", "tenure", "removal", "oath", "rule-making powers"
    ]
    for kw in domain_keywords:
        if kw.lower() in q_lower and kw not in found_phrases:
            found_phrases.append(kw)
    cap_phrases = re.findall(r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', q)
    for cp in cap_phrases:
        if cp not in found_phrases:
            found_phrases.append(cp)
    if not found_phrases:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', q_lower)
        filtered = [w for w in words if w not in STOPWORDS]
        found_phrases.extend(list(dict.fromkeys(filtered[:3])))
    return found_phrases[:4]

def detect_focus(question):
    q = question.lower()
    focus_keywords = [
        "duration", "citizenship", "fairness", "equality", "discrimination",
        "appointment", "removal", "oath", "tenure", "salary", "allowances",
        "power", "rule-making powers", "executive power", "secretarial staff",
        "presiding officer", "absence", "office", "conditions of service",
        "validity", "disqualification", "special address", "messages",
        "language", "rights", "duties", "procedure", "expenditure", "grants"
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
            sub_questions.append("What is the constitutional rule?")
            sub_questions.append("What is the trigger event?")
            sub_questions.append("What is the legal outcome?")

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
            sub_questions.append("What is the first constitutional concept in question?")
            sub_questions.append("What is the second constitutional concept in question?")
            sub_questions.append("How do they relate?")

    sub_questions = [clean_text(sq) for sq in sub_questions]
    return {"sub_questions": sub_questions}
