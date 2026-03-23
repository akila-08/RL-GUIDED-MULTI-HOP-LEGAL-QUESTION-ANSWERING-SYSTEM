"""
Legal QA Dataset Generator — ChromaDB Version (FIXED)
=======================================================
Source  : chroma.sqlite3 (cloned from GitHub repo)
Target  : 700 Simple + 233 Comparative + 233 Analytical + 234 Conditional = 1400 total
Strategy: Article-wise batching — ALL articles covered in order (no random skipping)

Fixes applied after diagnostics:
  1. Uses confirmed metadata key  → article_num
  2. Filters junk/amendment chunks → titles starting with Ins./Subs./Added by...
  3. Robust parser                 → handles ```json fences + curly quotes inside values
  4. Filters chunks with low char_count (< 80) — too short to generate from

Install: pip install google-genai chromadb
"""

import json, time, re, os, random
from google import genai
from google.genai import types
import chromadb

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GEMINI_API_KEY  = r"AIzaSyA-BRe_vvnEhuGxbiKBeFaKff97glryyNo"
GEMINI_MODEL    = "gemini-2.5-flash"

CHROMA_PATH     = r"C:\Users\sarat\OneDrive\Documents\Legal_QA_Retrival\RL-GUIDED-MULTI-HOP-LEGAL-QUESTION-ANSWERING-SYSTEM\db\constitution_db"

OUTPUT_JSON     = "qa_dataset_chroma.json"
PROGRESS_FILE   = "qa_progress_chroma.json"

TARGET_SIMPLE       = 700
TARGET_COMPARATIVE  = 233
TARGET_ANALYTICAL   = 233
TARGET_CONDITIONAL  = 234

ARTICLES_PER_BATCH  = 3   # articles grouped per API call

# Questions requested per call (slightly over to absorb duplicates)
SIMPLE_PER_CALL     = 7
COMP_PER_CALL       = 3
ANAL_PER_CALL       = 3
COND_PER_CALL       = 3

DELAY_SECONDS   = 65
MAX_RETRIES     = 5

# ─────────────────────────────────────────────
# JUNK FILTER
# Amendment footnote chunks — skip them
# ─────────────────────────────────────────────
JUNK_TITLE_PREFIXES = (
    "ins. by", "subs. by", "added by", "omitted by",
    "substituted by", "inserted by", "rep. by",
    "the constitution (", "w.e.f.", "act,"
)

def is_junk_chunk(meta):
    title = str(meta.get("title", "")).strip().lower()
    # Stub titles like "29." or "31B." — just an article number
    if re.match(r'^[\dA-Za-z]+\.$', title.strip()):
        return True
    for prefix in JUNK_TITLE_PREFIXES:
        if title.startswith(prefix):
            return True
    # Very short chunks
    if int(meta.get("char_count", 9999)) < 80:
        return True
    return False

# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────
BATCH_PROMPT = """You are a legal exam question designer for Indian Constitutional Law.

Below are excerpts from the Constitution of India covering these articles: {article_list}

Generate exactly:
- {n_simple} SIMPLE questions
- {n_comparative} COMPARATIVE complex questions
- {n_analytical} ANALYTICAL complex questions
- {n_conditional} CONDITIONAL complex questions

DEFINITIONS & EXAMPLES:

SIMPLE — Single fact, directly stated in text, Wh-questions only (no yes/no):
  "What right is given to minorities under Article 30(1)?"
  "Under which article can the President proclaim a National Emergency?"

COMPARATIVE — Compare/contrast two or more articles or rights:
  "What is the difference between the rights provided under Article 29(2) and Article 30(1)?"
  "How does the protection under Article 19(1)(a) differ from that under Article 19(1)(b)?"

ANALYTICAL — Analyse how articles work together or their combined legal effect:
  "How do Articles 29 and 30 together protect the educational rights of minorities in India?"
  "What is the combined effect of Articles 14 and 21 on personal liberty?"

CONDITIONAL — Scenario-based, if/can/what-happens-when questions:
  "Can the State deny admission to a student in a government-aided institution based on language?"
  "If a law violates Article 14, what remedy is available to the aggrieved person?"

RULES:
- Use ONLY the articles present in the excerpts below
- Spread questions across ALL the provided excerpts
- No yes/no questions in any category
- Every question must end with ?
- Do NOT repeat questions
- Use plain straight ASCII double-quotes inside the JSON — never curly or smart quotes

Excerpts:
{excerpts}

Return ONLY valid JSON — no markdown fences, no explanation, no extra text:
{{
  "simple": ["Q1?", "Q2?"],
  "comparative": ["Q1?"],
  "analytical": ["Q1?"],
  "conditional": ["Q1?"]
}}
"""

# ─────────────────────────────────────────────
# CHROMA LOADER
# ─────────────────────────────────────────────
def load_chroma_articles(chroma_path):
    print(f"📂 Connecting to ChromaDB...\n")
    client      = chromadb.PersistentClient(path=chroma_path)
    collections = client.list_collections()
    print(f"   Collections : {[c.name for c in collections]}")

    collection = None
    for c in collections:
        if any(kw in c.name.lower() for kw in ["constitution", "legal", "india"]):
            collection = client.get_collection(c.name)
            break
    if collection is None:
        collection = client.get_collection(collections[0].name)

    print(f"   Using       : '{collection.name}'  ({collection.count()} chunks)\n")

    result    = collection.get(include=["documents", "metadatas"])
    docs      = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    articles = {}
    skipped  = 0
    for text, meta in zip(docs, metadatas):
        if not text or not text.strip():
            continue
        if is_junk_chunk(meta):
            skipped += 1
            continue
        article_num = str(meta.get("article_num", "unknown")).strip()
        if article_num not in articles:
            articles[article_num] = []
        articles[article_num].append({"text": text.strip(), "meta": meta})

    print(f"   Junk chunks skipped : {skipped}")
    print(f"   Usable articles     : {len(articles)}")
    print(f"   Sample article_nums : {sorted(articles.keys(), key=article_sort_key)[:20]}\n")
    return articles

# ─────────────────────────────────────────────
# GEMINI CLIENT
# ─────────────────────────────────────────────
def setup_client():
    client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"✅ Gemini model : {GEMINI_MODEL}\n")
    return client

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"done_batches": []}

def save_progress(done_batches, dataset):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"done_batches": done_batches}, f)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

def call_gemini(client, prompt):
    wait = 70
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.85,
                    max_output_tokens=8192,
                )
            )
            return response.text
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                match = re.search(r"seconds:\s*(\d{1,4})", err)
                retry_after = min(int(match.group(1)) + 5, 120) if match else wait
                print(f"  ⏳ 429 quota — waiting {retry_after}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(retry_after)
                wait = min(wait * 2, 300)
            else:
                print(f"  ⚠️  Error attempt {attempt}: {err[:120]}")
                time.sleep(10 * attempt)
    print("  ✗ All retries exhausted — skipping batch.")
    return None

# ─────────────────────────────────────────────
# ROBUST PARSER
# Handles: ```json fences, curly quotes anywhere,
#          truncated outer braces, fallback regex
# ─────────────────────────────────────────────
def normalize_quotes(raw):
    return (raw
        .replace('\u2018', "'").replace('\u2019', "'")
        .replace('\u201a', "'").replace('\u201b', "'")
        .replace('\u201c', '"').replace('\u201d', '"')
        .replace('\u201e', '"').replace('\u201f', '"')
        .replace('\u2032', "'").replace('\u2033', '"')
        .replace('\u00ab', '"').replace('\u00bb', '"')
        .replace('\u2039', "'").replace('\u203a', "'")
    )

def sanitize_json_strings(raw):
    """
    Inside JSON string values, replace inner unescaped double-quotes
    with escaped ones so json.loads doesn't choke.
    Walks the string tracking in/out of JSON string context.
    """
    out    = []
    in_str = False
    i      = 0
    while i < len(raw):
        ch = raw[i]
        if ch == '\\' and in_str:
            out.append(ch)
            i += 1
            if i < len(raw):
                out.append(raw[i])
            i += 1
            continue
        if ch == '"':
            if not in_str:
                in_str = True
                out.append(ch)
            else:
                # Peek ahead: is the next non-whitespace a : , ] } ?
                # If yes, this is a closing quote; otherwise it's an inner quote
                j = i + 1
                while j < len(raw) and raw[j] in ' \t\n\r':
                    j += 1
                next_ch = raw[j] if j < len(raw) else ''
                if next_ch in ':,]}':
                    in_str = False
                    out.append(ch)
                else:
                    out.append('\\"')   # escape the inner quote
            i += 1
            continue
        out.append(ch)
        i += 1
    return ''.join(out)

def parse_response(raw):
    empty = [], [], [], []
    if not raw:
        return empty

    raw = raw.strip()

    # Strip markdown fences
    raw = re.sub(r'^```(?:json)?\s*\n?', '', raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r'\n?```\s*$',          '', raw).strip()

    # Normalize all curly/smart quotes to ASCII
    raw = normalize_quotes(raw)

    def clean(lst):
        return [q.strip() for q in lst if isinstance(q, str) and "?" in q]

    # Attempt 1: direct json.loads
    try:
        data = json.loads(raw)
        return (clean(data.get("simple",      [])),
                clean(data.get("comparative", [])),
                clean(data.get("analytical",  [])),
                clean(data.get("conditional", [])))
    except Exception:
        pass

    # Attempt 2: sanitize inner quotes then try again
    try:
        sanitized = sanitize_json_strings(raw)
        data = json.loads(sanitized)
        return (clean(data.get("simple",      [])),
                clean(data.get("comparative", [])),
                clean(data.get("analytical",  [])),
                clean(data.get("conditional", [])))
    except Exception:
        pass

    # Attempt 3: extract each list with regex
    try:
        result = []
        for key in ["simple", "comparative", "analytical", "conditional"]:
            m = re.search(rf'"{key}"\s*:\s*(\[.*?\])', raw, re.DOTALL)
            if m:
                try:
                    result.append(clean(json.loads(m.group(1))))
                except Exception:
                    # Try sanitizing the individual list string
                    result.append(clean(json.loads(sanitize_json_strings(m.group(1)))))
            else:
                result.append([])
        if any(result):
            return tuple(result)
    except Exception:
        pass

    # Attempt 4: last resort — grab anything that looks like a question
    try:
        all_qs = re.findall(r'"([^"]{10,300}\?)"', raw)
        if all_qs:
            print(f"  ⚠️  Fallback regex parse — recovered {len(all_qs)} questions")
            q4 = max(1, len(all_qs) // 4)
            return (all_qs[:q4], all_qs[q4:2*q4], all_qs[2*q4:3*q4], all_qs[3*q4:])
    except Exception:
        pass

    print(f"  ❌ Could not parse response. Snippet:\n{raw[:400]}\n")
    return empty

# ─────────────────────────────────────────────
# DEDUP
# ─────────────────────────────────────────────
def filter_new(questions, seen_set):
    new_qs = []
    for q in questions:
        norm = q.strip().lower()
        if norm not in seen_set:
            seen_set.add(norm)
            new_qs.append(q)
    return new_qs

# ─────────────────────────────────────────────
# ARTICLE SORT KEY
# ─────────────────────────────────────────────
def article_sort_key(name):
    nums  = re.findall(r'\d+', str(name))
    alpha = re.sub(r'\d', '', str(name)).strip().upper()
    return (int(nums[0]) if nums else 9999, alpha)

# ─────────────────────────────────────────────
# BUILD BATCHES
# ─────────────────────────────────────────────
def build_batches(articles_dict, articles_per_batch=3):
    sorted_articles = sorted(articles_dict.keys(), key=article_sort_key)
    print(f"📋 Total usable articles : {len(sorted_articles)}")
    print(f"   Order (first 20)     : {sorted_articles[:20]}\n")
    batches = []
    for i in range(0, len(sorted_articles), articles_per_batch):
        batches.append(sorted_articles[i : i + articles_per_batch])
    print(f"📦 Total batches : {len(batches)}  ({articles_per_batch} articles each)\n")
    return batches

# ─────────────────────────────────────────────
# COUNTS
# ─────────────────────────────────────────────
def counts(dataset):
    s  = sum(1 for d in dataset if d["type"] == "simple")
    cp = sum(1 for d in dataset if d["type"] == "comparative")
    an = sum(1 for d in dataset if d["type"] == "analytical")
    cd = sum(1 for d in dataset if d["type"] == "conditional")
    return s, cp, an, cd

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    articles_dict = load_chroma_articles(CHROMA_PATH)
    batches       = build_batches(articles_dict, ARTICLES_PER_BATCH)
    total_batches = len(batches)

    client       = setup_client()
    progress     = load_progress()
    done_batches = progress.get("done_batches", [])

    dataset = []
    seen    = set()
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, encoding="utf-8") as f:
            dataset = json.load(f)
        for entry in dataset:
            seen.add(entry.get("question", "").strip().lower())
        s, cp, an, cd = counts(dataset)
        print(f"↩️  Resuming — Simple={s}  Comp={cp}  Anal={an}  Cond={cd}\n")

    print(f"🎯 Targets : S={TARGET_SIMPLE} | CP={TARGET_COMPARATIVE} "
          f"| AN={TARGET_ANALYTICAL} | CD={TARGET_CONDITIONAL}\n")

    for batch_idx, article_group in enumerate(batches):

        s, cp, an, cd = counts(dataset)
        if s >= TARGET_SIMPLE and cp >= TARGET_COMPARATIVE \
                and an >= TARGET_ANALYTICAL and cd >= TARGET_CONDITIONAL:
            print(f"\n🎯 All targets reached! Stopping.")
            break

        if batch_idx in done_batches:
            print(f"[{batch_idx+1}/{total_batches}] ⏭  Skipping (already done)")
            continue

        need_s  = max(0, TARGET_SIMPLE      - s)
        need_cp = max(0, TARGET_COMPARATIVE - cp)
        need_an = max(0, TARGET_ANALYTICAL  - an)
        need_cd = max(0, TARGET_CONDITIONAL - cd)

        req_s  = min(SIMPLE_PER_CALL, need_s  + 2) if need_s  > 0 else 0
        req_cp = min(COMP_PER_CALL,   need_cp + 1) if need_cp > 0 else 0
        req_an = min(ANAL_PER_CALL,   need_an + 1) if need_an > 0 else 0
        req_cd = min(COND_PER_CALL,   need_cd + 1) if need_cd > 0 else 0

        if req_s == 0 and req_cp == 0 and req_an == 0 and req_cd == 0:
            break

        # Build excerpts
        excerpts_block     = ""
        all_articles_label = []
        for article_num in article_group:
            chunks = articles_dict.get(article_num, [])
            chunks_sorted = sorted(chunks, key=lambda c: len(c["text"]), reverse=True)
            for chunk in chunks_sorted[:3]:
                text  = chunk["text"][:700]
                title = chunk["meta"].get("title", "")
                part  = chunk["meta"].get("part",  "")
                excerpts_block += f"\n[Article {article_num} — {title} | {part}]\n{text}\n"
            all_articles_label.append(f"Article {article_num}")

        article_label = ", ".join(all_articles_label)
        print(f"[{batch_idx+1}/{total_batches}] 📄 {article_label}")
        print(f"   Requesting → S:{req_s}  CP:{req_cp}  AN:{req_an}  CD:{req_cd}  "
              f"| Still need → S:{need_s}  CP:{need_cp}  AN:{need_an}  CD:{need_cd}")

        prompt = BATCH_PROMPT.format(
            article_list  = article_label,
            n_simple      = req_s,
            n_comparative = req_cp,
            n_analytical  = req_an,
            n_conditional = req_cd,
            excerpts      = excerpts_block
        )

        raw = call_gemini(client, prompt)
        simple_qs, comp_qs, anal_qs, cond_qs = parse_response(raw)

        simple_qs = filter_new(simple_qs, seen)
        comp_qs   = filter_new(comp_qs,   seen)
        anal_qs   = filter_new(anal_qs,   seen)
        cond_qs   = filter_new(cond_qs,   seen)

        s, cp, an, cd = counts(dataset)
        simple_qs = simple_qs[:max(0, TARGET_SIMPLE      - s)]
        comp_qs   = comp_qs  [:max(0, TARGET_COMPARATIVE - cp)]
        anal_qs   = anal_qs  [:max(0, TARGET_ANALYTICAL  - an)]
        cond_qs   = cond_qs  [:max(0, TARGET_CONDITIONAL - cd)]

        for q in simple_qs:
            dataset.append({"question": q, "type": "simple",      "articles": all_articles_label})
        for q in comp_qs:
            dataset.append({"question": q, "type": "comparative", "articles": all_articles_label})
        for q in anal_qs:
            dataset.append({"question": q, "type": "analytical",  "articles": all_articles_label})
        for q in cond_qs:
            dataset.append({"question": q, "type": "conditional", "articles": all_articles_label})

        s, cp, an, cd = counts(dataset)
        added = len(simple_qs) + len(comp_qs) + len(anal_qs) + len(cond_qs)
        print(f"   ✔ Added {added}  |  S:{s}/{TARGET_SIMPLE}  CP:{cp}/{TARGET_COMPARATIVE}  "
              f"AN:{an}/{TARGET_ANALYTICAL}  CD:{cd}/{TARGET_CONDITIONAL}")

        done_batches.append(batch_idx)
        save_progress(done_batches, dataset)

        if s >= TARGET_SIMPLE and cp >= TARGET_COMPARATIVE \
                and an >= TARGET_ANALYTICAL and cd >= TARGET_CONDITIONAL:
            print(f"\n🎯 All targets reached!")
            break

        print(f"   ⏱  Waiting {DELAY_SECONDS}s...\n")
        time.sleep(DELAY_SECONDS)

    # Final trim & shuffle
    simples      = [d for d in dataset if d["type"] == "simple"]     [:TARGET_SIMPLE]
    comparatives = [d for d in dataset if d["type"] == "comparative"][:TARGET_COMPARATIVE]
    analyticals  = [d for d in dataset if d["type"] == "analytical"] [:TARGET_ANALYTICAL]
    conditionals = [d for d in dataset if d["type"] == "conditional"][:TARGET_CONDITIONAL]

    final = simples + comparatives + analyticals + conditionals
    random.shuffle(final)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*55}")
    print(f"🎉 Generation Complete!")
    print(f"   Simple      : {len(simples)}  / {TARGET_SIMPLE}")
    print(f"   Comparative : {len(comparatives)} / {TARGET_COMPARATIVE}")
    print(f"   Analytical  : {len(analyticals)} / {TARGET_ANALYTICAL}")
    print(f"   Conditional : {len(conditionals)} / {TARGET_CONDITIONAL}")
    print(f"   TOTAL       : {len(final)}")
    print(f"   Saved to    : {OUTPUT_JSON}")
    print(f"{'='*55}")

    print("\n── 2 Sample Simple ──")
    for q in random.sample(simples, min(2, len(simples))):
        print(f"  • {q['question']}")
    print("\n── 2 Sample Comparative ──")
    for q in random.sample(comparatives, min(2, len(comparatives))):
        print(f"  • {q['question']}")
    print("\n── 2 Sample Analytical ──")
    for q in random.sample(analyticals, min(2, len(analyticals))):
        print(f"  • {q['question']}")
    print("\n── 2 Sample Conditional ──")
    for q in random.sample(conditionals, min(2, len(conditionals))):
        print(f"  • {q['question']}")

if __name__ == "__main__":
    main()