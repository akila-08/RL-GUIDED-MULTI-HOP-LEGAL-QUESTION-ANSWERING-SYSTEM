import json, time, random, re, os
from google import genai
# import google.generativeai as genai
from google.genai import types

# ── CONFIG ─────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCSMREwo9d406eg5bumcnKO5UCdOF2YGAI")

GEMINI_MODEL   = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-2.5-flash-lite"

INPUT_DATASET  = "C:\\Users\\aarth\\OneDrive\\Desktop\\RL+NLP PACK\\qa_dataset_chroma.json"
OUTPUT_JSON    = "C:\\Users\\aarth\\OneDrive\\Desktop\\RL+NLP PACK\\decompose_dataset.json"
PROGRESS_FILE  = "C:\\Users\\aarth\\OneDrive\\Desktop\\RL+NLP PACK\\decomp_progress.json"
COMPLEX_TYPES  = ["analytical", "conditional", "comparative"]
TARGET_DECOMP  = 700
DELAY_SECONDS  = 13
MAX_RETRIES    = 5
MIN_SUB_Q      = 2
MAX_SUB_Q      = 4

# FIX 1: Replaced <INSERT LEGAL QUESTION HERE> with {question} placeholder
PROMPT_TEMPLATE = """You are a legal reasoning assistant specialized in analytical decomposition of complex legal questions.

Your task is to break down a given legal question into a small set of clear, logically ordered sub-questions that collectively enable accurate answering of the original question.

### Instructions:
1. Carefully read the input legal question.
2. Identify the core legal issue and required supporting elements (rules, procedures, provisions).
3. Decompose the question into **2 to 5 sub-questions**.

### Strict Requirements:
- Sub-questions **1 to N-1 must be atomic**:
  - Focus on **only one concept**
  - Enable **single-hop retrieval** (answerable from one provision/source)
- Do NOT create sub-questions that:
  - Compare two concepts
  - Combine multiple legal ideas
  - Require conditional reasoning
  - Contain multiple clauses or embedded questions
- Ensure:
  - No redundancy or overlap
  - Minimal but complete coverage
  - Clear legal phrasing

### Synthesis Step (MANDATORY):
- The **final sub-question (last ID)** must:
  - Combine insights from previous sub-questions
  - Capture the **core legal reasoning required to answer the main question**
  - Be the **only multi-hop question**
- This final step should:
  - Use type = "application"
  - Depend on all relevant previous sub-questions

### Logical Structure:
- Order: **foundational → specific → synthesis**
- Keep dependencies minimal except for the final step

### Output Format (STRICT):
Return a JSON object with the following structure:

{{
  "main_question": "<original question>",
  "sub_questions": [
    {{
      "id": 1,
      "question": "<atomic sub-question>",
      "type": "<one of: fact | legal_issue | rule | application | exception>",
      "depends_on": []
    }},
    {{
      "id": 2,
      "question": "<atomic sub-question>",
      "type": "<...>",
      "depends_on": [1]
    }},
    {{
      "id": N,
      "question": "<synthesis sub-question combining prior answers>",
      "type": "application",
      "depends_on": [<relevant previous ids>]
    }}
  ]
}}

### Notes:
- "type" definitions:
  - fact → factual clarification
  - legal_issue → identification of the issue
  - rule → legal provision or requirement
  - application → applying or combining rules
  - exception → narrow exception
- Prefer **rule-type** for procedural/legal queries
- Only the **final sub-question may be multi-hop**
- Earlier sub-questions must remain strictly **single-hop**

### Input:
{question}"""

# ── SETUP ──────────────────────────────────
client        = genai.Client(api_key=GEMINI_API_KEY)
active_model  = GEMINI_MODEL
rpd_exhausted = False

print(f"✅ Model        : {active_model}")
print(f"📊 Rate limits  : 5 RPM, 20 RPD (free tier)")
print(f"⏱  Delay        : {DELAY_SECONDS}s between calls")

with open(INPUT_DATASET) as f:
    data = json.load(f)

complex_qs = [d for d in data if d.get("type") in COMPLEX_TYPES]
random.seed(42)
random.shuffle(complex_qs)
print(f"📂 Complex      : {len(complex_qs)} questions")
print(f"⏱  Est. time    : ~{TARGET_DECOMP * DELAY_SECONDS // 60} min\n")

if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        prog = json.load(f)
    dataset  = prog["dataset"]
    done_ids = set(prog["done_ids"])
    print(f"↩️  Resuming: {len(dataset)} done")
else:
    dataset  = []
    done_ids = set()


def save():
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"done_ids": list(done_ids), "dataset": dataset}, f)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


def sanitize(value):
    if isinstance(value, list):
        value = [str(v).strip() for v in value if v and str(v).strip()]
        return ", ".join(value) if value else "Not specified"
    return str(value).strip() if value and str(value).strip() else "Not specified"


# FIX 2: parse() now correctly handles list of dicts returned by the structured prompt
def parse(raw):
    if not raw:
        return []
    raw = (raw
           .replace('\u201c', '"').replace('\u201d', '"')
           .replace('\u2018', "'").replace('\u2019', "'"))
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw.strip()).strip()

    # Primary: parse structured JSON with sub_questions as list of dicts
    try:
        d  = json.loads(raw)
        qs = d.get("sub_questions", [])

        # FIX 2a: Extract dicts (structured output from prompt)
        if qs and isinstance(qs[0], dict):
            valid = []
            for q in qs:
                question_text = q.get("question", "").strip()
                if "?" in question_text and len(question_text.split()) >= 4:
                    valid.append({
                        "id":         q.get("id"),
                        "question":   question_text,
                        "type":       q.get("type", "rule"),
                        "depends_on": q.get("depends_on", []),
                    })
            if len(valid) >= MIN_SUB_Q:
                return valid[:MAX_SUB_Q]

        # Fallback: plain strings in sub_questions list
        elif qs and isinstance(qs[0], str):
            valid = [{"id": i+1, "question": q.strip(), "type": "rule", "depends_on": []}
                     for i, q in enumerate(qs)
                     if "?" in q and len(q.split()) >= 4]
            if len(valid) >= MIN_SUB_Q:
                return valid[:MAX_SUB_Q]

    except Exception:
        pass

    # Regex fallback: extract quoted questions from malformed output
    matches = re.findall(r'"([^"]{10,300}\?)"', raw)
    valid   = [{"id": i+1, "question": m.strip(), "type": "rule", "depends_on": []}
               for i, m in enumerate(matches) if len(m.split()) >= 4]
    if len(valid) >= MIN_SUB_Q:
        return valid[:MAX_SUB_Q]

    # Line-by-line fallback
    valid = []
    for i, line in enumerate(raw.split('\n')):
        line = re.sub(r'^[-•*\d\.]+\s*', '', line.strip().strip('"').strip("'"))
        if line.endswith('?') and len(line.split()) >= 4 and len(line) > 15:
            valid.append({"id": i+1, "question": line, "type": "rule", "depends_on": []})
    if len(valid) >= MIN_SUB_Q:
        return valid[:MAX_SUB_Q]

    return []


def make_config(model):
    return types.GenerateContentConfig(
        temperature       = 0.2,
        max_output_tokens = 2048,
    )


def call_gemini(prompt):
    global active_model, rpd_exhausted
    wait = 65

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model    = active_model,
                contents = prompt,
                config   = make_config(active_model),
            )
            return response.text

        except Exception as e:
            err = str(e)

            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                if "day" in err.lower() or "daily" in err.lower() or "quota" in err.lower():
                    if active_model != FALLBACK_MODEL:
                        print(f"  🔄 Quota exhausted on {active_model} → switching to {FALLBACK_MODEL}")
                        active_model = FALLBACK_MODEL
                        time.sleep(5)
                        continue
                    else:
                        print(f"  ❌ Quota exhausted on all models. Stopping.")
                        rpd_exhausted = True
                        return None

                m = re.search(r"seconds[\"']?:\s*(\d{1,4})", err)
                w = min(int(m.group(1)) + 5, 120) if m else wait
                print(f"  ⏳ Rate limit ({active_model}) → waiting {w}s "
                      f"(attempt {attempt}/{MAX_RETRIES})")
                time.sleep(w)
                wait = min(wait * 2, 300)

                if attempt == 3 and active_model != FALLBACK_MODEL:
                    print(f"  🔄 Switching to fallback: {FALLBACK_MODEL}")
                    active_model = FALLBACK_MODEL

            elif "not found" in err.lower() or "deprecated" in err.lower():
                print(f"  ❌ Model not found: {err[:150]}")
                print(f"     → Switching to fallback: {FALLBACK_MODEL}")
                active_model = FALLBACK_MODEL
                time.sleep(5)

            elif "billing" in err.lower() or "payment" in err.lower():
                print(f"  ❌ Billing error — enable at: https://aistudio.google.com/")
                rpd_exhausted = True
                return None

            else:
                print(f"  ⚠️  Error (attempt {attempt}): {err[:150]}")
                time.sleep(10)

    return None


# ── MAIN LOOP ──────────────────────────────
print("🚀 Starting...\n")
record_id  = len(dataset)
failed_log = []

for item in complex_qs:
    if len(dataset) >= TARGET_DECOMP:
        break

    if rpd_exhausted:
        print("\n⛔ Quota exhausted. Progress saved — re-run to continue.")
        break

    q_id = item["question"][:60]
    if q_id in done_ids:
        continue

    question = item.get("question", "")
    q_type   = item.get("type", "")
    articles = item.get("articles", [])

    if not question.strip():
        done_ids.add(q_id)
        continue

    if record_id % 20 == 0:
        pct = record_id / TARGET_DECOMP * 100
        print(f"[{record_id}/{TARGET_DECOMP}] {pct:.0f}% | "
              f"{q_type}: {question[:50]}...")

    # FIX 1: Now correctly injects the question into the prompt
    prompt = PROMPT_TEMPLATE.format(question=sanitize(question))

    raw    = call_gemini(prompt)
    sub_qs = parse(raw)

    if len(sub_qs) >= MIN_SUB_Q:
        # FIX 3: Store full structured sub-questions (with id, type, depends_on)
        dataset.append({
            "id":                f"decomp_{record_id:04d}",
            "complex_question":  question,
            "question_type":     q_type,
            "sub_questions":     sub_qs,                          # now list of dicts
            "num_sub_questions": len(sub_qs),
            "source_articles":   articles,
        })
        done_ids.add(q_id)
        record_id += 1

        if record_id % 20 == 0:
            save()
            print(f"   💾 Saved {record_id} records")
    else:
        failed_log.append(question[:60])

    time.sleep(DELAY_SECONDS)

save()
print(f"\n{'='*50}")
print(f"🎉 Done!")
print(f"   Records  : {len(dataset)}")
print(f"   Failed   : {len(failed_log)}")
print(f"   File     : {OUTPUT_JSON}")
print(f"{'='*50}")

if failed_log:
    print(f"\nFailed questions (first 5):")
    for q in failed_log[:5]:
        print(f"  - {q}")