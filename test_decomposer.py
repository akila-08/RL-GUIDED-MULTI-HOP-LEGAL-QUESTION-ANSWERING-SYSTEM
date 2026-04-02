import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("./decomp_model").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("./decomp_model")

def generate_subquestions(question, verbose=False):
    input_text = "decompose: " + question

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=256,
        truncation=True
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,        # 🔧 use max_new_tokens, not max_length
            num_beams=4,
            no_repeat_ngram_size=0,    # 🔧 disable — breaks <rule>/<apply> tag repetition
            repetition_penalty=1.0,    # 🔧 disable — interferes with structured output
            early_stopping=True,
            forced_eos_token_id=tokenizer.eos_token_id
        )

    # 🔧 Keep special tokens so structure is visible, then parse
    raw = tokenizer.decode(outputs[0], skip_special_tokens=False)
    raw = raw.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "").strip()

    if verbose:
        print("Raw output:", repr(raw))

    return parse_subquestions(raw)


def parse_subquestions(raw_output):
    """Parse <rule> and <apply> tags into a structured result."""
    lines = raw_output.strip().split("\n")
    result = []

    for line in lines:
        line = line.strip()
        if line.startswith("<rule>"):
            q = line.replace("<rule>", "").strip()
            result.append({"type": "rule", "question": q})
        elif line.startswith("<apply>"):
            q = line.replace("<apply>", "").strip()
            result.append({"type": "apply", "question": q})
        elif line:
            # Fallback: untagged line, treat as rule
            result.append({"type": "rule", "question": line})

    return result


# =========================
# TEST
# =========================
question = "How does Article 183 differ from Article 184?"
subquestions = generate_subquestions(question, verbose=True)

print(f"\nQuestion: {question}")
print(f"Decomposed into {len(subquestions)} sub-questions:\n")
for i, sq in enumerate(subquestions, 1):
    tag = f"[{sq['type'].upper()}]"
    print(f"  {i}. {tag} {sq['question']}")


