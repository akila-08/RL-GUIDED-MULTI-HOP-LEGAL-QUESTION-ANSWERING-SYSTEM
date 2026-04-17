import sys
sys.path.insert(0, 'd:/Projects/RL+NLP/RL-GUIDED-MULTI-HOP-LEGAL-QUESTION-ANSWERING-SYSTEM')
from pipeline.complexity import explain_complexity

questions = [
    "Compare the roles of Municipalities and Panchayats in local governance.",
    "How does the Right to Equality differ from the Prohibition of Discrimination?",
    "If two states have a legal dispute, explain how the original jurisdiction of the Supreme Court applies and why it is exclusive.",
    "What is Article 21?",
    "What does the President do during a national emergency?",
]

print(f"{'Question':<65} {'Type':<13} {'Raw':>5} {'Floor':>5} {'Bonus':>5} {'Score':>5} {'Verdict'}")
print("-" * 115)
for q in questions:
    r = explain_complexity(q)
    print(f"{q[:64]:<65} {r['question_type']:<13} {r['classifier_raw']:>5.2f} {r['type_floor']:>5.2f} {r['lexical_bonus']:>5.3f} {r['final_score']:>5.3f} {r['verdict']}")
