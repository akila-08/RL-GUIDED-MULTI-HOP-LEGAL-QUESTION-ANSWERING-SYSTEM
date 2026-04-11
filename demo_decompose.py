import sys, os
sys.path.append('d:/Projects/RL+NLP/RL-GUIDED-MULTI-HOP-LEGAL-QUESTION-ANSWERING-SYSTEM')
from pipeline.baseline_rules import dataset_style_decompose_v3
questions = [
    "Compare the roles of Municipalities and Panchayats in local governance.",
    "How does the Right to Equality differ from the Prohibition of Discrimination?"
]
for q in questions:
    subs = dataset_style_decompose_v3(q)
    print('Question:', q)
    print('Sub-questions:')
    for s in subs:
        print(' -', s)
    print('---')
