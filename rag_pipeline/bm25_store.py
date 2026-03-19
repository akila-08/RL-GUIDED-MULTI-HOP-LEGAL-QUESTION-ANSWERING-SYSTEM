from rank_bm25 import BM25Okapi
import pickle
from config import BM25_PATH

def build_bm25(docs):
    tokenized = [d["text"].split() for d in docs]
    
    bm25 = BM25Okapi(tokenized)
    
    with open(BM25_PATH, "wb") as f:
        pickle.dump((bm25, docs), f)
    
    print("BM25 built!")

def load_bm25():
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)