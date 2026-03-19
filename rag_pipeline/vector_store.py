import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from config import EMBEDDING_MODEL, FAISS_INDEX_PATH, DOC_STORE_PATH

def build_faiss(docs):
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    with open(DOC_STORE_PATH, "wb") as f:
        pickle.dump(docs, f)
    
    print("FAISS index built!")

def load_faiss():
    index = faiss.read_index(FAISS_INDEX_PATH)
    
    with open(DOC_STORE_PATH, "rb") as f:
        docs = pickle.load(f)
    
    return index, docs