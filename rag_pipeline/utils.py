import numpy as np

def hybrid_search(query, bm25, docs, faiss_index, embed_model, top_k=5):
    # BM25
    bm25_scores = bm25.get_scores(query.split())
    
    # Dense
    query_vec = embed_model.encode([query])
    D, I = faiss_index.search(query_vec, len(docs))
    
    dense_scores = np.zeros(len(docs))
    for rank, idx in enumerate(I[0]):
        dense_scores[idx] = 1 / (rank + 1)
    
    # Combine
    final_scores = 0.5 * bm25_scores + 0.5 * dense_scores
    
    top_indices = final_scores.argsort()[-top_k:][::-1]
    
    return [docs[i] for i in top_indices]