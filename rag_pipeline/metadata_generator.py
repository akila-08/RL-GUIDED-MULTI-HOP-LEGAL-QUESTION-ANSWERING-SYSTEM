from sklearn.feature_extraction.text import TfidfVectorizer

def generate_keywords(docs, top_k=5):
    texts = [d["text"] for d in docs]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    feature_names = vectorizer.get_feature_names_out()
    
    for i, doc in enumerate(docs):
        tfidf_scores = X[i].toarray()[0]
        top_indices = tfidf_scores.argsort()[-top_k:]
        
        keywords = [feature_names[j] for j in top_indices]
        
        doc["keywords"] = keywords
        doc["source"] = "Constitution of India"
        doc["type"] = "constitutional_law"
    
    return docs