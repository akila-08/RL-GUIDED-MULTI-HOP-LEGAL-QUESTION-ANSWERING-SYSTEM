import re
from config import CHUNK_SIZE, CHUNK_OVERLAP

def split_into_articles(text):
    pattern = r'(Article\s+\d+.*?)((?=Article\s+\d+)|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    articles = []
    for match in matches:
        articles.append(match[0])
    return articles

def chunk_text(text):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = words[i:i + CHUNK_SIZE]
        chunks.append(" ".join(chunk))
    
    return chunks

def create_chunks(articles):
    all_chunks = []
    
    for art in articles:
        art_num = extract_article_number(art)
        chunks = chunk_text(art)
        
        for i, c in enumerate(chunks):
            all_chunks.append({
                "text": c,
                "article": art_num,
                "chunk_id": f"{art_num}_{i}"
            })
    
    return all_chunks

def extract_article_number(text):
    match = re.search(r'Article\s+(\d+)', text)
    return match.group(1) if match else "unknown"