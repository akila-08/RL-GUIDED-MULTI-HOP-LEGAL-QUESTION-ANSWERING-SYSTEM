from pdf_loader import load_pdf
from text_cleaner import clean_text
from chunker import split_into_articles, create_chunks
from metadata_generator import generate_keywords
from vector_store import build_faiss
from bm25_store import build_bm25
from config import PDF_PATH

def main():
    print("Loading PDF...")
    text = load_pdf(PDF_PATH)
    
    print("Cleaning text...")
    text = clean_text(text)
    
    print("Splitting into articles...")
    articles = split_into_articles(text)
    
    print("Chunking...")
    chunks = create_chunks(articles)
    
    print("Generating metadata...")
    docs = generate_keywords(chunks)
    
    print("Building FAISS...")
    build_faiss(docs)
    
    print("Building BM25...")
    build_bm25(docs)
    
    print("RAG setup complete!")

if __name__ == "__main__":
    main()