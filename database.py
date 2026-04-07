import chromadb

# ── Connect to ChromaDB ──
client = chromadb.PersistentClient(path=r"C:\Users\sujan\RL-GUIDED-MULTI-HOP-LEGAL-QUESTION-ANSWERING-SYSTEM\db\constitution_db")

# ── List all collections ──
collections = client.list_collections()
print(f"Collections found: {[c.name for c in collections]}\n")

collection = client.get_collection(collections[0].name)

# ── Total count ──
print(f"Total chunks in DB: {collection.count()}\n")

# ── Fetch first 5 chunks ──
results = collection.get(
    limit=5,
    include=["documents", "metadatas", "embeddings"]
)

for i in range(len(results["ids"])):
    print("=" * 60)
    print(f"  CHUNK #{i+1}")
    print("=" * 60)
    print(f"  ID         : {results['ids'][i]}")
    print(f"  Metadata   : {results['metadatas'][i]}")
    print(f"  Embedding  : {results['embeddings'][i][:5]} ... (vector, showing first 5 values)")
    print(f"  Text       :\n\n{results['documents'][i]}\n")