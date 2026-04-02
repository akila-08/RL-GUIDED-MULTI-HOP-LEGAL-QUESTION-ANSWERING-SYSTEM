from ingestion.embedder import get_collection

col = get_collection()
res = col.get(limit=5, include=["metadatas"])

for m in res["metadatas"]:
    print(m)