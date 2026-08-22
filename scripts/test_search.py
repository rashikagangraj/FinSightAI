import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.indexer import _get_chroma_collection

collection = _get_chroma_collection()
print("Collection count:", collection.count())

# Try document query with where_document
res = collection.get(where_document={"$contains": "AIKART"}, include=["documents", "metadatas"])
print("Where_document AIKART results:", len(res["ids"]))
for doc, meta in zip(res["documents"], res["metadatas"]):
    print("Source:", meta.get("source"))
    print("Doc preview:", doc[:150])
    print("-" * 30)
