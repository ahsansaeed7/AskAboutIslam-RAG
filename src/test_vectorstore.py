import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from vectorstore import get_chroma_collection, add_chunks_to_collection

print("Getting collection...")
collection = get_chroma_collection()
print(f"Collection ready: {collection.name}")

test_chunks = [
    {"id": "test_1", "text": "The Five Pillars of Islam are Shahada, Salah, Zakat, Sawm, and Hajj.", "section": "Beliefs and practices"},
]

print("Adding chunks...")
add_chunks_to_collection(collection, test_chunks)
print("Done")




from vectorstore import get_chroma_collection

collection = get_chroma_collection()
print("Total documents:", collection.count())

# This is your "SELECT * LIMIT 5" equivalent
results = collection.get(limit=5, include=["documents", "metadatas", "embeddings"])
for i in range(len(results["ids"])):
    print("\nID:", results["ids"][i])
    print("Text:", results["documents"][i])
    print("Metadata:", results["metadatas"][i])
    print("Embedding (first 5 values):", results["embeddings"][i][:5], "... (384 numbers total)")