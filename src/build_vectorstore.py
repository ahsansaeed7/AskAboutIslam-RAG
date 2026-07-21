

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from vectorstore import get_chroma_collection, add_chunks_to_collection

# Build an absolute path to the project root, regardless of where this script is run from
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHUNKS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "islam_chunks.json")

def load_chunks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    print(f"Loading chunks from: {CHUNKS_PATH}")
    chunks = load_chunks(CHUNKS_PATH)

    print(f"Loaded {len(chunks)} chunks")

    collection = get_chroma_collection()

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        add_chunks_to_collection(collection, batch)
        print(f"Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    print(f"\nDone. Collection now has {collection.count()} documents.")