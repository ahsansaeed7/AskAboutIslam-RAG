import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.dirname(__file__)
sys.path.append(PROJECT_ROOT)
sys.path.append(SRC_DIR)

from vectorstore import get_chroma_collection

def retrieve_relevant_chunks(question: str, n_results: int = 3) -> list[dict]:
    """
    Given a question, returns the top-n most relevant chunks with their metadata.
    """
    collection = get_chroma_collection()
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "section": results["metadatas"][0][i]["section"],
            "distance": results["distances"][0][i]  # lower = more similar
        })
    return chunks


if __name__ == "__main__":
    # Quick test
    question = "When did Islam start?"
    results = retrieve_relevant_chunks(question)
    for r in results:
        print(f"\nSection: {r['section']}")
        print(f"Distance: {r['distance']:.4f}")
        print(f"Text: {r['text'][:200]}...")