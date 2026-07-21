import chromadb
from chromadb.utils import embedding_functions
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME

def get_chroma_collection():
    """
    Initializes a persistent Chroma client and returns the collection.
    Chroma will auto-generate embeddings locally using sentence-transformers.
    """
    # This creates/loads a local folder-based database — no server, no signup
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Free, local embedding model — no API key, no cost, no rate limits
    local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=local_ef
    )

    return collection


def add_chunks_to_collection(collection, chunks: list[dict]):
    """
    Adds chunked documents to the Chroma collection.
    chunks: list of dicts like {"id": "chunk_1", "text": "...", "section": "History"}
    """
    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [{"section": chunk["section"]} for chunk in chunks]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    print(f"Added {len(chunks)} chunks to Chroma collection '{collection.name}'")


if __name__ == "__main__":
    # Quick test
    collection = get_chroma_collection()
    print(f"Collection '{collection.name}' currently has {collection.count()} documents")