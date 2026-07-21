def build_rag_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_text = "\n\n".join([
        f"[Section: {chunk['section']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    ])

    prompt = f"""You are a helpful assistant answering questions strictly based on the provided context about Islam, sourced from Wikipedia.

Context:
{context_text}

Question: {question}

Instructions:
- Answer only using the information in the context above.
- If the context doesn't contain enough information to answer, say so clearly instead of guessing.
- Mention which section(s) your answer is based on.

Answer:"""
    return prompt