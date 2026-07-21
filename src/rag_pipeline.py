import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.dirname(__file__)
sys.path.append(PROJECT_ROOT)
sys.path.append(SRC_DIR)

import requests
from retriever import retrieve_relevant_chunks
from prompt_templates import build_rag_prompt
from config import OLLAMA_BASE_URL, OLLAMA_MODEL


from llm_provider import call_llm

def answer_question(question: str, n_results: int = 3) -> dict:
    """
    Full RAG pipeline: retrieve relevant chunks, build prompt, generate answer.
    Returns the answer along with the sources used.
    """
    # Step 1: Retrieve relevant chunks
    retrieved_chunks = retrieve_relevant_chunks(question, n_results=n_results)

    # Step 2: Build the prompt
    prompt = build_rag_prompt(question, retrieved_chunks)

    # Step 3: Generate answer
    answer = call_llm(prompt)

    return {
        "question": question,
        "answer": answer,
        "sources": [chunk["section"] for chunk in retrieved_chunks]
    }


if __name__ == "__main__":
    test_questions = [
        "What are the Five Pillars of Islam?",
        "What is the difference between Sunni and Shia Islam?",
        "What is the population of penguins in Antarctica?"  # out-of-scope test
    ]

    for q in test_questions:
        print("=" * 80)
        result = answer_question(q)
        print(f"Q: {result['question']}")
        print(f"\nA: {result['answer']}")
        print(f"\nSources: {result['sources']}")
        print()