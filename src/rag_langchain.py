import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.dirname(__file__)
sys.path.append(PROJECT_ROOT)
sys.path.append(SRC_DIR)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
from config import LLM_PROVIDER, OLLAMA_MODEL, GROQ_API_KEY, GROQ_MODEL

if LLM_PROVIDER == "groq":
    from langchain_groq import ChatGroq
else:
    from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, OLLAMA_MODEL


# --- Step 1: Connect to your EXISTING Chroma DB (same data, no re-embedding needed) ---
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    collection_name=CHROMA_COLLECTION_NAME,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# --- Step 2: Prompt template ---
prompt = ChatPromptTemplate.from_template("""You are a helpful assistant answering questions strictly based on the provided context about Islam, sourced from Wikipedia.

Context:
{context}

Question: {question}

Instructions:
- Answer only using the information in the context above.
- If the context doesn't contain enough information to answer, say so clearly instead of guessing.
- Mention which section(s) your answer is based on.

Answer:""")


# --- Step 3: LLM ---
if LLM_PROVIDER == "groq":
    llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)
else:
    llm = OllamaLLM(model=OLLAMA_MODEL)


# --- Step 4: Helper to format retrieved docs with section metadata ---
def format_docs(docs):
    return "\n\n".join([
        f"[Section: {doc.metadata.get('section', 'Unknown')}]\n{doc.page_content}"
        for doc in docs
    ])


# --- Step 5: Chain everything together with LCEL ---
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def answer_question(question: str) -> dict:
    """
    Runs the LangChain RAG pipeline and returns the answer along with sources.
    """
    retrieved_docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)

    return {
        "question": question,
        "answer": answer,
        "sources": [doc.metadata.get("section", "Unknown") for doc in retrieved_docs]
    }


if __name__ == "__main__":
    test_questions = [
        "What are the Five Pillars of Islam?",
        "What is the difference between Sunni and Shia Islam?",
        "What is the population of penguins in Antarctica?"
    ]

    for q in test_questions:
        print("=" * 80)
        result = answer_question(q)
        print(f"Q: {result['question']}")
        print(f"\nA: {result['answer']}")
        print(f"\nSources: {result['sources']}")
        print()