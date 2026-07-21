import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.dirname(__file__)
sys.path.append(PROJECT_ROOT)
sys.path.append(SRC_DIR)

from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from retriever import retrieve_relevant_chunks
from prompt_templates import build_rag_prompt
from llm_provider import call_llm


# --- Define the shared state that flows through the graph ---
class RAGState(TypedDict):
    question: str
    current_query: str
    retrieved_chunks: List[dict]
    answer: str
    verdict: str
    is_grounded: bool
    retry_count: int
    max_retries: int

# --- Node 1: Retrieve chunks ---
def retrieve_node(state: RAGState) -> RAGState:
    print(f"\n[retrieve] Query: {state['current_query']}")
    chunks = retrieve_relevant_chunks(state["current_query"], n_results=3)
    state["retrieved_chunks"] = chunks
    return state


# --- Node 2: Generate an answer ---
def generate_node(state: RAGState) -> RAGState:
    print("[generate] Drafting answer...")
    prompt = build_rag_prompt(state["question"], state["retrieved_chunks"])
    answer = call_llm(prompt)
    state["answer"] = answer
    return state

# --- Rule-based check for common refusal phrases (fast, deterministic, no LLM call needed) ---
REFUSAL_PHRASES = [
    "cannot find", "can't find", "does not mention", "doesn't mention",
    "no mention of", "not mentioned in", "does not provide", "doesn't provide",
    "cannot be answered", "can't be answered", "no information about",
    "context does not contain", "context doesn't contain", "unable to provide",
    "cannot determine", "can't determine", "not specify", "no specific",
    "there is no mention"
]

def is_refusal_answer(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in REFUSAL_PHRASES)


# --- Node 3: Check if the answer is actually grounded in the retrieved context ---
def check_grounding_node(state: RAGState) -> RAGState:
    print("[check_grounding] Verifying answer is supported by context...")

    # Fast path: rule-based refusal detection, skips the LLM judge entirely
    if is_refusal_answer(state["answer"]):
        print("[check_grounding] Rule-based check: detected refusal phrase → INSUFFICIENT")
        state["verdict"] = "INSUFFICIENT"
        state["is_grounded"] = False
        return state

    # Otherwise, fall back to LLM-as-judge for real grounding verification
    context_text = "\n\n".join([c["text"] for c in state["retrieved_chunks"]])

    grounding_prompt = f"""You are a strict fact-checker. Given the CONTEXT and the ANSWER below, classify the ANSWER into exactly one category.

CONTEXT:
{context_text}

ANSWER:
{state['answer']}

Categories:
- GROUNDED: the answer provides real, useful information that is fully supported by the context.
- NOT_GROUNDED: the answer contains claims or facts NOT found in the context (hallucination).

Respond with only one word: GROUNDED or NOT_GROUNDED.

Response:"""

    verdict = call_llm(grounding_prompt).strip().upper()
    print(f"[check_grounding] LLM verdict: {verdict}")

    state["verdict"] = verdict
    state["is_grounded"] = "GROUNDED" in verdict and "NOT_GROUNDED" not in verdict
    return state

# --- Node 4: Reformulate the query for a retry ---
def reformulate_node(state: RAGState) -> RAGState:
    state["retry_count"] += 1
    print(f"[reformulate] Retry #{state['retry_count']} — rewriting query...")

    reformulate_prompt = f"""The following question did not retrieve enough relevant context 
to answer well: "{state['question']}"

Rewrite this question as a different search query, using different keywords or phrasing, 
to try to retrieve more relevant information. Respond with ONLY the rewritten query, nothing else.

Rewritten query:"""

    new_query = call_llm(reformulate_prompt).strip()
    print(f"[reformulate] New query: {new_query}")
    state["current_query"] = new_query
    return state

# --- Conditional edge: decide whether to retry or finish ---
def should_retry(state: RAGState) -> str:
    if state["is_grounded"]:
        return "finish"
    if state["retry_count"] >= state["max_retries"]:
        print("[should_retry] Max retries reached — finishing anyway.")
        return "finish"
    return "retry"


# --- Build the graph ---
def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("check_grounding", check_grounding_node)
    graph.add_node("reformulate", reformulate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "check_grounding")

    graph.add_conditional_edges(
        "check_grounding",
        should_retry,
        {
            "finish": END,
            "retry": "reformulate"
        }
    )
    graph.add_edge("reformulate", "retrieve")

    return graph.compile()


app = build_graph()


def answer_question(question: str, max_retries: int = 2) -> dict:
    initial_state: RAGState = {
        "question": question,
        "current_query": question,
        "retrieved_chunks": [],
        "answer": "",
        "verdict": "",
        "is_grounded": False,
        "retry_count": 0,
        "max_retries": max_retries
    }
    

    final_state = app.invoke(initial_state)

    return {
        "question": question,
        "answer": final_state["answer"],
        "sources": [c["section"] for c in final_state["retrieved_chunks"]],
        "retries_used": final_state["retry_count"],
        "is_grounded": final_state["is_grounded"]
    }


if __name__ == "__main__":
    test_questions = [
        "What are the Five Pillars of Islam?",
        "When did Islam start?",
        "What is the population of penguins in Antarctica?"
    ]

    for q in test_questions:
        print("=" * 80)
        result = answer_question(q)
        print(f"\nQ: {result['question']}")
        print(f"\nA: {result['answer']}")
        print(f"\nSources: {result['sources']}")
        print(f"Retries used: {result['retries_used']}")
        print(f"Final grounding status: {result['is_grounded']}")
        print()