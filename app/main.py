import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.append(PROJECT_ROOT)
sys.path.append(SRC_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional
import time

from rag_pipeline import answer_question as answer_raw
from rag_langchain import answer_question as answer_langchain
from rag_langgraph import answer_question as answer_langgraph


app = FastAPI(title="Islam QnA RAG API", version="1.0")

# Allow your frontend (running on a different port/domain) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend URL before production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    pipeline: Literal["raw", "langchain", "langgraph"] = "raw"


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    pipeline_used: str
    retries_used: Optional[int] = None
    is_grounded: Optional[bool] = None
    response_time_seconds: float


@app.get("/")
def root():
    return {
        "message": "Islam QnA RAG API is running",
        "available_pipelines": ["raw", "langchain", "langgraph"],
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start_time = time.time()

    try:
        if request.pipeline == "raw":
            result = answer_raw(request.question)
            retries_used = None
            is_grounded = None

        elif request.pipeline == "langchain":
            result = answer_langchain(request.question)
            retries_used = None
            is_grounded = None

        elif request.pipeline == "langgraph":
            result = answer_langgraph(request.question)
            retries_used = result.get("retries_used")
            is_grounded = result.get("is_grounded")

        else:
            raise HTTPException(status_code=400, detail=f"Unknown pipeline: {request.pipeline}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running pipeline: {str(e)}")

    elapsed = time.time() - start_time

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        pipeline_used=request.pipeline,
        retries_used=retries_used,
        is_grounded=is_grounded,
        response_time_seconds=round(elapsed, 2)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)