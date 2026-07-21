# Islam RAG Project

This repository provides a starter scaffold for building a retrieval-augmented generation (RAG) system over Islamic Wikipedia content.

## Structure

- data/raw: original scraped content
- data/processed: chunked sections and metadata
- data/eval: evaluation questions and answers
- src: Python modules for scraping, chunking, embeddings, indexing, retrieval, and prompting
- evaluation: scripts to run manual or automated evaluation
- app: a simple web UI for interacting with the RAG pipeline

## Next steps

1. Populate the raw source text in data/raw/islam_wikipedia.txt.
2. Implement the scraping and chunking logic in src.
3. Add embedding generation and vector search.
4. Build the prompt and evaluation flow.
