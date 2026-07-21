import json
import tiktoken
import os

# Using OpenAI's tokenizer for accurate token counting
# (still useful even though we're using local embeddings — it's just for measuring text length consistently)
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits text into chunks of roughly `chunk_size` tokens with `overlap` tokens
    shared between consecutive chunks, to avoid losing context at boundaries.
    """
    tokens = encoding.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text_str = encoding.decode(chunk_tokens)
        chunks.append(chunk_text_str)
        start += chunk_size - overlap  # move forward, but overlap with previous chunk

    return chunks


def process_sections_into_chunks(sections_json_path: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Reads the section-structured JSON (from scraper.py) and splits each section's
    text into smaller chunks, attaching section metadata to every chunk.
    """
    with open(sections_json_path, "r", encoding="utf-8") as f:
        sections = json.load(f)

    all_chunks = []
    chunk_counter = 0

    for section in sections:
        section_title = section["section"]
        section_text = section["text"]

        token_count = count_tokens(section_text)

        if token_count <= chunk_size:
            # Short section — keep as a single chunk
            all_chunks.append({
                "id": f"chunk_{chunk_counter}",
                "text": section_text,
                "section": section_title
            })
            chunk_counter += 1
        else:
            # Long section — split into multiple overlapping chunks
            sub_chunks = chunk_text(section_text, chunk_size=chunk_size, overlap=overlap)
            for sub_chunk in sub_chunks:
                all_chunks.append({
                    "id": f"chunk_{chunk_counter}",
                    "text": sub_chunk,
                    "section": section_title
                })
                chunk_counter += 1

    return all_chunks


if __name__ == "__main__":
    input_path = "data/processed/islam_sections.json"
    output_path = "data/processed/islam_chunks.json"

    chunks = process_sections_into_chunks(input_path, chunk_size=500, overlap=50)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Created {len(chunks)} chunks from {input_path}")
    print(f"Saved to {output_path}")

    # Quick sanity check — print token count distribution
    token_counts = [count_tokens(c["text"]) for c in chunks]
    print(f"Min tokens: {min(token_counts)}, Max tokens: {max(token_counts)}, Avg: {sum(token_counts)//len(token_counts)}")