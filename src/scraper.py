import wikipediaapi
import json
import os

def scrape_wikipedia_article(title: str, output_dir_raw: str, output_dir_processed: str):
    wiki = wikipediaapi.Wikipedia(
        user_agent='IslamQnAProject/1.0 (student-learning-project)',
        language='en'
    )

    page = wiki.page(title)

    if not page.exists():
        print(f"Page '{title}' not found.")
        return

    # Save raw flat text
    os.makedirs(output_dir_raw, exist_ok=True)
    raw_path = os.path.join(output_dir_raw, f"{title.lower()}_wikipedia.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(page.text)
    print(f"Saved raw text to {raw_path} ({len(page.text)} characters)")

    # Save section-structured version (for chunking with metadata later)
    def extract_sections(sections, parent_title=""):
        result = []
        for s in sections:
            full_title = f"{parent_title} > {s.title}" if parent_title else s.title
            if s.text.strip():
                result.append({"section": full_title, "text": s.text})
            result.extend(extract_sections(s.sections, full_title))
        return result

    section_data = extract_sections(page.sections)

    os.makedirs(output_dir_processed, exist_ok=True)
    processed_path = os.path.join(output_dir_processed, f"{title.lower()}_sections.json")
    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump(section_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(section_data)} sections to {processed_path}")

    return section_data


if __name__ == "__main__":
    scrape_wikipedia_article(
        title="Islam",
        output_dir_raw="data/raw",
        output_dir_processed="data/processed"
    )