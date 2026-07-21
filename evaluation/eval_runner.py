import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import csv
from datetime import datetime
from src.rag_pipeline import answer_question


def load_test_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(test_questions_path: str, output_csv_path: str):
    questions = load_test_questions(test_questions_path)
    results = []

    for i, item in enumerate(questions):
        question = item["question"]
        expected_section = item.get("expected_section")
        q_type = item.get("type", "unspecified")

        print(f"[{i+1}/{len(questions)}] Running: {question}")

        try:
            result = answer_question(question)
            answer = result["answer"]
            sources = ", ".join(result["sources"])
        except Exception as e:
            answer = f"ERROR: {str(e)}"
            sources = ""

        results.append({
            "question": question,
            "type": q_type,
            "expected_section": expected_section or "",
            "answer": answer,
            "retrieved_sources": sources,
            "manual_correct": "",       # you fill this in manually: yes/no
            "manual_grounded": "",      # you fill this in manually: yes/no
            "notes": ""                 # optional manual notes
        })

    # Save to CSV for manual review/scoring
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Results saved to {output_csv_path}")
    print(f"Open the CSV and fill in 'manual_correct' and 'manual_grounded' columns (yes/no) for each row.")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    test_questions_path = os.path.join(PROJECT_ROOT, "data", "eval", "test_questions.json")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_path = os.path.join(PROJECT_ROOT, "evaluation", "results", f"eval_results_{timestamp}.csv")

    run_evaluation(test_questions_path, output_csv_path)