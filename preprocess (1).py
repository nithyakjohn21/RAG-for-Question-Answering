"""Step 1: build the NQ-Open question subset."""
import json

from datasets import load_dataset

from config import DATA_DIR, NQ_FILE, NQ_NUM_QUESTIONS


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def build_subset(n: int = NQ_NUM_QUESTIONS):
    # First n questions of the NQ-Open train split (not a random sample).
    dataset = load_dataset("nq_open", split=f"train[:{n}]")

    subset = []
    for i, item in enumerate(dataset, start=1):
        answers = item.get("answer", [])
        if not answers:
            continue
        subset.append(
            {
                "id": i,
                "question": normalize(item["question"]),
                # Only the first gold answer is kept (see README: Limitations).
                "gold_answer": normalize(answers[0]),
            }
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(NQ_FILE, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(subset)} examples to {NQ_FILE}")


if __name__ == "__main__":
    build_subset()
