import json
from datasets import load_dataset


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def build_subset(output_path: str = "data/nq_subset.json", n: int = 200):
    dataset = load_dataset("nq_open", split=f"train[:{n}]")

    subset = []
    for i, item in enumerate(dataset, start=1):
        question = normalize(item["question"])

        answers = item.get("answer", [])
        if not answers:
            continue

        gold_answer = normalize(answers[0])

        subset.append(
            {
                "id": i,
                "question": question,
                "gold_answer": gold_answer,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(subset)} examples to {output_path}")


if __name__ == "__main__":
    build_subset()