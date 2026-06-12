import json
import re
from pathlib import Path


DATA_DIR = Path("data")
RESULTS_FILE = DATA_DIR / "results_hf.json"


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s,.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def main():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    baseline_correct = 0
    retrieval_correct = 0

    rows = []

    for item in results:
        gold = item["gold_answer"]
        baseline = item["baseline_answer"]
        retrieval = item["retrieval_answer"]

        b_em = exact_match(baseline, gold)
        r_em = exact_match(retrieval, gold)

        baseline_correct += b_em
        retrieval_correct += r_em

        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "gold_answer": gold,
                "baseline_answer": baseline,
                "retrieval_answer": retrieval,
                "baseline_em": b_em,
                "retrieval_em": r_em,
            }
        )

    total = len(results)
    baseline_acc = baseline_correct / total if total else 0
    retrieval_acc = retrieval_correct / total if total else 0

    print("\n=== Evaluation Summary ===")
    print(f"Total questions:         {total}")
    print(f"Baseline correct:        {baseline_correct}")
    print(f"Retrieval correct:       {retrieval_correct}")
    print(f"Baseline EM accuracy:    {baseline_acc:.2%}")
    print(f"Retrieval EM accuracy:   {retrieval_acc:.2%}")

    print("\n=== Per-question Results ===")
    for row in rows:
        print(f"\nQ{row['id']}: {row['question']}")
        print(f"Gold:        {row['gold_answer']}")
        print(f"Baseline:    {row['baseline_answer']}   | EM={row['baseline_em']}")
        print(f"Retrieval:   {row['retrieval_answer']}   | EM={row['retrieval_em']}")


if __name__ == "__main__":
    main()
