"""Step 4: score results with EM, token-level F1 and retrieval diagnostics.

Reproduces Tables 1 and 2 of the report.
Usage:  python evaluate.py            # summary tables
        python evaluate.py --verbose  # plus per-question output
"""
import argparse
import json
import re
from collections import Counter, defaultdict

from config import RESULTS_FILE

QUESTION_TYPES = ["who", "what", "when", "which", "how", "where"]


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s,.-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def gold_in_context(item: dict) -> bool:
    gold = normalize_answer(item["gold_answer"])
    return any(gold in normalize_answer(p["text"]) for p in item["retrieved_passages"])


def question_type(question: str) -> str:
    # prefix match, so "who's" -> who and "what's" -> what
    return next((t for t in QUESTION_TYPES if question.startswith(t)), "other")


def main(verbose: bool = False):
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    n = len(results)
    rows = []
    for item in results:
        gold = item["gold_answer"]
        rows.append(
            {
                **item,
                "b_em": exact_match(item["baseline_answer"], gold),
                "r_em": exact_match(item["retrieval_answer"], gold),
                "b_f1": token_f1(item["baseline_answer"], gold),
                "r_f1": token_f1(item["retrieval_answer"], gold),
                "covered": gold_in_context(item),
                "type": question_type(item["question"]),
            }
        )

    def avg(key, subset=rows):
        return sum(r[key] for r in subset) / len(subset) if subset else 0.0

    b_em, r_em = sum(r["b_em"] for r in rows), sum(r["r_em"] for r in rows)
    improved = sum(r["r_f1"] > r["b_f1"] for r in rows)
    worsened = sum(r["r_f1"] < r["b_f1"] for r in rows)
    abstained = sum("could not find the answer" in r["retrieval_answer"].lower() for r in rows)

    print("\n=== Table 1: Overall results ===")
    print(f"{'Method':<22}{'Correct':>10}{'EM':>9}{'F1':>8}")
    print(f"{'Baseline (closed-book)':<22}{b_em:>6}/{n:<3}{b_em / n:>9.1%}{avg('b_f1'):>8.3f}")
    print(f"{'RAG (BM25)':<22}{r_em:>6}/{n:<3}{r_em / n:>9.1%}{avg('r_f1'):>8.3f}")

    print("\n=== Retrieval diagnostics ===")
    print(f"Gold answer in top-k context:   {avg('covered'):.1%}")
    print(f"Improved / worsened / same (F1): {improved} / {worsened} / {n - improved - worsened}")
    print(f"RAG abstentions:                {abstained}")

    print("\n=== Table 2: By question type ===")
    print(f"{'Type':<8}{'n':>4}{'Base EM':>9}{'RAG EM':>8}{'Base F1':>9}{'RAG F1':>8}")
    groups = defaultdict(list)
    for r in rows:
        groups[r["type"]].append(r)
    for t in QUESTION_TYPES + ["other"]:
        g = groups.get(t, [])
        if g:
            print(f"{t:<8}{len(g):>4}{avg('b_em', g):>9.1%}{avg('r_em', g):>8.1%}"
                  f"{avg('b_f1', g):>9.3f}{avg('r_f1', g):>8.3f}")

    if verbose:
        print("\n=== Per-question results ===")
        for r in rows:
            print(f"\nQ{r['id']}: {r['question']}")
            print(f"  Gold:     {r['gold_answer']}")
            print(f"  Baseline: {r['baseline_answer']}  | EM={r['b_em']} F1={r['b_f1']:.2f}")
            print(f"  RAG:      {r['retrieval_answer']}  | EM={r['r_em']} F1={r['r_f1']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="print per-question results")
    main(parser.parse_args().verbose)
