import json
import os
from pathlib import Path

from openai import OpenAI

from bm25_retriever import load_passages, BM25Retriever
from prompts import build_baseline_prompt, build_retrieval_prompt


DATA_DIR = Path("data")
NQ_FILE = DATA_DIR / "nq_subset.json"
PASSAGES_FILE = DATA_DIR / "wiki_passages_auto.jsonl"
RESULTS_FILE = DATA_DIR / "results_hf.json"

# Pick a model that is available through HF Inference Providers.
# You can change this later if needed.
MODEL_NAME = "katanemo/Arch-Router-1.5B:hf-inference"


def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ask_hf(prompt: str) -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is not set in the environment.")

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=64,
    )

    return completion.choices[0].message.content.strip()


def main():
    questions = load_questions(NQ_FILE)
    passages = load_passages(PASSAGES_FILE)
    retriever = BM25Retriever(passages)

    results = []

    for idx, item in enumerate(questions, start=1):
        qid = item["id"]
        question = item["question"]
        gold = item["gold_answer"]

        print(f"\nProcessing Q{idx}/{len(questions)}: {question}")

        baseline_prompt = build_baseline_prompt(question)
        baseline_answer = ask_hf(baseline_prompt)

        retrieved = retriever.retrieve(question, top_k=5)
        retrieval_prompt = build_retrieval_prompt(question, retrieved)
        retrieval_answer = ask_hf(retrieval_prompt)

        results.append(
            {
                "id": qid,
                "question": question,
                "gold_answer": gold,
                "baseline_answer": baseline_answer,
                "retrieval_answer": retrieval_answer,
                "retrieved_passages": retrieved,
            }
        )

        print(f"Gold:      {gold}")
        print(f"Baseline:  {baseline_answer}")
        print(f"Retrieved: {retrieval_answer}")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
