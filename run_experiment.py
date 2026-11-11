"""Step 3: run closed-book baseline and BM25-RAG on every question."""
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from bm25_retriever import BM25Retriever, load_passages
from config import (
    HF_BASE_URL,
    MAX_TOKENS,
    MODEL_NAME,
    NQ_FILE,
    RESULTS_DIR,
    RESULTS_FILE,
    TEMPERATURE,
    TOP_K,
)
from prompts import build_baseline_prompt, build_retrieval_prompt


def get_client() -> OpenAI:
    load_dotenv()  # reads HF_TOKEN from a local .env file if present
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is not set. Copy .env.example to .env and add your token.")
    return OpenAI(base_url=HF_BASE_URL, api_key=token)


def ask(client: OpenAI, prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:  # network / rate-limit errors
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  API error ({e}); retrying in {wait}s")
            time.sleep(wait)


def main():
    with open(NQ_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    retriever = BM25Retriever(load_passages())
    client = get_client()
    results = []

    for idx, item in enumerate(questions, start=1):
        question = item["question"]
        print(f"\nQ{idx}/{len(questions)}: {question}")

        baseline_answer = ask(client, build_baseline_prompt(question))

        retrieved = retriever.retrieve(question, top_k=TOP_K)
        retrieval_answer = ask(client, build_retrieval_prompt(question, retrieved))

        results.append(
            {
                "id": item["id"],
                "question": question,
                "gold_answer": item["gold_answer"],
                "baseline_answer": baseline_answer,
                "retrieval_answer": retrieval_answer,
                "retrieved_passages": retrieved,
            }
        )
        print(f"  Gold:      {item['gold_answer']}")
        print(f"  Baseline:  {baseline_answer}")
        print(f"  RAG:       {retrieval_answer}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
