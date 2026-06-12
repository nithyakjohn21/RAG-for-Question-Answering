import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi


DATA_DIR = Path("data")
PASSAGES_FILE = DATA_DIR / "wiki_passages_auto.jsonl"


STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "and",
    "or",
    "by",
    "with",
    "as",
    "from",
    "that",
    "this",
    "it",
    "who",
    "what",
    "when",
    "where",
    "which",
    "why",
    "how",
    "did",
    "does",
    "do",
    "has",
    "have",
    "had",
    "been",
    "into",
    "its",
    "their",
    "they",
    "them",
    "his",
    "her",
    "he",
    "she",
    "you",
    "your",
    "our",
    "us",
}


def preprocess_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def load_passages(path):
    passages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line))
    return passages


class BM25Retriever:
    def __init__(self, passages):
        self.passages = passages

        # Give extra weight to title terms by repeating the title
        self.search_docs = [
            f"{p['title']} {p['title']} {p['title']} {p['text']}" for p in passages
        ]

        self.tokenized_corpus = [preprocess_text(doc) for doc in self.search_docs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 3):
        tokenized_query = preprocess_text(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in ranked_indices:
            results.append(
                {
                    "score": float(scores[idx]),
                    "passage_id": self.passages[idx]["passage_id"],
                    "title": self.passages[idx]["title"],
                    "text": self.passages[idx]["text"],
                }
            )
        return results


def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    passages = load_passages(PASSAGES_FILE)
    retriever = BM25Retriever(passages)

    questions = load_questions(DATA_DIR / "nq_subset.json")

    # test first 5 questions from your real dataset
    for item in questions[:5]:
        test_query = item["question"]
        results = retriever.retrieve(test_query, top_k=3)

        print(f"\nQuery: {test_query}\n")
        for i, r in enumerate(results, start=1):
            print(f"Rank {i}")
            print(f"Title: {r['title']}")
            print(f"Score: {r['score']:.4f}")
            print(f"Text: {r['text'][:300]}...")
            print("-" * 80)


if __name__ == "__main__":
    main()
