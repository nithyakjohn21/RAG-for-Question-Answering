"""BM25 passage retriever (rank-bm25) with title up-weighting."""
import json
import re

from rank_bm25 import BM25Okapi

from config import NQ_FILE, PASSAGES_FILE, TITLE_REPEAT, TOP_K

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at",
    "to", "for", "and", "or", "by", "with", "as", "from", "that", "this",
    "it", "who", "what", "when", "where", "which", "why", "how", "did",
    "does", "do", "has", "have", "had", "been", "into", "its", "their",
    "they", "them", "his", "her", "he", "she", "you", "your", "our", "us",
}


def preprocess_text(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 1]


def load_passages(path=PASSAGES_FILE) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class BM25Retriever:
    def __init__(self, passages: list[dict]):
        self.passages = passages
        # Repeat the page title to give it extra weight for entity-centric queries.
        search_docs = [
            " ".join([p["title"]] * TITLE_REPEAT + [p["text"]]) for p in passages
        ]
        self.bm25 = BM25Okapi([preprocess_text(doc) for doc in search_docs])

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        scores = self.bm25.get_scores(preprocess_text(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "score": float(scores[i]),
                "passage_id": self.passages[i]["passage_id"],
                "title": self.passages[i]["title"],
                "text": self.passages[i]["text"],
            }
            for i in ranked
        ]


def main():
    """Quick sanity check: show top-3 passages for the first 5 questions."""
    retriever = BM25Retriever(load_passages())
    with open(NQ_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    for item in questions[:5]:
        print(f"\nQuery: {item['question']}\n")
        for rank, r in enumerate(retriever.retrieve(item["question"], top_k=3), start=1):
            print(f"Rank {rank} | {r['title']} | score={r['score']:.4f}")
            print(f"{r['text'][:300]}...")
            print("-" * 80)


if __name__ == "__main__":
    main()
