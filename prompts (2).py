def build_baseline_prompt(question: str) -> str:
    return f"""Answer the following factual question briefly.
Give only a short answer phrase, not a full sentence.

Question: {question}
Answer:"""


def build_retrieval_prompt(question: str, passages: list[dict]) -> str:
    context_blocks = []
    for i, p in enumerate(passages, start=1):
        context_blocks.append(f"[{i}] Title: {p['title']}\n{p['text']}")

    context = "\n\n".join(context_blocks)

    return f"""You are answering a factual question.

Use only the context below.
If the answer is present, return only the shortest exact answer phrase.
If the answer is not present, say exactly:
I could not find the answer in the context.

Context:
{context}

Question: {question}
Exact answer:"""
