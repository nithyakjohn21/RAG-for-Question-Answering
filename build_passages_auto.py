import json
import re
from pathlib import Path

import wikipedia
import wikipediaapi


DATA_DIR = Path("data")
NQ_FILE = DATA_DIR / "nq_subset.json"
OUT_FILE = DATA_DIR / "wiki_passages_auto.jsonl"
DEBUG_FILE = DATA_DIR / "wiki_page_debug.json"


BAD_TITLES = {
    "north", "south", "east", "west",
    "new testament", "old testament",
}

QUESTION_PREFIXES = [
    "who wrote ",
    "who is ",
    "who was ",
    "who does ",
    "who has ",
    "where did they film ",
    "where did ",
    "where is ",
    "where was ",
    "what is the capital of ",
    "what is the origin of the name ",
    "what is ",
    "what was ",
    "when was ",
    "when did ",
    "which side of ",
    "which ",
    "names of ",
    "what's ",
]


def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_passages(text: str, words_per_passage: int = 100):
    words = text.split()
    passages = []
    for i in range(0, len(words), words_per_passage):
        chunk = words[i:i + words_per_passage]
        if len(chunk) >= 30:
            passages.append(" ".join(chunk))
    return passages


def normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_bad_title(title: str) -> bool:
    t = title.strip().lower()

    if t in BAD_TITLES:
        return True

    # reject plain years like 1950, 2014
    if re.fullmatch(r"\d{4}", t):
        return True

    # reject pure numbers
    if re.fullmatch(r"\d+", t):
        return True

    return False


def extract_focus(question: str) -> str:
    q = question.lower().strip().rstrip("?")

    for p in QUESTION_PREFIXES:
        if q.startswith(p):
            return q[len(p):].strip()

    return q


def build_search_queries(question: str, gold_answer: str):
    q = question.lower().strip().rstrip("?")
    focus = extract_focus(question)

    queries = []

    # general focus
    if focus:
        queries.append(focus)

    # special patterns
    m = re.match(r"who wrote (.+)", q)
    if m:
        work = m.group(1).strip()
        queries.append(work)

    m = re.match(r"what is the capital of (.+)", q)
    if m:
        place = m.group(1).strip()
        queries.append(place)

    m = re.match(r"where did they film (.+)", q)
    if m:
        thing = m.group(1).strip()
        queries.append(thing)

    m = re.match(r"who does (.+) work for (.+)", q)
    if m:
        person = m.group(1).strip()
        work = m.group(2).strip()
        queries.append(person)
        queries.append(work)
        queries.append(f"{person} {work}")

    # use gold answer only if it is not obviously a bad page title
    if gold_answer and not is_bad_title(gold_answer):
        queries.append(gold_answer)

    # fallback
    queries.append(q)

    # deduplicate
    seen = set()
    final = []
    for x in queries:
        x = normalize(x)
        if x and x.lower() not in seen:
            seen.add(x.lower())
            final.append(x)

    return final


def wiki_search_titles(query: str, max_results: int = 5):
    try:
        return wikipedia.search(query, results=max_results)
    except Exception:
        return []


def fetch_page(wiki, title: str):
    try:
        page = wiki.page(title)
        if page.exists() and len(page.text.strip()) > 0:
            return page
    except Exception:
        return None
    return None

def tokenize_for_match(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return set(text.split())


def score_title(question: str, gold_answer: str, title: str):
    import re

    q = question.lower().strip()
    a = gold_answer.lower().strip()
    t = title.lower().strip()

    q_tokens = tokenize_for_match(question)
    a_tokens = tokenize_for_match(gold_answer)
    t_tokens = tokenize_for_match(title)

    score = 0

    # overlap rewards
    score += 2 * len(q_tokens & t_tokens)
    score += 3 * len(a_tokens & t_tokens)

    # exact/near-exact match rewards
    if t in q:
        score += 8
    if a and a in t:
        score += 8

    # stronger rewards for key extracted phrases
    m = re.match(r"who wrote (.+)", q)
    if m:
        target = m.group(1).strip()
        if t == target:
            score += 12

    m = re.match(r"what is the capital of (.+)", q)
    if m:
        target = m.group(1).strip()
        if t == target:
            score += 12

    m = re.match(r"where did they film (.+)", q)
    if m:
        target = m.group(1).strip()
        if t == target:
            score += 15
        elif t.startswith(target):
            score += 6

    # penalties
    if "disambiguation" in t:
        score -= 6
    if t.startswith("list of"):
        score -= 4

    # penalize bare years
    if re.fullmatch(r"\d{4}", t):
        score -= 6

    # penalize sequel / numbered variant pages a bit
    if re.search(r"\b2\b|\bii\b|\biii\b|\biv\b", t):
        score -= 4

    # penalize TV-series/version pages when simpler title exists
    if "(" in t and ")" in t:
        score -= 2

    return score

def build_wikipedia_passages():
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="ml4nlu-term-paper-experiment/1.0"
    )

    questions = load_questions(NQ_FILE)

    seen_titles = set()
    all_passages = []
    debug_rows = []
    passage_id = 1

    for item in questions:
        question = item["question"]
        gold_answer = item["gold_answer"]

        search_queries = build_search_queries(question, gold_answer)

        found_pages = []
        tried_titles = []

        print(f"\nProcessing: {question}")
        print(f"Gold answer: {gold_answer}")
        print("Search queries:", search_queries)

        candidate_pool = []

        for sq in search_queries:
            titles = wiki_search_titles(sq, max_results=8)

            for title in titles:
                tried_titles.append(title)

                if is_bad_title(title):
                    continue

                score = score_title(question, gold_answer, title)
                candidate_pool.append((score, title))

        # sort best titles first
        candidate_pool = sorted(candidate_pool, key=lambda x: x[0], reverse=True)

        seen_local = set()

        for score, title in candidate_pool:
            if title.lower() in seen_local:
                continue
            seen_local.add(title.lower())

            page = fetch_page(wiki, title)
            if page is None:
                continue

            if page.title not in seen_titles:
                seen_titles.add(page.title)
                found_pages.append(page.title)

                text = clean_text(page.text)
                passages = split_into_passages(text, words_per_passage=100)

                for p in passages:
                    all_passages.append({
                        "passage_id": passage_id,
                        "title": page.title,
                        "text": p
                    })
                    passage_id += 1

                print(f"  Added {len(passages)} passages from: {page.title} (score={score})")

            # keep KB focused: at most 2 pages per question
            if len(found_pages) >= 2:
                break

        debug_rows.append({
            "question": question,
            "gold_answer": gold_answer,
            "search_queries": search_queries,
            "tried_titles": tried_titles,
            "found_pages": found_pages
        })

        if not found_pages:
            print("  No useful page found.")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for row in all_passages:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(DEBUG_FILE, "w", encoding="utf-8") as f:
        json.dump(debug_rows, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_passages)} passages to {OUT_FILE}")
    print(f"Saved debug info to {DEBUG_FILE}")


if __name__ == "__main__":
    build_wikipedia_passages()