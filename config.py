"""Shared paths and experiment settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

NQ_FILE = DATA_DIR / "nq_subset.json"
PASSAGES_FILE = DATA_DIR / "wiki_passages.jsonl"
WIKI_DEBUG_FILE = DATA_DIR / "wiki_page_debug.json"
RESULTS_FILE = RESULTS_DIR / "results_hf.json"

# Dataset
NQ_NUM_QUESTIONS = 200

# Knowledge base construction
WORDS_PER_PASSAGE = 100
MIN_PASSAGE_WORDS = 30
MAX_PAGES_PER_QUESTION = 2
WIKI_SEARCH_RESULTS = 8

# Retrieval
TOP_K = 5
TITLE_REPEAT = 3  # title is repeated in the indexed text to up-weight it

# Generation (Hugging Face Inference Providers, OpenAI-compatible endpoint)
HF_BASE_URL = "https://router.huggingface.co/v1"
MODEL_NAME = "katanemo/Arch-Router-1.5B:hf-inference"
TEMPERATURE = 0
MAX_TOKENS = 64
