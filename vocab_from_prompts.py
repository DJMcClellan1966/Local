"""
Build a dictionary vocabulary from common user inputs (prompts) for building code and apps.
Words that appear in these prompts become the seed for the coding dictionary, so the
dictionary reflects what users actually say.

Usage:
  python vocab_from_prompts.py [common_prompts.txt] [common_coding_vocab.json]
  Then run dictionary_builder (or dictionary_token_builder); it will use common_coding_vocab.json if present.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

# Minimal fallback so we always have something to build from
FALLBACK_VOCAB = [
    "create", "build", "add", "form", "button", "api", "react", "python", "function",
    "input", "validation", "data", "user", "code", "app", "component", "request",
]


def extract_words(text: str, min_len: int = 1) -> list[str]:
    """Extract lowercase alphabetic tokens; allow alphanumeric and hyphen/underscore."""
    tokens = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())
    return [t for t in tokens if len(t) >= min_len]


def load_prompts(path: str) -> list[str]:
    """Load prompts from file (one per line; skip empty and # lines)."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def vocab_from_prompts(
    prompts: list[str],
    min_frequency: int = 1,
    min_length: int = 1,
    max_words: int = 5000,
) -> list[str]:
    """
    Extract vocabulary from prompts: all unique words, optionally ranked by frequency.
    Returns sorted list of words (most frequent first if min_frequency > 1).
    """
    counter = Counter()
    for line in prompts:
        for w in extract_words(line, min_len=min_length):
            counter[w] += 1
    # Filter by frequency, then sort by (freq desc, word asc)
    items = [(w, c) for w, c in counter.items() if c >= min_frequency]
    items.sort(key=lambda x: (-x[1], x[0]))
    words = [w for w, _ in items[:max_words]]
    return words


def build_common_coding_vocab(
    prompts_path: str = "common_prompts.txt",
    output_path: str = "common_coding_vocab.json",
    merge_with_fallback: bool = True,
    min_frequency: int = 1,
    min_length: int = 1,
    max_words: int = 5000,
) -> list[str]:
    """
    Build vocabulary from prompts file and save to JSON list.
    If merge_with_fallback, union with FALLBACK_VOCAB so we never have empty.
    """
    path = Path(prompts_path)
    if not path.exists():
        if merge_with_fallback:
            vocab = sorted(set(FALLBACK_VOCAB))
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(vocab, f, indent=2)
            print(f"Prompts file not found: {prompts_path}; wrote fallback vocab ({len(vocab)} words) to {output_path}")
            return vocab
        raise FileNotFoundError(prompts_path)
    prompts = load_prompts(str(path))
    words = vocab_from_prompts(prompts, min_frequency=min_frequency, min_length=min_length, max_words=max_words)
    if merge_with_fallback:
        words = sorted(set(words) | set(FALLBACK_VOCAB))
    else:
        words = sorted(set(words))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(words, f, indent=2)
    print(f"From {len(prompts)} prompts -> {len(words)} unique words -> {output_path}")
    return words


if __name__ == "__main__":
    prompts_path = sys.argv[1] if len(sys.argv) > 1 else "common_prompts.txt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "common_coding_vocab.json"
    build_common_coding_vocab(prompts_path, output_path)
