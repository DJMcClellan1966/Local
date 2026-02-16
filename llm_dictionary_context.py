"""
LLM context builder using dictionaries and grammar rules.
Lets an LLM "see" pos, def, and grammar for words plus phrase-structure rules
and learned sentence/POS patterns, so it can produce more consistent, grammar-aware text.

Usage:
  from llm_dictionary_context import load_all_dictionaries, build_llm_context, get_usual_patterns_text
  ctx = load_all_dictionaries()
  prompt_context = build_llm_context("Create a login form with validation", dictionaries=ctx)
  full_prompt = prompt_context + "\n\nUser request: " + user_input
  # Then call your LLM with full_prompt
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Optional

# Paths (override via env or arguments)
DEFAULT_PATHS = {
    "coding_dictionary": "coding_dictionary.json",
    "number_word_dictionary": "number_word_dictionary.json",
    "token_dictionary": "token_dictionary.json",
    "grammar_dictionary": "grammar_dictionary.json",
}


def load_dictionary(path: str) -> dict:
    """Load a JSON dictionary; return {} if missing."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_dictionaries(
    dir_path: str = ".",
    paths: Optional[dict] = None,
) -> dict:
    """
    Load all 4 dictionaries from dir_path.
    Returns dict with keys: coding_dictionary, number_word_dictionary, token_dictionary, grammar_dictionary.
    """
    paths = paths or DEFAULT_PATHS
    out = {}
    for key, filename in paths.items():
        full = os.path.join(dir_path, filename)
        out[key] = load_dictionary(full)
    return out


def get_word_info(
    word: str,
    grammar_dictionary: dict,
    coding_dictionary: Optional[dict] = None,
) -> dict:
    """
    Get pos, def, grammar (and pos_grammar, def_parts) for a word.
    Uses grammar_dictionary; falls back to coding_dictionary for pos/def if grammar_dictionary missing fields.
    """
    w = word.lower().strip()
    g = grammar_dictionary.get(w) or grammar_dictionary.get(word)
    if g:
        return {
            "word": w,
            "pos": g.get("pos", "unknown"),
            "def": g.get("def", ""),
            "grammar": g.get("grammar", "N"),
            "pos_grammar": g.get("pos_grammar", "N"),
            "def_parts": g.get("def_parts", []),
        }
    c = (coding_dictionary or {}).get(w) or (coding_dictionary or {}).get(word)
    if c:
        return {
            "word": w,
            "pos": c.get("pos", "unknown"),
            "def": c.get("def", ""),
            "grammar": "N",
            "pos_grammar": "N",
            "def_parts": [],
        }
    return {"word": w, "pos": "unknown", "def": "", "grammar": "N", "pos_grammar": "N", "def_parts": []}


def get_grammar_rules_text() -> str:
    """Return phrase-structure rules as readable text (from grammar_diagram)."""
    try:
        from grammar_diagram import get_grammar_rules
        rules = get_grammar_rules()
        lines = ["Phrase-structure rules:", "  S -> NP VP  (sentence = noun phrase + verb phrase)"]
        for lhs, rhs in rules:
            if lhs != "S" or rhs != ["NP", "VP"]:
                lines.append(f"  {lhs} -> {' '.join(rhs)}")
        return "\n".join(lines[:20])  # cap for context size
    except Exception:
        return "Phrase-structure rules: S -> NP VP; NP -> Det N | N | NP PP; VP -> V | V NP | V PP; PP -> P NP."


def extract_words_from_text(text: str, max_words: int = 50) -> list[str]:
    """Extract lowercase alphabetic tokens from text for dictionary lookup."""
    tokens = re.findall(r"[a-z]+", text.lower())
    seen = set()
    out = []
    for t in tokens:
        if t not in seen and len(t) > 1:
            seen.add(t)
            out.append(t)
            if len(out) >= max_words:
                break
    return out


def format_word_info(info: dict, include_def_parts: bool = False, def_parts_max: int = 5) -> str:
    """Format a single word's info for LLM context."""
    defn = info.get("def") or ""
    def_str = (defn[:80] + "...") if len(defn) > 80 else defn
    parts = [f"  {info.get('word', '')}: pos={info.get('pos', 'unknown')}, grammar={info.get('grammar', 'N')}, def={def_str}"]
    if include_def_parts and info.get("def_parts"):
        sample = info["def_parts"][:def_parts_max]
        parts.append(f"    def_parts (sample): {', '.join(f'{w}({g})' for w, g in sample)}")
    return "\n".join(parts)


# ================== Pattern learning ==================
def get_usual_patterns_from_grammar_dictionary(grammar_dictionary: dict) -> dict:
    """
    Aggregate patterns from grammar_dictionary: grammar symbol counts and example words.
    Helps the LLM see "usual" grammar roles (e.g. many words are N, Det is rare).
    """
    grammar_counts = Counter()
    grammar_examples: dict[str, list[str]] = {}
    for word, entry in grammar_dictionary.items():
        g = entry.get("grammar", "N")
        grammar_counts[g] += 1
        if g not in grammar_examples:
            grammar_examples[g] = []
        if len(grammar_examples[g]) < 8:
            grammar_examples[g].append(word)
    return {"counts": dict(grammar_counts), "examples": grammar_examples}


def get_usual_patterns_from_sentences(
    sentences: list[str],
    max_sentences: int = 20,
) -> list[tuple[str, str]]:
    """
    Diagram each sentence and return (sentence, bracket_structure).
    Teaches the LLM usual sentence patterns (e.g. S (NP (Det The) (N cat)) (VP (V sat) ...)).
    """
    try:
        from grammar_diagram import diagram_sentence, tree_to_brackets
    except ImportError:
        return []
    results = []
    for s in sentences[:max_sentences]:
        s = s.strip()
        if not s:
            continue
        try:
            root, _ = diagram_sentence(s, style="tree")
            if root:
                results.append((s, tree_to_brackets(root)))
        except Exception:
            pass
    return results


def get_usual_patterns_text(
    grammar_dictionary: dict,
    example_sentences: Optional[list[str]] = None,
) -> str:
    """
    Build a single "usual patterns" text block for LLM context:
    - Grammar symbol counts and example words
    - Optional: example sentence -> structure pairs
    """
    lines = ["Usual patterns (from dictionary and grammar):"]
    pats = get_usual_patterns_from_grammar_dictionary(grammar_dictionary)
    lines.append("  Grammar symbol counts (word types): " + ", ".join(f"{g}={c}" for g, c in sorted(pats["counts"].items(), key=lambda x: -x[1])))
    lines.append("  Example words by grammar: " + "; ".join(f"{g}: {', '.join(pats['examples'].get(g, [])[:5])}" for g in ["Det", "N", "V", "P", "Adj", "Adv"] if pats["examples"].get(g)))
    if example_sentences:
        pairs = get_usual_patterns_from_sentences(example_sentences)
        if pairs:
            lines.append("  Example sentence structures:")
            for sent, bracket in pairs[:5]:
                s = sent[:50] + ("..." if len(sent) > 50 else "")
                b = bracket[:70] + ("..." if len(bracket) > 70 else "")
                lines.append(f"    \"{s}\" -> {b}")
    return "\n".join(lines)


# Default example sentences for pattern learning (coding/prose style)
DEFAULT_EXAMPLE_SENTENCES = [
    "The function returns a value.",
    "Create a new component with state.",
    "The user can submit the form.",
    "We need to validate the input.",
    "Add error handling to the request.",
    "The API returns JSON data.",
    "Build a simple login form.",
    "Use the dictionary for context.",
]


def build_llm_context(
    user_input: str,
    dictionaries: Optional[dict] = None,
    include_grammar_rules: bool = True,
    include_usual_patterns: bool = True,
    include_word_info: bool = True,
    max_words: int = 30,
    include_def_parts_sample: bool = False,
    example_sentences: Optional[list[str]] = None,
) -> str:
    """
    Build context string to prepend to an LLM prompt so it "sees" pos, def, grammar rules,
    and usual patterns. Use this to get a better, grammar-aware response.

    Returns a single string (multiple sections) to place before the user's request.
    """
    if dictionaries is None:
        dictionaries = load_all_dictionaries()
    coding = dictionaries.get("coding_dictionary") or {}
    grammar_d = dictionaries.get("grammar_dictionary") or {}
    sections = []

    if include_grammar_rules:
        sections.append(get_grammar_rules_text())

    if include_usual_patterns and grammar_d:
        example_sents = example_sentences or DEFAULT_EXAMPLE_SENTENCES
        sections.append(get_usual_patterns_text(grammar_d, example_sents))

    if include_word_info and (grammar_d or coding):
        words = extract_words_from_text(user_input, max_words=max_words)
        if words:
            lines = ["Relevant words (pos, grammar, definition):"]
            for w in words[:max_words]:
                info = get_word_info(w, grammar_d, coding)
                if info.get("def") or info.get("pos") != "unknown":
                    lines.append(format_word_info(info, include_def_parts=include_def_parts_sample))
            if len(lines) > 1:
                sections.append("\n".join(lines))

    return "\n\n".join(sections) if sections else ""


def build_llm_prompt(
    user_input: str,
    dictionaries: Optional[dict] = None,
    system_prefix: str = "You are a precise, grammar-aware assistant. Use the dictionary and grammar context below.\n\n",
    **kwargs,
) -> str:
    """
    Full prompt for the LLM: system prefix + context (rules, patterns, word info) + user request.
    """
    context = build_llm_context(user_input, dictionaries=dictionaries, **kwargs)
    prompt = system_prefix
    if context:
        prompt += "Context (dictionary + grammar):\n" + context + "\n\n"
    prompt += "User request: " + user_input
    return prompt


# ================== Optional: integrate with dict_agent ==================
def retrieve_with_grammar(
    query: str,
    coding_dictionary: dict,
    grammar_dictionary: dict,
    top_k: int = 10,
) -> str:
    """
    Like dict_agent.retrieve but each line includes grammar symbol.
    Returns multiline string: "word (pos, grammar): def"
    """
    query_lower = query.lower()
    words = set(re.findall(r"[a-z]+", query_lower))
    matches = []
    for word, entry in coding_dictionary.items():
        score = sum(1 for w in words if w in word or w in entry.get("def", "").lower())
        if score > 0:
            g = grammar_dictionary.get(word) or {}
            grammar = g.get("grammar", "N")
            pos = entry.get("pos", "unknown")
            defn = entry.get("def", "")[:100]
            matches.append((score, word, pos, grammar, defn))
    matches.sort(key=lambda x: -x[0])
    return "\n".join(f"- {w} ({p}, {g}): {d}..." for _, w, p, g, d in matches[:top_k])
