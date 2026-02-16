"""
Dictionary Builder for Coding-Focused LLM/Agent
------------------------------------------------
- Starts with vocabulary from common user inputs (common_coding_vocab.json from
  common_prompts.txt) if present, else a curated list of ~1020 coding-related words
- Uses NLTK WordNet to assign POS (noun/verb/etc.) and a primary definition
- Recursively expands by extracting words from definitions and adding them
- Saves result as coding_dictionary.json

To build from prompts: python vocab_from_prompts.py  then run this script.
Run once to generate the file, then use it in dict_agent.py
"""

import json
import os
import re
from collections import deque

# ================== 1. Starter Vocab: from common prompts or curated list ==================
COMMON_VOCAB_FILE = "common_coding_vocab.json"

def _load_starter_vocab() -> list[str]:
    """Use vocab from common user inputs (common_coding_vocab.json) if present, else built-in list."""
    if os.path.exists(COMMON_VOCAB_FILE):
        try:
            with open(COMMON_VOCAB_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list) and loaded:
                return [str(w).lower() for w in loaded]
        except Exception:
            pass
    return _curated_starter_vocab()

def _curated_starter_vocab() -> list[str]:
    """Curated list of coding-related words (used when common_coding_vocab.json is not present)."""
    return [
    # Core Action Verbs
    "write", "create", "generate", "build", "make", "implement", "code", "develop", "fix", "debug", "correct", "solve",
    "improve", "optimize", "refactor", "rewrite", "convert", "translate", "add", "remove", "update", "modify",
    "explain", "describe", "show", "tell", "help", "assist", "provide", "give", "output", "return", "print", "display",
    "handle", "process", "validate", "check", "test", "run", "execute", "call", "invoke", "use", "import", "export",
    "define", "declare", "initialize", "throw", "catch", "try", "await", "async", "review", "analyze", "suggest",
    "propose", "design", "structure", "organize", "clean", "format", "style", "lint", "parse", "serialize",
    "deserialize", "encode", "decode", "encrypt", "decrypt", "hash", "authenticate", "authorize", "log", "trace",
    "monitor", "profile", "benchmark", "measure", "compare", "sort", "search", "filter", "map", "reduce", "group",
    "aggregate", "join", "merge", "split", "combine", "extract", "transform", "load", "query", "fetch", "send",
    "receive", "post", "get", "put", "delete", "patch",

    # Prompt-Engineering Glue & Instructions
    "please", "can", "you", "could", "help", "me", "need", "have", "here", "using", "with", "in", "for", "to", "that",
    "which", "when", "if", "then", "else", "step", "by", "explain", "reason", "chain", "of", "thought", "think",
    "carefully", "detailed", "simple", "clear", "concise", "clean", "best", "way", "practices", "efficient", "modern",
    "professional", "example", "give", "an", "sample", "code", "full", "complete", "just", "the", "only", "no",
    "explanation", "comments", "add", "docstring", "markdown", "code", "block", "fenced", "output", "avoid", "include",
    "remove", "use", "external", "libraries", "standard", "library", "pythonic", "idiomatic", "es6", "modern", "syntax",
    "handle", "edge", "cases", "error", "handling", "include", "tests", "unit", "integration", "with", "tests",
    "expected", "output", "input", "example", "constraints", "requirements", "follow", "adhere", "to", "respect",
    "based", "on", "extend", "modify", "from", "this", "improve", "refactor", "optimize", "make", "faster", "smaller",
    "readable", "maintainable", "act", "as", "you", "are", "role", "expert", "senior", "developer", "python", "expert",
    "react", "developer",

    # Languages & Frameworks / Libraries
    "python", "javascript", "js", "react", "node", "nodejs", "typescript", "ts", "html", "css", "java", "csharp", "c",
    "cpp", "c++", "sql", "rust", "go", "golang", "swift", "kotlin", "ruby", "php", "bash", "shell", "powershell",
    "docker", "kubernetes", "aws", "azure", "gcp", "firebase", "react", "native", "nextjs", "vue", "angular", "svelte",
    "django", "flask", "fastapi", "express", "nestjs", "spring", "springboot", "laravel", "symfony", "tensorflow",
    "pytorch", "keras", "scikit", "learn", "pandas", "numpy", "huggingface", "transformers", "langchain", "openai",
    "anthropic", "groq", "ollama", "tailwind", "bootstrap", "materialui", "chakraui", "antd", "jest", "pytest",
    "unittest", "vitest", "cypress", "selenium", "graphql", "apollo", "relay", "prisma", "sqlalchemy", "mongoose",
    "sequelize", "redis", "kafka", "rabbitmq", "postgresql", "mysql", "sqlite", "mongodb", "supabase", "vercel",
    "netlify", "heroku", "railway", "flyio",

    # Code Elements & Artifacts
    "function", "def", "method", "class", "object", "component", "hook", "state", "props", "ref", "context", "reducer",
    "effect", "usestate", "useeffect", "usereducer", "usecontext", "useref", "variable", "const", "let", "var",
    "array", "list", "tuple", "dict", "dictionary", "set", "map", "promise", "async", "await", "callback", "event",
    "listener", "handler", "api", "endpoint", "route", "request", "response", "json", "xml", "yaml", "toml", "file",
    "path", "directory", "folder", "module", "package", "import", "from", "export", "default", "interface", "type",
    "enum", "struct", "union", "pointer", "loop", "for", "while", "do", "if", "else", "elif", "switch", "case",
    "break", "continue", "try", "catch", "except", "finally", "raise", "throw", "error", "exception", "bug", "issue",
    "traceback", "stack", "trace", "log", "console", "log", "print", "debug", "breakpoint", "test", "unit", "test",
    "assert", "mock", "patch", "spy", "fixture", "return", "yield", "generator", "lambda", "anonymous",
    "comprehension", "list", "comprehension", "dict", "comprehension", "set", "comprehension", "decorator",
    "annotation", "metadata", "middleware", "controller", "model", "view", "template", "service", "repository",
    "schema", "migration", "validation", "form", "input", "button", "div", "span", "paragraph", "header", "footer",
    "nav", "section", "article", "aside", "main", "flex", "grid", "layout", "responsive", "media", "query", "css",
    "class", "id", "selector", "pseudo", "hover", "focus", "active", "key", "index", "value", "item", "element",
    "child", "parent", "sibling", "prop", "drilling", "lifting", "state", "memo", "callback", "usememo", "usecallback",

    # Constraints & Qualifiers
    "fast", "efficient", "performant", "secure", "safe", "clean", "readable", "maintainable", "scalable", "simple",
    "minimal", "short", "one", "liner", "no", "dependencies", "no", "external", "libraries", "use", "only", "built",
    "in", "standard", "library", "pythonic", "idiomatic", "modern", "es2020", "es6", "async", "await", "promises",
    "functional", "object", "oriented", "oop", "immutable", "pure", "function", "side", "effect", "free", "handle",
    "edge", "cases", "add", "error", "handling", "validation", "sanitization", "logging", "comments", "docstring",
    "type", "hints", "mypy", "pep8", "black", "ruff", "lint", "no", "console", "no", "print", "return", "value",
    "input", "output", "stdin", "stdout", "file", "input", "example", "input", "expected", "output", "constraints",
    "n", "log", "n", "o", "1", "space", "time", "complexity", "space", "complexity", "big", "o", "optimized",
    "vectorized", "parallel", "concurrent", "thread", "safe", "async", "safe", "production", "ready", "ready", "to",
    "deploy", "with", "setup", "with", "requirements", "txt", "with", "dockerfile", "with", "readme",

    # Debugging & Errors
    "error", "bug", "issue", "problem", "exception", "traceback", "stack", "overflow", "undefined", "null", "nil",
    "none", "nan", "typeerror", "valueerror", "keyerror", "indexerror", "attributeerror", "syntaxerror", "importerror",
    "filenotfounderror", "connectionerror", "timeout", "httperror", "failed", "not", "working", "crashes", "hangs",
    "slow", "memory", "leak", "infinite", "loop", "recursion", "error", "stack", "overflow", "wrong", "output",
    "unexpected", "behavior", "incorrect", "result", "fix", "bug", "debug", "code", "what", "wrong", "why", "is",
    "this", "happening", "how", "to", "fix", "explain", "error", "resolve", "issue", "handle", "exception", "raise",
    "custom", "error", "log", "error", "print", "debug", "breakpoint", "here",

    # UI/Frontend-Specific
    "button", "form", "input", "textarea", "select", "option", "checkbox", "radio", "label", "fieldset", "legend",
    "table", "tr", "td", "th", "thead", "tbody", "ul", "ol", "li", "div", "span", "p", "h1", "h2", "h3", "header",
    "footer", "nav", "aside", "main", "section", "article", "dialog", "modal", "alert", "toast", "notification",
    "dropdown", "menu", "accordion", "tabs", "carousel", "slider", "tooltip", "popover", "badge", "avatar", "card",
    "grid", "flexbox", "container", "wrapper", "layout", "responsive", "mobile", "first", "dark", "mode", "theme",
    "color", "scheme", "css", "variable", "tailwind", "class", "bootstrap", "class", "styled", "components", "emotion",
    "sass", "scss", "less", "postcss", "jsx", "tsx", "render", "rerender", "mount", "unmount", "componentdidmount",
    "componentwillunmount", "useeffect", "cleanup", "key", "prop", "fragment", "portal", "suspense", "lazy",
    "error", "boundary"
]

# Deduplicate and clean (keep words that are alphabetic, alphanumeric, or contain - _ +)
def _keep_word(w: str) -> bool:
    if len(w) < 1:
        return False
    w = w.lower()
    if w.isalpha() or w.isalnum():
        return True
    if "-" in w or "_" in w or "+" in w:
        return True
    return False

starter_vocab = sorted(set(w.lower() for w in _load_starter_vocab() if _keep_word(w)))

print(f"Starting with {len(starter_vocab)} unique words" + (f" (from {COMMON_VOCAB_FILE})" if os.path.exists(COMMON_VOCAB_FILE) else " (curated list)"))

# ================== 2. NLTK / WordNet Setup ==================
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.corpus import wordnet as wn

def get_pos_and_def(word: str) -> tuple[str, str]:
    """
    Get the most common POS tag and definition from WordNet.
    Fallback to coding-specific heuristics if no synset found.
    """
    synsets = wn.synsets(word)
    if synsets:
        syn = synsets[0]  # most frequent sense
        pos_map = {'n': 'noun', 'v': 'verb', 'a': 'adjective', 'r': 'adverb', 's': 'adjective'}
        pos = pos_map.get(syn.pos(), 'unknown')
        definition = syn.definition().strip().lower()
        return pos, definition

    # Coding-specific fallbacks
    if word in {"react", "nextjs", "tailwind", "chakraui", "antd", "usestate", "useeffect", "hook", "component", "jsx", "tsx"}:
        return "noun", f"a {word} is a concept, component, hook or library in modern JavaScript / React development"
    if word.endswith("error") or word in {"bug", "exception", "traceback", "issue"}:
        return "noun", f"an {word} refers to a problem, failure or unexpected behavior in software"
    if word in {"pythonic", "idiomatic", "pep8", "mypy", "ruff", "black"}:
        return "adjective", f"describes code or style that follows {word} conventions or best practices"
    if word in {"async", "await", "promise", "callback"}:
        return "adjective", f"relating to asynchronous programming patterns in JavaScript"

    # Generic fallback
    return "noun", f"a {word} is a term commonly used in software development, programming, or web technologies"

# ================== 3. Build & Expand Dictionary ==================
def build_dictionary(
    vocab: list[str],
    max_iterations: int = 40,
    verbose: bool = True,
) -> dict[str, dict[str, str]]:
    """
    Build a dictionary of word -> {pos, def} using NLTK WordNet.
    Expands recursively from definitions until no new words are found.
    """
    dictionary: dict[str, dict[str, str]] = {}
    all_words = set(vocab)
    queue = deque(vocab)

    iteration = 0
    while queue and iteration < max_iterations:
        iteration += 1
        current_batch = list(queue)
        queue.clear()
        added_this_round = 0

        for word in current_batch:
            if word in dictionary:
                continue

            pos, definition = get_pos_and_def(word)
            dictionary[word] = {"pos": pos, "def": definition}

            tokens = re.findall(r"[a-z]+", definition.lower())
            for token in tokens:
                if token not in all_words and len(token) > 2:
                    all_words.add(token)
                    queue.append(token)
                    added_this_round += 1

        if verbose:
            print(f"Iter {iteration:2d} | New words added: {added_this_round:4d} | Total words: {len(all_words):6d}")

    return dictionary


def save_dictionary(dictionary: dict, path: str = "coding_dictionary.json") -> None:
    """Save dictionary to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, indent=2, ensure_ascii=False)


def build_number_word_dictionary(word_dictionary: dict) -> dict[str, str]:
    """
    Build number->word mapping from all words in the coding dictionary:
    - every main entry key (word)
    - every pos value (noun, verb, adjective, etc.)
    - every word token in each definition (def)
    All collected words are deduplicated, sorted alphabetically (case-insensitive),
    then numbered 1, 2, 3, ...
    Returns a number-to-word dict (keys as strings for JSON: "1", "2", ...).
    """
    all_words: set[str] = set()
    for word, entry in word_dictionary.items():
        all_words.add(word)
        all_words.add(entry["pos"].strip().lower())
        # tokenize definition into words (letters only)
        for token in re.findall(r"[a-z]+", entry["def"].lower()):
            all_words.add(token)
    sorted_words = sorted(all_words, key=str.lower)
    return {str(i): w for i, w in enumerate(sorted_words, start=1)}


def save_number_word_dictionary(
    number_word_dictionary: dict, path: str = "number_word_dictionary.json"
) -> None:
    """Save number->word mapping to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(number_word_dictionary, f, indent=2, ensure_ascii=False)


def build_token_dictionary(
    coding_dictionary: dict,
    number_word_dictionary: dict,
) -> dict[str, dict[str, list[str] | str]]:
    """
    Tokenize the full coding_dictionary using number_word_dictionary.
    Word, pos, and def are all tokenized. Each entry is:
    {word_token: {"pos": pos_token, "def": [def_tokens, ...]}}.
    Words not in number_word_dictionary get token "0" (unknown).
    """
    # word -> number (string "1", "2", ...)
    word_to_number = {w: num for num, w in number_word_dictionary.items()}
    unknown = "0"

    token_dictionary: dict[str, dict[str, list[str] | str]] = {}
    for word, entry in coding_dictionary.items():
        word_token = word_to_number.get(word.lower() if isinstance(word, str) else word, unknown)
        pos_token = word_to_number.get(entry["pos"].strip().lower(), unknown)
        def_tokens = [
            word_to_number.get(tok, unknown)
            for tok in re.findall(r"[a-z]+", entry["def"].lower())
        ]
        token_dictionary[word_token] = {"pos": pos_token, "def": def_tokens}
    return token_dictionary


def save_token_dictionary(
    token_dictionary: dict, path: str = "token_dictionary.json"
) -> None:
    """Save tokenized dictionary to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(token_dictionary, f, indent=2, ensure_ascii=False)


# ================== 4. Run when executed as script ==================
if __name__ == "__main__":
    print("Building dictionary...")
    dictionary = build_dictionary(starter_vocab, max_iterations=40, verbose=True)
    print(f"\nDone. Final dictionary size: {len(dictionary):,} entries")
    print(f"Started with {len(starter_vocab):,} -> added {len(dictionary) - len(starter_vocab):,} new words")
    save_dictionary(dictionary)
    print("Saved to: coding_dictionary.json")

    number_word_dictionary = build_number_word_dictionary(dictionary)
    save_number_word_dictionary(number_word_dictionary)
    print("Saved to: number_word_dictionary.json")

    token_dictionary = build_token_dictionary(dictionary, number_word_dictionary)
    save_token_dictionary(token_dictionary)
    print("Saved to: token_dictionary.json")
    print("You can now use these files in your dict_agent.py script.")