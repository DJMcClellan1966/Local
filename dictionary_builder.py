"""
Dictionary Builder for Coding-Focused LLM/Agent
------------------------------------------------
- Starts with a curated list of ~1020 coding-related words/phrases
- Uses NLTK WordNet to assign POS (noun/verb/etc.) and a primary definition
- Recursively expands by extracting words from definitions and adding them
- Continues until no new words are found (fully closed/self-contained dictionary)
- Saves result as coding_dictionary.json

Run once to generate the file, then use it in dict_agent.py
"""

import json
import re
from collections import deque

# ================== 1. Curated Starter Vocab (~1020 items) ==================
# This is the deduplicated, grouped list from our earlier curation
starter_vocab = [
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

# Deduplicate and clean
starter_vocab = sorted(set(w.lower() for w in starter_vocab if len(w) > 1 and w.isalpha() or '-' in w or '_' in w))

print(f"Starting with {len(starter_vocab)} unique words")

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
dictionary = {}
all_words = set(starter_vocab)
queue = deque(starter_vocab)

print("Building dictionary...")

iteration = 0
max_iterations = 40  # safety limit

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

        # Extract new words from this definition
        tokens = re.findall(r"[a-z]+", definition.lower())
        for token in tokens:
            if token not in all_words and len(token) > 2:
                all_words.add(token)
                queue.append(token)
                added_this_round += 1

    print(f"Iter {iteration:2d} | New words added: {added_this_round:4d} | Total words: {len(all_words):6d}")

print(f"\nDone. Final dictionary size: {len(dictionary):,} entries")
print(f"Started with {len(starter_vocab):,} → added {len(dictionary) - len(starter_vocab):,} new words")

# ================== 4. Save to JSON ==================
with open("coding_dictionary.json", "w", encoding="utf-8") as f:
    json.dump(dictionary, f, indent=2, ensure_ascii=False)

print("\nSaved to: coding_dictionary.json")
print("You can now use this file in your dict_agent.py script.")