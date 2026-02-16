"""
Dictionary Token Builder - builds all 4 dictionaries in one run.
Uses dictionary_builder for coding_dictionary, number_word_dictionary, token_dictionary,
and grammar_diagram for grammar_dictionary.

Output files:
  1. coding_dictionary.json    (word -> {pos, def})
  2. number_word_dictionary.json (number -> word, alphabetical)
  3. token_dictionary.json    (word_token -> {pos, def} as token lists)
  4. grammar_dictionary.json  (word -> {pos, pos_grammar, def, grammar, def_parts})

Usage:
  python dictionary_token_builder.py                    # build all 4 from scratch
  python dictionary_token_builder.py --use-existing-coding   # load coding_dictionary.json, build 2–4
  python dictionary_token_builder.py --no-grammar      # build only 1–3
  python dictionary_token_builder.py --no-tag-defs    # grammar_dictionary without def_parts (faster)
"""

import json
import os
import sys


def load_coding_dictionary(path: str = "coding_dictionary.json") -> dict:
    """Load coding_dictionary from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_all_four(
    use_existing_coding: bool = False,
    build_grammar: bool = True,
    tag_definitions: bool = True,
    verbose: bool = True,
    coding_path: str = "coding_dictionary.json",
) -> dict:
    """
    Build all 4 dictionaries in order.
    Returns dict with keys: coding_dictionary, number_word_dictionary, token_dictionary, grammar_dictionary
    (only includes keys for dictionaries that were built).
    """
    from dictionary_builder import (
        starter_vocab,
        build_dictionary,
        save_dictionary,
        build_number_word_dictionary,
        save_number_word_dictionary,
        build_token_dictionary,
        save_token_dictionary,
    )
    from grammar_diagram import build_grammar_dictionary, save_grammar_dictionary

    result = {}

    # ---------- 1. Coding dictionary ----------
    if use_existing_coding and os.path.exists(coding_path):
        if verbose:
            print(f"Loading existing {coding_path}...")
        coding_dictionary = load_coding_dictionary(coding_path)
        if verbose:
            print(f"  Loaded {len(coding_dictionary):,} entries.")
    else:
        if verbose:
            print("Building coding_dictionary (NLTK WordNet + expansion)...")
        coding_dictionary = build_dictionary(starter_vocab, max_iterations=40, verbose=verbose)
        save_dictionary(coding_dictionary, coding_path)
        if verbose:
            print(f"  Saved {coding_path} ({len(coding_dictionary):,} entries).")
    result["coding_dictionary"] = coding_dictionary

    # ---------- 2. Number–word dictionary ----------
    if verbose:
        print("Building number_word_dictionary (alphabetical, all words from coding + pos + def)...")
    number_word_dictionary = build_number_word_dictionary(coding_dictionary)
    save_number_word_dictionary(number_word_dictionary)
    if verbose:
        print(f"  Saved number_word_dictionary.json ({len(number_word_dictionary):,} entries).")
    result["number_word_dictionary"] = number_word_dictionary

    # ---------- 3. Token dictionary ----------
    if verbose:
        print("Building token_dictionary (word/pos/def tokenized by number_word_dictionary)...")
    token_dictionary = build_token_dictionary(coding_dictionary, number_word_dictionary)
    save_token_dictionary(token_dictionary)
    if verbose:
        print(f"  Saved token_dictionary.json ({len(token_dictionary):,} entries).")
    result["token_dictionary"] = token_dictionary

    # ---------- 4. Grammar dictionary ----------
    if build_grammar:
        if verbose:
            print("Building grammar_dictionary (grammar on word, pos, all words in def; using grammar_diagram rules)...")
        grammar_dictionary = build_grammar_dictionary(coding_dictionary, tag_definitions=tag_definitions)
        save_grammar_dictionary(grammar_dictionary)
        if verbose:
            print(f"  Saved grammar_dictionary.json ({len(grammar_dictionary):,} entries).")
        result["grammar_dictionary"] = grammar_dictionary
    else:
        if verbose:
            print("Skipping grammar_dictionary (--no-grammar).")

    return result


def main() -> None:
    use_existing = "--use-existing-coding" in sys.argv
    no_grammar = "--no-grammar" in sys.argv
    no_tag_defs = "--no-tag-defs" in sys.argv

    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print("Options:")
        print("  --use-existing-coding   Load coding_dictionary.json instead of building (faster rerun).")
        print("  --no-grammar            Build only coding, number_word, token (skip grammar_dictionary).")
        print("  --no-tag-defs           Build grammar_dictionary without def_parts (faster).")
        sys.exit(0)

    print("Dictionary Token Builder - building all 4 dictionaries.")
    print()
    build_all_four(
        use_existing_coding=use_existing,
        build_grammar=not no_grammar,
        tag_definitions=not no_tag_defs,
        verbose=True,
    )
    print()
    print("Done. Output files: coding_dictionary.json, number_word_dictionary.json,")
    print("                    token_dictionary.json, grammar_dictionary.json")


if __name__ == "__main__":
    main()
