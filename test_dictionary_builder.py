"""
Tests for dictionary_builder: NLTK WordNet integration and dictionary structure.
Requires: pip install nltk && python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
"""

import json
import os
import tempfile
import pytest

# NLTK must be available for these tests
pytest.importorskip("nltk")

from dictionary_builder import (
    get_pos_and_def,
    build_dictionary,
    save_dictionary,
    starter_vocab,
)


def test_get_pos_and_def_returns_tuple():
    """get_pos_and_def returns (pos, definition) both strings."""
    pos, definition = get_pos_and_def("function")
    assert isinstance(pos, str)
    assert isinstance(definition, str)
    assert len(pos) > 0
    assert len(definition) > 0


def test_get_pos_and_def_pos_values():
    """POS is one of noun, verb, adjective, adverb, unknown."""
    valid_pos = {"noun", "verb", "adjective", "adverb", "unknown"}
    for word in ["write", "code", "error", "fast"]:
        pos, _ = get_pos_and_def(word)
        assert pos in valid_pos, f"word={word} pos={pos}"


def test_get_pos_and_def_coding_fallback():
    """Coding-specific words get fallback definitions (e.g. react, jsx)."""
    pos, definition = get_pos_and_def("react")
    assert "react" in definition.lower() or "javascript" in definition.lower()
    pos2, definition2 = get_pos_and_def("jsx")
    assert "jsx" in definition2.lower() or "javascript" in definition2.lower()


def test_build_dictionary_minimum_structure():
    """build_dictionary returns dict of word -> {pos, def}."""
    small_vocab = ["function", "test", "code"]
    result = build_dictionary(small_vocab, max_iterations=5, verbose=False)
    assert isinstance(result, dict)
    assert len(result) >= len(small_vocab)
    for word in small_vocab:
        assert word in result, f"Missing word: {word}"
        entry = result[word]
        assert "pos" in entry and "def" in entry
        assert isinstance(entry["pos"], str)
        assert isinstance(entry["def"], str)


def test_build_dictionary_has_types():
    """Dictionary entries have sensible pos (noun/verb/adjective/etc.)."""
    small_vocab = ["write", "function", "error", "fast"]
    result = build_dictionary(small_vocab, max_iterations=5, verbose=False)
    valid_pos = {"noun", "verb", "adjective", "adverb", "unknown"}
    for word, entry in result.items():
        assert entry["pos"] in valid_pos, f"word={word} pos={entry['pos']}"


def test_save_and_load_dictionary():
    """save_dictionary writes valid JSON that can be loaded with correct structure."""
    small_vocab = ["api", "endpoint"]
    dictionary = build_dictionary(small_vocab, max_iterations=3, verbose=False)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        save_dictionary(dictionary, path=path)
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert isinstance(loaded, dict)
        for word in small_vocab:
            assert word in loaded
            assert "pos" in loaded[word] and "def" in loaded[word]
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_full_build_produces_dictionary():
    """Running build_dictionary on starter_vocab produces a non-empty dictionary."""
    # Use a small subset so test stays fast
    subset = list(starter_vocab)[:50]
    result = build_dictionary(subset, max_iterations=10, verbose=False)
    assert len(result) >= 50
    for word in subset:
        assert word in result
        assert result[word]["pos"] in ("noun", "verb", "adjective", "adverb", "unknown")
        assert len(result[word]["def"]) > 0
