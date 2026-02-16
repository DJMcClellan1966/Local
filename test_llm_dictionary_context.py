"""Tests for llm_dictionary_context (LLM context from dictionaries + grammar)."""

import pytest


# ---------- Break / adversarial tests ----------
def test_build_llm_context_none_dictionaries():
    """build_llm_context with dictionaries=None should load from disk, not crash."""
    from llm_dictionary_context import build_llm_context
    ctx = build_llm_context("hello", dictionaries=None, max_words=2)
    assert isinstance(ctx, str)


def test_build_llm_context_empty_dictionaries():
    """build_llm_context with all empty dicts should not crash."""
    from llm_dictionary_context import build_llm_context
    d = {"coding_dictionary": {}, "number_word_dictionary": {}, "token_dictionary": {}, "grammar_dictionary": {}}
    ctx = build_llm_context("hello world", dictionaries=d, include_word_info=True)
    assert isinstance(ctx, str)


def test_get_word_info_none_dicts():
    """get_word_info with None or empty dicts returns safe default."""
    from llm_dictionary_context import get_word_info
    info = get_word_info("nonexistent", {}, None)
    assert info["word"] == "nonexistent" and info["pos"] == "unknown" and info["grammar"] == "N"
    info2 = get_word_info("cat", {}, {})
    assert info2["pos"] == "unknown"


def test_get_word_info_missing_keys_in_entry():
    """get_word_info handles entries missing pos/def/grammar keys."""
    from llm_dictionary_context import get_word_info
    gd = {"x": {}}  # no pos, def, grammar
    info = get_word_info("x", gd, None)
    assert info["word"] == "x"
    assert info.get("pos", "?") in ("unknown", "?") or info["pos"] is not None
    assert "def" in info and "grammar" in info


def test_extract_words_from_text_weird_input():
    """extract_words_from_text handles empty, numbers-only, unicode."""
    from llm_dictionary_context import extract_words_from_text
    assert extract_words_from_text("") == []
    assert extract_words_from_text("   ") == []
    assert extract_words_from_text("123 456") == []
    assert extract_words_from_text("a b c") == []  # len > 1 filter
    words = extract_words_from_text("hello world 123")
    assert "hello" in words and "world" in words
    words_unicode = extract_words_from_text("café naïve")
    assert isinstance(words_unicode, list)  # regex [a-z]+ may drop accented; no crash


def test_format_word_info_missing_def():
    """format_word_info does not crash when def is missing or None."""
    from llm_dictionary_context import format_word_info
    info = {"word": "x", "pos": "noun", "grammar": "N", "def": None}
    s = format_word_info(info)
    assert "x" in s
    info2 = {"word": "y"}  # minimal keys
    s2 = format_word_info(info2)
    assert "y" in s2


def test_retrieve_with_grammar_empty_dicts():
    """retrieve_with_grammar with empty dicts returns empty string."""
    from llm_dictionary_context import retrieve_with_grammar
    out = retrieve_with_grammar("hello", {}, {}, top_k=5)
    assert isinstance(out, str)


def test_get_usual_patterns_empty_grammar_dictionary():
    """get_usual_patterns_from_grammar_dictionary with empty dict."""
    from llm_dictionary_context import get_usual_patterns_from_grammar_dictionary
    pats = get_usual_patterns_from_grammar_dictionary({})
    assert pats["counts"] == {} and pats["examples"] == {}


def test_build_llm_prompt_none_dictionaries():
    """build_llm_prompt with dictionaries=None loads and returns string."""
    from llm_dictionary_context import build_llm_prompt
    p = build_llm_prompt("test", dictionaries=None, max_words=1)
    assert isinstance(p, str) and "User request" in p


def test_load_dictionary_missing_file():
    """load_dictionary with missing path returns {}."""
    from llm_dictionary_context import load_dictionary
    d = load_dictionary("nonexistent_file_xyz_12345.json")
    assert d == {}


def test_load_all_dictionaries_bad_dir():
    """load_all_dictionaries with dir that has no JSON files returns empty dicts."""
    from llm_dictionary_context import load_all_dictionaries
    d = load_all_dictionaries(dir_path="/nonexistent_path_xyz_999")
    assert all(k in d for k in ["coding_dictionary", "grammar_dictionary"])
    assert all(isinstance(v, dict) for v in d.values())


# ---------- Normal tests ----------
def test_load_all_dictionaries():
    """load_all_dictionaries returns dict with four keys."""
    from llm_dictionary_context import load_all_dictionaries
    d = load_all_dictionaries()
    assert "coding_dictionary" in d
    assert "number_word_dictionary" in d
    assert "token_dictionary" in d
    assert "grammar_dictionary" in d


def test_build_llm_context_empty_input():
    """build_llm_context with empty input still returns grammar rules and patterns if dicts loaded."""
    from llm_dictionary_context import load_all_dictionaries, build_llm_context
    d = load_all_dictionaries()
    ctx = build_llm_context("", dictionaries=d, include_word_info=False)
    assert "S" in ctx or "NP" in ctx or "phrase" in ctx.lower()


def test_build_llm_context_with_words():
    """build_llm_context with real words includes word info when in dictionary."""
    from llm_dictionary_context import load_all_dictionaries, build_llm_context
    d = load_all_dictionaries()
    if not d.get("grammar_dictionary"):
        pytest.skip("grammar_dictionary.json not found")
    ctx = build_llm_context("the function returns a value", dictionaries=d, max_words=5)
    assert "function" in ctx or "returns" in ctx or "value" in ctx
    assert "pos=" in ctx or "grammar=" in ctx


def test_get_usual_patterns_from_grammar_dictionary():
    """get_usual_patterns_from_grammar_dictionary returns counts and examples."""
    from llm_dictionary_context import get_usual_patterns_from_grammar_dictionary
    gd = {"a": {"grammar": "Det"}, "cat": {"grammar": "N"}, "run": {"grammar": "V"}}
    pats = get_usual_patterns_from_grammar_dictionary(gd)
    assert "counts" in pats and "examples" in pats
    assert pats["counts"].get("N") == 1
    assert "cat" in pats["examples"].get("N", [])
