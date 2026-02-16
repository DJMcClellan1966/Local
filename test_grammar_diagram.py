"""Tests for grammar_diagram: grammar rules and sentence diagramming."""

import pytest

pytest.importorskip("nltk")

from grammar_diagram import (
    get_grammar_rules,
    diagram_sentence,
    tree_to_brackets,
    tree_to_diagram,
    word_to_grammar_symbol,
    pos_label_to_grammar,
    tag_to_grammar,
    build_grammar_dictionary,
    save_grammar_dictionary,
    load_grammar_dictionary,
    GRAMMAR_RULES,
    PENN_TO_SYMBOL,
)


def test_grammar_rules_exist():
    """Phrase-structure rules are defined."""
    rules = get_grammar_rules()
    assert len(rules) >= 5
    # S -> NP VP
    assert ("S", ["NP", "VP"]) in rules
    # PP -> P NP
    assert ("PP", ["P", "NP"]) in rules


def test_diagram_sentence_returns_tree_and_string():
    """diagram_sentence returns (Node, diagram string)."""
    root, diagram = diagram_sentence("The cat sat.")
    assert root is not None
    assert root.symbol == "S"
    assert "NP" in diagram or "VP" in diagram
    assert "The" in diagram or "cat" in diagram


def test_brackets_contain_s():
    """Bracket notation contains S at top level."""
    root, _ = diagram_sentence("The cat sat on the mat.")
    brackets = tree_to_brackets(root)
    assert brackets.startswith("(S ")


def test_word_to_grammar_symbol():
    """Words get correct phrase-structure symbol from pos and closed-class lists."""
    assert word_to_grammar_symbol("the", "noun") == "Det"
    assert word_to_grammar_symbol("on", "adjective") == "P"
    assert word_to_grammar_symbol("cat", "noun") == "N"
    assert word_to_grammar_symbol("run", "verb") == "V"
    assert word_to_grammar_symbol("fast", "adverb") == "Adv"
    assert word_to_grammar_symbol("big", "adjective") == "Adj"


def test_pos_label_to_grammar():
    """pos_label_to_grammar maps pos labels using POS_TO_GRAMMAR."""
    assert pos_label_to_grammar("noun") == "N"
    assert pos_label_to_grammar("verb") == "V"
    assert pos_label_to_grammar("adjective") == "Adj"
    assert pos_label_to_grammar("adverb") == "Adv"
    assert pos_label_to_grammar("unknown") == "N"


def test_tag_to_grammar():
    """tag_to_grammar maps Penn tags using PENN_TO_SYMBOL."""
    assert tag_to_grammar("NN") == "N"
    assert tag_to_grammar("VB") == "V"
    assert tag_to_grammar("DT") == "Det"
    assert tag_to_grammar("IN") == "P"


def test_build_grammar_dictionary():
    """build_grammar_dictionary adds grammar on word, pos word, and def_parts using rules."""
    coding = {"hello": {"pos": "noun", "def": "a greeting"}, "the": {"pos": "noun", "def": "determiner"}}
    gd = build_grammar_dictionary(coding, tag_definitions=False)
    assert "hello" in gd and gd["hello"]["grammar"] == "N"
    assert gd["hello"]["pos_grammar"] == "N"
    assert gd["the"]["grammar"] == "Det"
    assert gd["hello"]["pos"] == "noun" and gd["hello"]["def"] == "a greeting"
    assert "def_parts" in gd["hello"] and gd["hello"]["def_parts"] == []
    gd2 = build_grammar_dictionary(coding, tag_definitions=True)
    assert "def_parts" in gd2["hello"]
    if gd2["hello"]["def_parts"]:
        for w, sym in gd2["hello"]["def_parts"]:
            assert sym in ("Det", "N", "V", "P", "Adj", "Adv", "Conj")
