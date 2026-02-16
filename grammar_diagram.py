"""
Sentence diagramming using normal grammar rules.
Uses phrase-structure rules (CFG) to parse a sentence and output a tree diagram.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ================== 1. Grammar rules (phrase structure) ==================
# Standard English rules: S = Sentence, NP = Noun Phrase, VP = Verb Phrase,
# PP = Prepositional Phrase, Det = Determiner, N = Noun, V = Verb, P = Preposition,
# Adj = Adjective, Adv = Adverb

GRAMMAR_RULES = [
    # Sentence
    ("S", ["NP", "VP"]),
    # Noun phrase
    ("NP", ["Det", "N"]),
    ("NP", ["Det", "Adj", "N"]),
    ("NP", ["N"]),
    ("NP", ["NP", "PP"]),
    ("NP", ["Adj", "N"]),
    # Verb phrase
    ("VP", ["V"]),
    ("VP", ["V", "NP"]),
    ("VP", ["V", "NP", "PP"]),
    ("VP", ["V", "PP"]),
    ("VP", ["VP", "PP"]),
    ("VP", ["Adv", "V"]),
    ("VP", ["V", "Adv"]),
    ("VP", ["V", "Adj"]),  # linking verb + predicate adj
    # Prepositional phrase
    ("PP", ["P", "NP"]),
]

# Map Penn Treebank POS tags (NLTK) to our grammar symbols
PENN_TO_SYMBOL = {
    "DT": "Det",   # determiner (the, a)
    "WDT": "Det",
    "PDT": "Det",
    "NN": "N", "NNS": "N", "NNP": "N", "NNPS": "N",  # noun
    "VB": "V", "VBD": "V", "VBG": "V", "VBN": "V", "VBP": "V", "VBZ": "V",  # verb
    "IN": "P", "TO": "P",  # preposition
    "JJ": "Adj", "JJR": "Adj", "JJS": "Adj",  # adjective
    "RB": "Adv", "RBR": "Adv", "RBS": "Adv", "WRB": "Adv",  # adverb
    "PRP": "N", "PRP$": "Det",  # pronoun / possessive
    "EX": "Det",  # existential there
    "CC": "Conj", "CONJ": "Conj",  # conjunction (and, or) - optional handling
}

# Map coding_dictionary pos (noun, verb, ...) to phrase-structure grammar symbol
POS_TO_GRAMMAR = {
    "noun": "N",
    "verb": "V",
    "adjective": "Adj",
    "adverb": "Adv",
    "unknown": "N",
}

# Closed-class words: grammar role overrides (identify as Det or P regardless of dict pos)
DETERMINERS = frozenset(
    "the a an this that these those every each some any no my your his her its our their "
    "one another other what which whose".split()
)
PREPOSITIONS = frozenset(
    "in on at to for of with by from as into through during before after over under "
    "between about against without per".split()
)


@dataclass
class Node:
    """A node in the parse tree: either a non-terminal (symbol + children) or terminal (word)."""
    symbol: str
    word: Optional[str] = None
    children: list[Node] = field(default_factory=list)

    def is_terminal(self) -> bool:
        return self.word is not None

    def __str__(self) -> str:
        if self.word is not None:
            return f"{self.symbol}({self.word})"
        return self.symbol


def _tagged_to_terminals(tagged: list[tuple[str, str]]) -> list[Node]:
    """Convert (word, penn_tag) list to list of terminal Nodes."""
    nodes = []
    for word, tag in tagged:
        sym = PENN_TO_SYMBOL.get(tag, "N")  # default noun for unknown
        nodes.append(Node(symbol=sym, word=word))
    return nodes


def _try_rule(rule_lhs: str, rule_rhs: list[str], nodes: list[Node], start: int) -> Optional[tuple[Node, int]]:
    """Try to match rule_rhs at nodes[start:]; return (new Node, end_index) or None."""
    if start + len(rule_rhs) > len(nodes):
        return None
    children = []
    i = start
    for sym in rule_rhs:
        n = nodes[i]
        if n.symbol != sym:
            return None
        children.append(n)
        i += 1
    return (Node(symbol=rule_lhs, children=children), i)


def _parse_span(nodes: list[Node], start: int, end: int) -> Optional[Node]:
    """
    Try to parse nodes[start:end] as a single constituent.
    Returns a single Node covering the span, or None.
    """
    length = end - start
    if length <= 0:
        return None
    if length == 1:
        return nodes[start]

    # Try each rule that could produce this span
    for rule_lhs, rule_rhs in GRAMMAR_RULES:
        if len(rule_rhs) != length:
            continue
        res = _try_rule(rule_lhs, rule_rhs, nodes, start)
        if res is not None:
            return res[0]
    return None


def _parse_sentence_cky(nodes: list[Node]) -> Optional[Node]:
    """
    Simple CKY-style: try to combine adjacent constituents with a rule.
    Build chart of (start, end) -> list of possible Nodes.
    """
    n = len(nodes)
    if n == 0:
        return None
    # chart[i][j] = list of Nodes spanning nodes[i:j]
    chart = [[[] for _ in range(n + 1)] for _ in range(n + 1)]
    for i in range(n):
        chart[i][i + 1] = [nodes[i]]

    for span in range(2, n + 1):
        for i in range(n - span + 1):
            j = i + span
            for k in range(i + 1, j):
                for left in chart[i][k]:
                    for right in chart[k][j]:
                        # Try binary rule: X -> left.symbol right.symbol
                        for rule_lhs, rule_rhs in GRAMMAR_RULES:
                            if len(rule_rhs) == 2 and rule_rhs[0] == left.symbol and rule_rhs[1] == right.symbol:
                                chart[i][j].append(Node(symbol=rule_lhs, children=[left, right]))
            # Try unary: apply rule that matches entire span
            if span >= 2:
                for rule_lhs, rule_rhs in GRAMMAR_RULES:
                    if len(rule_rhs) == span:
                        res = _try_rule(rule_lhs, rule_rhs, nodes, i)
                        if res is not None and res[1] == j:
                            chart[i][j].append(res[0])

    # Prefer S at root
    for node in chart[0][n]:
        if node.symbol == "S":
            return node
    for node in chart[0][n]:
        return node
    return None


def _parse_sentence_simple(tagged: list[tuple[str, str]]) -> Optional[Node]:
    """
    Simpler approach: assume order NP VP (subject then predicate).
    Find first verb as VP start, build NP from left, VP from verb onward, then attach PP if present.
    """
    if not tagged:
        return None
    nodes = _tagged_to_terminals(tagged)
    n = len(nodes)
    if n == 1:
        return Node(symbol="S", children=[Node(symbol=nodes[0].symbol, word=nodes[0].word)])

    # Find best verb index (first V)
    vi = None
    for i in range(n):
        if nodes[i].symbol == "V":
            vi = i
            break
    if vi is None:
        # No verb: treat whole thing as NP
        np = _combine_into_np(nodes, 0, n)
        return Node(symbol="S", children=[np]) if np else nodes[0]

    # NP = 0..vi, VP = vi..n (may contain PP)
    np = _combine_into_np(nodes, 0, vi)
    vp = _combine_into_vp(nodes, vi, n)
    if np is None:
        np = nodes[0]
    if vp is None:
        vp = Node(symbol="VP", children=nodes[vi:])
    return Node(symbol="S", children=[np, vp])


def _combine_into_np(nodes: list[Node], start: int, end: int) -> Optional[Node]:
    """Build one NP from nodes[start:end] (Det? Adj* N or N, or N PP)."""
    if start >= end:
        return None
    if start + 1 == end:
        return nodes[start]
    # Det N or Det Adj N or N
    if nodes[start].symbol == "Det" and end - start >= 2:
        if nodes[start + 1].symbol == "N":
            return Node(symbol="NP", children=[nodes[start], nodes[start + 1]])
        if end - start >= 3 and nodes[start + 1].symbol == "Adj" and nodes[start + 2].symbol == "N":
            return Node(symbol="NP", children=[nodes[start], nodes[start + 1], nodes[start + 2]])
    if nodes[start].symbol in ("N", "Adj") and end - start >= 2:
        if nodes[start + 1].symbol == "N":
            return Node(symbol="NP", children=[nodes[start], nodes[start + 1]])
    if nodes[start].symbol == "N":
        return Node(symbol="NP", children=[nodes[start]])
    return Node(symbol="NP", children=nodes[start:end])


def _combine_into_vp(nodes: list[Node], start: int, end: int) -> Optional[Node]:
    """Build VP from nodes[start:end] (V, V NP, V NP PP, etc.)."""
    if start >= end:
        return None
    if start + 1 == end:
        return Node(symbol="VP", children=[nodes[start]])
    # V NP [PP]* or V PP
    i = start
    if nodes[i].symbol != "V":
        return Node(symbol="VP", children=nodes[start:end])
    i += 1
    kids = [nodes[start]]
    while i < end:
        if nodes[i].symbol == "P" and i + 1 < end:
            # PP
            np = _combine_into_np(nodes, i + 1, end)
            if np is None and i + 2 <= end:
                np = Node(symbol="NP", children=nodes[i + 1:end])
            if np is not None:
                pp = Node(symbol="PP", children=[nodes[i], np])
                kids.append(pp)
                break
        if nodes[i].symbol in ("Det", "N", "Adj"):
            # start of NP
            j = i
            while j < end and nodes[j].symbol in ("Det", "Adj", "N"):
                j += 1
            np = _combine_into_np(nodes, i, j)
            if np is None:
                np = Node(symbol="NP", children=nodes[i:j])
            kids.append(np)
            i = j
            continue
        i += 1
    return Node(symbol="VP", children=kids)


# ================== 2. POS tagging ==================
def tag_sentence(text: str) -> list[tuple[str, str]]:
    """Tokenize and POS-tag a sentence using NLTK. Returns list of (word, penn_tag)."""
    try:
        import nltk
        from nltk import word_tokenize, pos_tag
        for resource in ("punkt_tab", "punkt", "averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
            try:
                if "punkt" in resource:
                    nltk.data.find(f"tokenizers/{resource}")
                else:
                    nltk.data.find(f"taggers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)
        tokens = word_tokenize(text)
        return pos_tag(tokens)
    except Exception as e:
        raise RuntimeError(f"POS tagging requires NLTK: pip install nltk and download punkt, averaged_perceptron_tagger. {e}") from e


# ================== 3. Diagram output ==================
def tree_to_diagram(node: Node, indent: int = 0, prefix: str = "") -> str:
    """Render parse tree as indented text diagram."""
    if node.is_terminal():
        return f"{prefix}{node.symbol}({node.word})\n"
    lines = [f"{prefix}{node.symbol}\n"]
    for i, child in enumerate(node.children):
        is_last = i == len(node.children) - 1
        branch = "  "  # 2 spaces per level
        sub_prefix = prefix + branch
        lines.append(tree_to_diagram(child, indent + 1, sub_prefix))
    return "".join(lines)


def tree_to_brackets(node: Node) -> str:
    """Render parse tree in bracket notation: (S (NP ...) (VP ...))."""
    if node.is_terminal():
        return f"({node.symbol} {node.word})"
    return "(" + node.symbol + " " + " ".join(tree_to_brackets(c) for c in node.children) + ")"


# ================== 4. Public API ==================
def diagram_sentence(text: str, style: str = "tree") -> tuple[Optional[Node], str]:
    """
    Diagram a sentence using normal grammar rules.
    Returns (parse_tree_node, diagram_string).
    style: "tree" (indented) or "brackets".
    """
    tagged = tag_sentence(text)
    if not tagged:
        return None, ""
    root = _parse_sentence_simple(tagged)
    if root is None:
        root = Node(symbol="S", children=_tagged_to_terminals(tagged))
    if style == "brackets":
        return root, tree_to_brackets(root)
    return root, tree_to_diagram(root)


def get_grammar_rules() -> list[tuple[str, list[str]]]:
    """Return the list of phrase-structure rules (lhs, rhs)."""
    return list(GRAMMAR_RULES)


# ================== 5. Grammar dictionary (words + pos + def + grammar parts, using rules) ==================
def word_to_grammar_symbol(word: str, pos: str) -> str:
    """Map a word and its pos to the phrase-structure symbol (Det, N, V, P, Adj, Adv). Uses rules: closed-class lists + POS_TO_GRAMMAR."""
    w = word.lower().strip()
    if w in DETERMINERS:
        return "Det"
    if w in PREPOSITIONS:
        return "P"
    return POS_TO_GRAMMAR.get(pos.lower().strip(), "N")


def pos_label_to_grammar(pos_label: str) -> str:
    """Map the pos label (e.g. 'noun', 'verb') to phrase-structure symbol using POS_TO_GRAMMAR."""
    return POS_TO_GRAMMAR.get(pos_label.lower().strip(), "N")


def tag_to_grammar(penn_tag: str) -> str:
    """Map Penn Treebank tag to grammar symbol using PENN_TO_SYMBOL (used for def words)."""
    return PENN_TO_SYMBOL.get(penn_tag, "N")


def build_grammar_dictionary(
    coding_dictionary: dict,
    tag_definitions: bool = True,
) -> dict[str, dict]:
    """
    Build grammar_dictionary from coding_dictionary using rules.
    Grammar is assigned to: (1) every entry word, (2) the pos word, (3) every word in def.
    Each entry: { word: { "pos", "pos_grammar", "def", "grammar", "def_parts" } }.
    - grammar: phrase-structure symbol for this word (rules: DETERMINERS/PREPOSITIONS + POS_TO_GRAMMAR).
    - pos_grammar: grammar symbol for the pos label (noun->N, verb->V, etc.) using POS_TO_GRAMMAR.
    - def_parts: list of [word, grammar] for every token in def, using NLTK tag + PENN_TO_SYMBOL.
    """
    grammar_dictionary = {}
    for word, entry in coding_dictionary.items():
        pos = entry.get("pos", "unknown")
        defn = entry.get("def", "")
        grammar = word_to_grammar_symbol(word, pos)
        pos_grammar = pos_label_to_grammar(pos)
        out = {"pos": pos, "pos_grammar": pos_grammar, "def": defn, "grammar": grammar}
        if tag_definitions and defn:
            try:
                tagged = tag_sentence(defn)
                def_parts = [
                    [w, tag_to_grammar(tag)]
                    for w, tag in tagged
                ]
                out["def_parts"] = def_parts
            except Exception:
                out["def_parts"] = []
        else:
            out["def_parts"] = []
        grammar_dictionary[word] = out
    return grammar_dictionary


def save_grammar_dictionary(
    grammar_dictionary: dict,
    path: str = "grammar_dictionary.json",
) -> None:
    """Save grammar_dictionary to JSON."""
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(grammar_dictionary, f, indent=2, ensure_ascii=False)


def load_grammar_dictionary(path: str = "grammar_dictionary.json") -> dict:
    """Load grammar_dictionary from JSON."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ================== 6. CLI ==================
if __name__ == "__main__":
    import sys
    import json

    if "--build-grammar-dict" in sys.argv:
        # Build grammar_dictionary from coding_dictionary.json
        coding_path = "coding_dictionary.json"
        tag_defs = "--no-tag-defs" not in sys.argv  # default True: grammar on all words in def
        with open(coding_path, "r", encoding="utf-8") as f:
            coding_dictionary = json.load(f)
        grammar_dictionary = build_grammar_dictionary(coding_dictionary, tag_definitions=tag_defs)
        save_grammar_dictionary(grammar_dictionary)
        print(f"Built grammar_dictionary: {len(grammar_dictionary)} entries -> grammar_dictionary.json")
        print("(grammar on: word, pos word, all words in def; using rules)")
        for i, (w, e) in enumerate(list(grammar_dictionary.items())[:4]):
            parts = len(e.get("def_parts", []))
            print(f"  {w}: grammar={e['grammar']}, pos_grammar={e.get('pos_grammar')}, def_parts={parts} tokens")
        sys.exit(0)

    rules = get_grammar_rules()
    print("Grammar rules (phrase structure):")
    for lhs, rhs in rules:
        print(f"  {lhs} -> {' '.join(rhs)}")
    print()
    example = sys.argv[1] if len(sys.argv) > 1 else "The cat sat on the mat."
    if example.startswith("-"):
        example = "The cat sat on the mat."
    print(f"Sentence: {example}")
    print()
    try:
        root, diagram = diagram_sentence(example, style="tree")
        print("Diagram (tree):")
        print(diagram)
        print("Brackets:", tree_to_brackets(root))
    except Exception as e:
        print("Error:", e)
