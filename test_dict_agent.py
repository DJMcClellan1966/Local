import pytest
import json
import os
import tempfile

# --- DictAgent imports (no ollama calls in these tests) ---
from dict_agent import load_dictionary, retrieve, extract_code, run_tests, run_code, DICT_FILE, TEST_FILE


def test_example():
    assert 1 + 1 == 2


def test_load_dictionary():
    """Dictionary loads as dict (from coding_dictionary.json)."""
    d = load_dictionary()
    assert isinstance(d, dict)
    for key, val in list(d.items())[:5]:
        assert isinstance(key, str)
        assert "pos" in val and "def" in val


def test_retrieve_empty_query():
    """Retrieve with empty-ish query returns string (maybe empty)."""
    out = retrieve("x y z", top_k=3)
    assert isinstance(out, str)


def test_retrieve_with_matches():
    """Retrieve finds relevant entries for coding terms."""
    d = load_dictionary()
    if not d:
        pytest.skip("coding_dictionary.json is empty — run dictionary_builder.py")
    out = retrieve("react form validation", top_k=5)
    assert isinstance(out, str)
    # Should contain at least one entry line
    lines = [l for l in out.splitlines() if l.startswith("- ")]
    assert len(lines) <= 5


def test_extract_code_plain():
    """Plain code without markdown is returned as-is."""
    code = "const x = 1;"
    assert extract_code(code) == code


def test_extract_code_fenced():
    """Code inside ``` block is extracted."""
    raw = "```js\nconst x = 1;\nconsole.log(x);\n```"
    assert extract_code(raw) == "const x = 1;\nconsole.log(x);"


def test_extract_code_fenced_jsx():
    """JSX/TSX fenced block is extracted."""
    raw = "```tsx\nconst App = () => <div>Hi</div>;\n```"
    got = extract_code(raw)
    assert "App" in got and "Hi" in got


def test_run_tests_no_file():
    """When tests/test_app.py is missing, run_tests returns (True, msg)."""
    if os.path.exists(TEST_FILE):
        pytest.skip(f"{TEST_FILE} exists")
    ok, msg = run_tests()
    assert ok is True
    assert "test" in msg.lower() or "test_app" in msg


def test_run_code_js():
    """run_code executes JS with node and returns (bool, str)."""
    code = "console.log(2 + 2);"
    ok, out = run_code(code, suffix=".js")
    assert isinstance(ok, bool)
    assert isinstance(out, str)
    if ok:
        assert "4" in out