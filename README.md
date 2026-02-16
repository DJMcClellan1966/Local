# Local DictAgent – Personal Coding Assistant

A simple local agent using Ollama + custom dictionary for constrained, verifiable code generation.

## Setup

1. Install Ollama → https://ollama.com
2. Pull a small model:
   ```bash
   ollama pull phi3:mini
   ollama pull gemma3:1b   # faster alternative
   ```
3. Generate dictionaries (run once). **Dictionary from common user inputs** (recommended for code/app prompts):
   ```bash
   python vocab_from_prompts.py                    # builds common_coding_vocab.json from common_prompts.txt
   python dictionary_token_builder.py             # then build all 4 dicts (coding dict uses that vocab)
   ```
   Edit `common_prompts.txt` (one prompt per line) to add your own typical requests; re-run `vocab_from_prompts.py` and then the token builder to refresh. If `common_coding_vocab.json` is missing, the dictionary falls back to a built-in curated list.
   Or build all 4 without prompt-derived vocab: `python dictionary_token_builder.py` (uses curated list).
   Options: `--use-existing-coding` (reuse coding_dictionary.json), `--no-grammar`, `--no-tag-defs`.
4. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
5. Run:
   ```bash
   python dict_agent.py
   ```

## Usage

Type app requests like:
- Create a login form in React with email validation
- Build a simple Python API endpoint for user data

The agent retrieves dictionary context, fills templates, runs code, tests, and iterates until it works (or max attempts reached).

## Build apps from templates + dictionary (no external AI)

**All local, no third-party agent.** Build apps using templates and dictionary-driven slot filling. No LLM required.

```bash
python local_app_builder.py "Create a login form with email and password"
python local_app_builder.py "Build a Python API endpoint for user data" --template python_api
```

- **Templates**: `react_form`, `python_api` (and any you add under `templates/`).
- **Slot filling**: Request text is parsed with keyword rules (login → LoginForm, email/password → fields); uses your dictionary terms for template matching.
- **Optional local LLM**: Only if you choose it, a small loop uses **your** local Ollama + dictionaries to refine output. The “agent” is created from this app and your LLM—no external agent service.

```bash
python local_app_builder.py "Make a contact form" --use-llm   # optional: refine with local Ollama
```

Add custom templates: put files in `templates/` (e.g. `templates/my_form.txt` with `{component_name}`, `{fields}`, etc.); they are loaded automatically.

## Sentence diagramming (grammar rules)

`grammar_diagram.py` diagrams sentences using normal phrase-structure rules (S → NP VP, NP → Det N, VP → V NP, PP → P NP, etc.):

```bash
python grammar_diagram.py "The cat sat on the mat."
```

**Grammar dictionary** (grammar on all words, pos word, and all words in def, using rules):

```bash
python grammar_diagram.py --build-grammar-dict
```

Produces `grammar_dictionary.json`. For each entry:
- **grammar**: phrase-structure symbol for the **word** (Det, N, V, P, Adj, Adv) using rules: DETERMINERS/PREPOSITIONS + POS_TO_GRAMMAR.
- **pos_grammar**: grammar symbol for the **pos label** (e.g. noun→N, verb→V) using POS_TO_GRAMMAR.
- **def_parts**: list of `[word, grammar]` for **every token in the definition**, using NLTK tag + PENN_TO_SYMBOL.

Use `--no-tag-defs` to skip NLTK tagging of definitions (faster; `def_parts` will be empty).

**Build all 4 dictionaries** (coding, number_word, token, grammar) in one run:

```bash
python dictionary_token_builder.py
python dictionary_token_builder.py --use-existing-coding   # reuse coding_dictionary.json
python dictionary_token_builder.py --no-grammar            # only coding, number_word, token
```

Requires NLTK (`pip install nltk`; first run may download punkt and tagger data).

## LLM with pos, def, grammar rules and sentence patterns

The LLM can use the dictionaries and grammar to **see pos and def per word**, **phrase-structure rules**, and **usual sentence/POS patterns**, so it learns better patterns and produces more consistent text.

**`llm_dictionary_context.py`** builds context for any LLM prompt:

- **Grammar rules** – S → NP VP, NP → Det N, VP → V NP, etc.
- **Usual patterns** – grammar symbol counts (e.g. N=6806, V=1829), example words by grammar, and example sentence structures (diagrammed).
- **Word info** – for each word in the request: pos, def, grammar symbol.

Use it standalone or in the agent:

```python
from llm_dictionary_context import load_all_dictionaries, build_llm_context, build_llm_prompt
d = load_all_dictionaries()
context = build_llm_context("Create a login form with validation", dictionaries=d)
prompt = build_llm_prompt("Add a submit button", dictionaries=d)  # full prompt for your LLM
# Then call ollama.generate(prompt=prompt) or any LLM API with prompt
```

**Grammar-aware agent** (same flow as dict_agent, but injects grammar + patterns + word pos/def/grammar into the prompt):

```bash
python dict_agent.py --grammar
```

Or in code: `run_agent_with_grammar("Create a form")` instead of `run_agent(...)`.

## Tips
- Add more templates in APP_TEMPLATES
- Create `tests/test_app.py` for real pytest checks
- Use lower temperature for more deterministic output

Enjoy building & learning!