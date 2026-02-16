# Local DictAgent – Personal Coding Assistant

A simple local agent using Ollama + custom dictionary for constrained, verifiable code generation.

## Setup

1. Install Ollama → https://ollama.com
2. Pull a small model:
ollama pull phi3:mini
ollama pull gemma3:1b   # faster alternative text 3. Generate `coding_dictionary.json` (use the builder script from earlier chats)
4. Install deps:
pip install -r requirements.txt
text5. Run:
python dict_agent.py
text## Usage

Type app requests like:
- Create a login form in React with email validation
- Build a simple Python API endpoint for user data

The agent retrieves dictionary context, fills templates, runs code, tests, and iterates until it works (or max attempts reached).

## Tips
- Add more templates in APP_TEMPLATES
- Create `tests/test_app.py` for real pytest checks
- Use lower temperature for more deterministic output

Enjoy building & learning!