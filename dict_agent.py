import json
import ollama
import re
import subprocess
import tempfile
import os
from typing import List, Dict, Tuple

# ================= CONFIG =================
MODEL = "phi3:mini"           # Change to "gemma3:1b" for faster CPU, or "qwen3:4b" for better code
DICT_FILE = "coding_dictionary.json"
TEST_FILE = "tests/test_app.py"  # Create this file if you want auto-testing

# Example templates (expand as needed)
APP_TEMPLATES = {
    "react_form": """
import React, { useState } from 'react';

function {component_name}({{ onSubmit }}) {{
  const [formData, setFormData] = useState({{ {initial_state} }});

  const handleChange = (e) => {{
    setFormData({{ ...formData, [e.target.name]: e.target.value }});
  }};

  const handleSubmit = (e) => {{
    e.preventDefault();
    {validation}
    onSubmit(formData);
  }};

  return (
    <form onSubmit={{handleSubmit}}>
      {fields}
      <button type="submit">Submit</button>
    </form>
  );
}}

export default {component_name};
"""
    # Add more templates here, e.g. "nextjs_page", "python_api", etc.
}

# ================= Load & Retrieve =================
def load_dictionary() -> Dict:
    try:
        with open(DICT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {DICT_FILE} not found.")
        return {}
    except Exception as e:
        print(f"Dictionary load error: {e}")
        return {}

dictionary = load_dictionary()

def retrieve(query: str, top_k: int = 10) -> str:
    query_lower = query.lower()
    matches = sorted(
        [(sum(1 for w in re.findall(r'[a-z]+', query_lower) if w in word or w in entry['def'].lower()), word, entry)
         for word, entry in dictionary.items()],
        key=lambda x: x[0], reverse=True
    )[:top_k]
    return "\n".join(f"- {w} ({e['pos']}): {e['def']}" for _, w, e in matches)

# ================= Tools =================
def run_code(code: str, suffix: str = '.js') -> Tuple[bool, str]:
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(code)
        temp_path = f.name
    try:
        # For JS/React: use node (change command for Python, etc.)
        result = subprocess.run(['node', temp_path], capture_output=True, text=True, timeout=10)
        os.unlink(temp_path)
        return result.returncode == 0, result.stdout + "\n" + result.stderr
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return False, str(e)

def run_tests() -> Tuple[bool, str]:
    try:
        result = subprocess.run(['pytest', TEST_FILE, '-q'], capture_output=True, text=True, timeout=15)
        return result.returncode == 0, result.stdout + "\n" + result.stderr
    except Exception as e:
        return False, str(e)

# ================= Simple LLM Judge =================
def judge_output(generated: str, request: str) -> float:
    prompt = f"""Rate 0–100 how well this code matches the request (focus on correctness, completeness, executability).
Request: {request}
Code:
{generated}

Output ONLY a number 0-100."""
    try:
        resp = ollama.generate(model="gemma3:1b", prompt=prompt)  # use tiny model as judge
        return float(resp['response'].strip())
    except:
        return 0.0

# ================= Agent Loop =================
def run_agent(request: str, template_key: str = "react_form", max_attempts: int = 3) -> str:
    print(f"Starting agent for: {request}")
    context = retrieve(request)
    template = APP_TEMPLATES.get(template_key, "No template found.")

    current_code = ""
    for attempt in range(1, max_attempts + 1):
        prompt = f"""You are a strict, rule-following coding agent using ONLY my dictionary context.
Dictionary entries:
{context}

Template:
{template}

User request: {request}

Think step-by-step:
1. Parse request using dictionary terms.
2. Fill template exactly.
3. Make code clean and complete.

Previous attempt (if any):
{current_code}

Output ONLY the full code — no explanations.

If previous had errors, fix them now."""

        response = ollama.generate(model=MODEL, prompt=prompt, options={"temperature": 0.25})
        current_code = response['response'].strip()

        print(f"Attempt {attempt}: generated code length {len(current_code)}")

        # Quick verification
        ok, output = run_code(current_code)
        tests_ok, test_msg = run_tests()
        score = judge_output(current_code, request)

        print(f"  Exec: {'OK' if ok else 'FAIL'} | Tests: {'PASS' if tests_ok else 'FAIL'} | Judge: {score:.0f}/100")
        print(f"  Output snippet: {output[:200]}...")

        if ok and tests_ok and score >= 85:
            print("Success!")
            return current_code

    print("Max attempts reached. Returning best effort.")
    return current_code

# ================= CLI Loop =================
if __name__ == "__main__":
    print("DictAgent ready. Enter requests below (type 'quit' to exit)\n")
    while True:
        user_input = input("Request: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        if not user_input:
            continue

        result = run_agent(user_input)
        print("\n" + "="*60)
        print("Generated Code:\n")
        print(result)
        print("="*60 + "\n")