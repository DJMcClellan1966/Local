"""
Local app builder: templates + dictionary-driven slot filling. No external AI.
Builds apps from user requests by matching dictionary terms, choosing a template,
and filling slots with parsed request content. Optionally use a local LLM (Ollama)
only as a refinement step from within this app.

Usage:
  python local_app_builder.py "Create a login form with email and password"
  python local_app_builder.py "Build a Python API endpoint for user data" --template python_api
  python local_app_builder.py "Make a contact form" --use-llm   # optional: refine with local Ollama
"""

import json
import os
import re
from typing import Optional

# ================== Templates (expand for flexibility) ==================
TEMPLATES = {
    "react_form": """import React, {{ useState }} from 'react';

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
""",
    "python_api": """from fastapi import FastAPI
app = FastAPI()

@app.get("/{route_path}")
def {handler_name}():
    return {{"message": "{response_message}", "data": []}}

# Run: uvicorn main:app --reload
""",
}

# Keywords (from your dictionaries) that map request intent -> template
FORM_KEYWORDS = ("form", "login", "signup", "contact", "submit", "input", "button", "field", "email", "password", "validation", "react", "component")
API_KEYWORDS = ("api", "endpoint", "rest", "get", "post", "fastapi", "express", "route", "request", "response", "python", "server")


def _load_coding_dictionary() -> dict:
    path = "coding_dictionary.json"
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def match_template(request: str, coding_dictionary: Optional[dict] = None) -> str:
    """
    Choose template from request using keyword overlap. No LLM.
    Returns template key: "react_form", "python_api", or default "react_form".
    """
    words = _extract_words(request)
    form_score = sum(1 for w in FORM_KEYWORDS if w in words)
    api_score = sum(1 for w in API_KEYWORDS if w in words)
    if api_score > form_score:
        return "python_api"
    return "react_form"


def fill_slots_react_form(request: str) -> dict:
    """
    Fill slots for react_form from request text. Rules and regex only.
    """
    words = _extract_words(request)
    # Component name
    if "login" in words or "signin" in words:
        component_name = "LoginForm"
    elif "contact" in words:
        component_name = "ContactForm"
    elif "signup" in words or "register" in words:
        component_name = "SignupForm"
    else:
        component_name = "Form"
    # Field names from common patterns
    fields_list = []
    if "email" in words:
        fields_list.append("email")
    if "password" in words:
        fields_list.append("password")
    if "name" in words:
        fields_list.append("name")
    if "message" in words:
        fields_list.append("message")
    if not fields_list:
        fields_list = ["email", "password"]
    # initial_state: { email: '', password: '' }
    initial_state = ", ".join(f"{f}: ''" for f in fields_list)
    # fields: JSX inputs
    fields_lines = []
    for f in fields_list:
        t = "password" if f == "password" else "text"
        if f == "message":
            fields_lines.append(f'      <textarea name="{f}" value={{formData.{f}}} onChange={{handleChange}} placeholder="{f}" />')
        else:
            fields_lines.append(f'      <input name="{f}" type="{t}" value={{formData.{f}}} onChange={{handleChange}} placeholder="{f}" />')
    fields = "\n".join(fields_lines)
    # validation
    if "validation" in words or "validate" in words:
        validation = "if (!formData.email && formData.email !== undefined) return;"
    else:
        validation = ""
    return {
        "component_name": component_name,
        "initial_state": initial_state,
        "fields": fields,
        "validation": validation,
    }


def fill_slots_python_api(request: str) -> dict:
    """Fill slots for python_api template."""
    words = _extract_words(request)
    route_path = "users" if "user" in words else "items"
    handler_name = "get_items" if "item" in words else "get_users"
    response_message = "User data" if "user" in words else "Data"
    return {
        "route_path": route_path,
        "handler_name": handler_name,
        "response_message": response_message,
    }


def build(
    request: str,
    template_key: Optional[str] = None,
    coding_dictionary: Optional[dict] = None,
    templates: Optional[dict] = None,
) -> str:
    """
    Build app from request: match template, fill slots, render. No LLM.
    Pass templates=load_templates_from_dir() to use custom templates from ./templates.
    """
    coding_dictionary = coding_dictionary or _load_coding_dictionary()
    tpl = templates or TEMPLATES
    if template_key is None:
        template_key = match_template(request, coding_dictionary)
    template_str = tpl.get(template_key)
    if not template_str:
        template_key = "react_form"
        template_str = tpl.get("react_form", TEMPLATES["react_form"])
    if template_key == "react_form":
        slots = fill_slots_react_form(request)
    elif template_key == "python_api":
        slots = fill_slots_python_api(request)
    else:
        slots = fill_slots_react_form(request)  # fallback: same slots for custom form-like templates
    try:
        return template_str.format(**slots)
    except KeyError:
        for k in ("component_name", "initial_state", "fields", "validation"):
            slots.setdefault(k, "")
        return template_str.format(**slots)


def build_with_llm(request: str, template_key: Optional[str] = None) -> str:
    """
    Optional: refine or generate using local Ollama only (no external agent).
    Uses dict_agent's local loop; the "agent" is created from this app + your LLM.
    """
    try:
        from dict_agent import run_agent, run_agent_with_grammar, load_dictionary
    except ImportError:
        return build(request, template_key)
    if load_dictionary():
        return run_agent_with_grammar(request, template_key or "react_form", max_attempts=2)
    return run_agent(request, template_key or "react_form", max_attempts=2)


def load_templates_from_dir(path: str = "templates") -> dict:
    """
    Load additional templates from .txt or .json files in path.
    File name (without ext) = template key. Content = template string with {slot} placeholders.
    Merges into TEMPLATES for greater flexibility.
    """
    out = dict(TEMPLATES)
    if not os.path.isdir(path):
        return out
    for name in os.listdir(path):
        if name.startswith("."):
            continue
        key = os.path.splitext(name)[0]
        full = os.path.join(path, name)
        try:
            with open(full, "r", encoding="utf-8") as f:
                out[key] = f.read()
        except Exception:
            pass
    return out


# ================== CLI ==================
if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    use_llm = "--use-llm" in args
    if "--use-llm" in args:
        args.remove("--use-llm")
    template_arg = None
    for i, a in enumerate(args):
        if a == "--template" and i + 1 < len(args):
            template_arg = args[i + 1]
            break
    request = " ".join(a for a in args if not a.startswith("--") and a != template_arg).strip()
    if not request:
        request = "Create a login form with email and password"
    tpl = load_templates_from_dir()
    if use_llm:
        print("Building with local LLM (Ollama)...")
        result = build_with_llm(request, template_arg)
    else:
        print("Building from templates + dictionary (no LLM)...")
        result = build(request, template_arg, templates=tpl)
    print("\n" + "=" * 60)
    print("Generated code:\n")
    print(result)
    print("=" * 60)
