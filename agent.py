#!/usr/bin/env python3
"""
Termux AI Agent - Agentic coding assistant for Termux
Uses OpenRouter free models with comprehensive tooling
"""

import os
import sys
import json
import subprocess
import time
import re
import glob as globlib
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import quote_plus
from html.parser import HTMLParser

try:
    from openai import OpenAI
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "openai", "-q"])
    from openai import OpenAI

# Optional deps - install if missing
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Installing web dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "requests", "beautifulsoup4", "lxml", "-q"])
    import requests
    from bs4 import BeautifulSoup

# =============================================================================
# CONFIG
# =============================================================================

env_file = Path.home() / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"\''))

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = os.environ.get("AGENT_MODEL", "kwaipilot/kat-coder-pro:free")

MODELS = {
    "1": ("kwaipilot/kat-coder-pro:free", "KAT-Coder-Pro V1"),
    "2": ("mistralai/devstral-2512:free", "Devstral 2 2512"),
    "3": ("tngtech/deepseek-r1t2-chimera:free", "DeepSeek R1T2 Chimera"),
    "4": ("deepseek/deepseek-r1-0528:free", "DeepSeek R1 0528"),
    "5": ("deepseek/deepseek-chat-v3-0324:free", "DeepSeek Chat V3"),
    "6": ("x-ai/grok-4-fast:free", "Grok 4 Fast"),
}

SYSTEM_PROMPT = """You are an AI coding assistant in Termux. You have these tools:

FILE OPERATIONS:
- read_file(path) - read file contents
- write_file(path, content) - create/overwrite file
- edit_file(path, old, new) - find and replace text
- list_dir(path) - list directory
- find_files(pattern, path?) - glob search for files

SEARCH:
- grep(pattern, path?) - search file contents with regex
- search_code(query, path?) - search for code patterns

SHELL:
- run(cmd) - execute shell command
- git(args) - run git command

WEB:
- web_fetch(url) - fetch URL as markdown
- web_search(query) - search the web

Respond with JSON to use tools:
{"tool": "read_file", "path": "file.py"}
{"tool": "write_file", "path": "file.py", "content": "..."}
{"tool": "edit_file", "path": "file.py", "old": "old text", "new": "new text"}
{"tool": "list_dir", "path": "."}
{"tool": "find_files", "pattern": "*.py", "path": "."}
{"tool": "grep", "pattern": "def main", "path": "."}
{"tool": "search_code", "query": "function that handles auth"}
{"tool": "run", "cmd": "npm install"}
{"tool": "git", "args": "status"}
{"tool": "web_fetch", "url": "https://example.com"}
{"tool": "web_search", "query": "python requests tutorial"}
{"tool": "done", "message": "Task completed"}

For regular chat, respond normally without JSON.
Chain tools as needed - read before edit, search before modify.
Always explain what you're doing."""

# =============================================================================
# TOOLS
# =============================================================================

def read_file(path):
    """Read file contents"""
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: File not found: {path}"
        if p.stat().st_size > 100000:
            return p.read_text()[:100000] + "\n... (truncated)"
        return p.read_text()
    except Exception as e:
        return f"Error: {e}"

def write_file(path, content):
    """Write content to file"""
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Written {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

def edit_file(path, old, new):
    """Find and replace in file"""
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: File not found: {path}"
        content = p.read_text()
        if old not in content:
            return f"Error: Text not found in {path}"
        count = content.count(old)
        updated = content.replace(old, new)
        p.write_text(updated)
        return f"Replaced {count} occurrence(s) in {path}"
    except Exception as e:
        return f"Error: {e}"

def list_dir(path="."):
    """List directory contents"""
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: Directory not found: {path}"
        items = []
        for item in sorted(p.iterdir()):
            prefix = "📁 " if item.is_dir() else "📄 "
            items.append(f"{prefix}{item.name}")
        return "\n".join(items) if items else "(empty)"
    except Exception as e:
        return f"Error: {e}"

def find_files(pattern, path="."):
    """Find files matching glob pattern"""
    try:
        p = Path(path).expanduser()
        matches = list(p.glob(pattern))
        if not matches:
            matches = list(p.rglob(pattern))  # recursive fallback
        results = [str(m.relative_to(p) if m.is_relative_to(p) else m) for m in matches[:50]]
        return "\n".join(results) if results else "No matches found"
    except Exception as e:
        return f"Error: {e}"

def grep(pattern, path="."):
    """Search file contents with regex"""
    try:
        p = Path(path).expanduser()
        regex = re.compile(pattern, re.IGNORECASE)
        results = []

        files = p.rglob("*") if p.is_dir() else [p]
        for f in files:
            if not f.is_file():
                continue
            if f.suffix in ['.pyc', '.so', '.o', '.a', '.bin', '.exe']:
                continue
            if any(part.startswith('.') for part in f.parts):
                continue
            try:
                content = f.read_text(errors='ignore')
                for i, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        results.append(f"{f}:{i}: {line[:100]}")
                        if len(results) >= 30:
                            return "\n".join(results) + "\n... (more results)"
            except:
                continue
        return "\n".join(results) if results else "No matches found"
    except Exception as e:
        return f"Error: {e}"

def search_code(query, path="."):
    """Smart code search - searches for functions, classes, patterns"""
    patterns = [
        rf"def\s+\w*{re.escape(query)}\w*",  # functions
        rf"class\s+\w*{re.escape(query)}\w*",  # classes
        rf"async\s+def\s+\w*{re.escape(query)}\w*",  # async functions
        rf"{re.escape(query)}",  # literal
    ]
    results = []
    for pat in patterns:
        r = grep(pat, path)
        if r and "No matches" not in r:
            results.append(r)
    return "\n---\n".join(results) if results else "No matches found"

def run(cmd):
    """Run shell command"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=120
        )
        output = result.stdout + result.stderr
        return output[:8000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out (120s)"
    except Exception as e:
        return f"Error: {e}"

def git(args):
    """Run git command"""
    return run(f"git {args}")

def web_fetch(url):
    """Fetch URL and convert to readable text"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; TermuxAgent/1.0)'}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'lxml')

        # Remove scripts, styles, nav, footer
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()

        # Get text
        text = soup.get_text(separator='\n', strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)

        return text[:15000] if len(text) > 15000 else text
    except Exception as e:
        return f"Error fetching {url}: {e}"

def web_search(query):
    """Search the web using DuckDuckGo"""
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; TermuxAgent/1.0)'}
        resp = requests.get(url, headers=headers, timeout=15)

        soup = BeautifulSoup(resp.text, 'lxml')
        results = []

        for result in soup.select('.result')[:8]:
            title_el = result.select_one('.result__title')
            snippet_el = result.select_one('.result__snippet')
            link_el = result.select_one('.result__url')

            if title_el:
                title = title_el.get_text(strip=True)
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                link = link_el.get_text(strip=True) if link_el else ""
                results.append(f"**{title}**\n{link}\n{snippet}\n")

        return "\n".join(results) if results else "No results found"
    except Exception as e:
        return f"Error searching: {e}"

# =============================================================================
# TOOL DISPATCH
# =============================================================================

TOOLS = {
    "read_file": lambda p: read_file(p.get("path", "")),
    "write_file": lambda p: write_file(p.get("path", ""), p.get("content", "")),
    "edit_file": lambda p: edit_file(p.get("path", ""), p.get("old", ""), p.get("new", "")),
    "list_dir": lambda p: list_dir(p.get("path", ".")),
    "find_files": lambda p: find_files(p.get("pattern", "*"), p.get("path", ".")),
    "grep": lambda p: grep(p.get("pattern", ""), p.get("path", ".")),
    "search_code": lambda p: search_code(p.get("query", ""), p.get("path", ".")),
    "run": lambda p: run(p.get("cmd", "")),
    "git": lambda p: git(p.get("args", "")),
    "web_fetch": lambda p: web_fetch(p.get("url", "")),
    "web_search": lambda p: web_search(p.get("query", "")),
    "done": lambda p: None,
}

def execute_tool(tool_call):
    """Execute a tool call"""
    tool = tool_call.get("tool", "")
    if tool in TOOLS:
        return TOOLS[tool](tool_call)
    return f"Unknown tool: {tool}"

def parse_tool_call(text):
    """Extract JSON tool call from response"""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except:
        pass
    return None

# =============================================================================
# UI
# =============================================================================

def show_models():
    print("\n\033[96m=== Models ===\033[0m")
    for k, (mid, name) in MODELS.items():
        print(f"  {k}) {name}")
    print()

def select_model():
    global MODEL
    show_models()
    choice = input("Select (1-6) or model ID: ").strip()
    if choice in MODELS:
        MODEL = MODELS[choice][0]
        print(f"\033[92m→ {MODELS[choice][1]}\033[0m\n")
    elif choice:
        MODEL = choice
        print(f"\033[92m→ {MODEL}\033[0m\n")

# =============================================================================
# MAIN
# =============================================================================

def main():
    global MODEL

    if not OPENROUTER_API_KEY:
        print("\033[91mMissing OPENROUTER_API_KEY\033[0m")
        print("Get free key: https://openrouter.ai/keys")
        print("Then: export OPENROUTER_API_KEY=sk-or-...")
        print("Or:   echo 'OPENROUTER_API_KEY=sk-or-...' > ~/.env")
        sys.exit(1)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Get model display name
    model_name = MODEL
    for k, (mid, name) in MODELS.items():
        if mid == MODEL:
            model_name = name
            break

    print(f"\033[96m{'═'*50}\033[0m")
    print(f"\033[96m  Termux AI Agent\033[0m")
    print(f"\033[96m{'═'*50}\033[0m")
    print(f"Model: \033[93m{model_name}\033[0m")
    print(f"Dir:   \033[90m{os.getcwd()}\033[0m")
    print(f"\nTools: read, write, edit, grep, find, run, git, web")
    print(f"Cmds:  /model /clear /quit")
    print()

    while True:
        try:
            user_input = input("\033[92mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["/quit", "/q", "quit", "exit"]:
            break
        if user_input.lower() in ["/clear", "/c"]:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Cleared.\n")
            continue
        if user_input.lower() in ["/model", "/m"]:
            select_model()
            continue
        if user_input.lower() in ["/models"]:
            show_models()
            continue

        messages.append({"role": "user", "content": user_input})

        # Agent loop with retry
        while True:
            reply = None
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        max_tokens=4096,
                    )
                    reply = response.choices[0].message.content
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"\033[93m[retry {attempt+1}]\033[0m ", end="", flush=True)
                        time.sleep(2)
                    else:
                        print(f"\033[91mError: {e}\033[0m\n")

            if not reply:
                break

            messages.append({"role": "assistant", "content": reply})
            tool_call = parse_tool_call(reply)

            if tool_call:
                tool_name = tool_call.get("tool", "")
                print(f"\033[93m[{tool_name}]\033[0m ", end="")

                if tool_name == "done":
                    print(tool_call.get("message", "Done"))
                    break

                result = execute_tool(tool_call)
                if result:
                    display = result[:300] + "..." if len(str(result)) > 300 else result
                    print(display)
                    messages.append({"role": "user", "content": f"Tool result:\n{result}"})
                else:
                    break
            else:
                print(f"\033[94mAgent:\033[0m {reply}\n")
                break

if __name__ == "__main__":
    main()
