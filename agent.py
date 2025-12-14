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

# Session/Memory paths
AGENT_DIR = Path.home() / ".termux-agent"
SESSION_FILE = AGENT_DIR / "session.json"
MEMORY_FILE = AGENT_DIR / "memory.json"

# Compact threshold (% of context window)
COMPACT_THRESHOLD = 0.70  # Compact at 70% full
KEEP_RECENT_PCT = 0.25    # Keep 25% for recent messages

# Prevent infinite tool loops
MAX_TOOL_CALLS_PER_TURN = 10

# Model ID, Display Name, Context Window (tokens)
# Pricing verified Dec 2025
MODELS = {
    # Free tier - $0 input/output
    "1": ("kwaipilot/kat-coder-pro:free", "KAT-Coder-Pro V1 [FREE]", 262144),
    "2": ("mistralai/devstral-2512:free", "Devstral 2 2512 123B [FREE]", 262144),
    "3": ("tngtech/deepseek-r1t2-chimera:free", "DeepSeek R1T2 Chimera 671B [FREE]", 163840),
    # Cheap paid - reasoning model
    "4": ("deepseek/deepseek-v3.2-speciale", "DeepSeek V3.2 Speciale [$0.27/$0.41]", 163840),
}

def get_context_window():
    """Get context window for current model"""
    for k, (mid, name, ctx) in MODELS.items():
        if mid == MODEL:
            return ctx
    return 32768  # Conservative default

def estimate_tokens(text):
    """Rough token estimate (~4 chars per token)"""
    return len(text) // 4

def estimate_messages_tokens(messages):
    """Estimate total tokens in message list"""
    total = 0
    for m in messages:
        total += estimate_tokens(m.get("content", ""))
        total += 4  # Role overhead
    return total

SYSTEM_PROMPT = """You are a senior software engineer operating as an autonomous coding agent in Termux on Android. You excel at understanding codebases, planning implementations, and executing precise changes.

## Core Principles

1. **Understand Before Acting**: Always read relevant code before modifying it. Use `project_context` for new codebases, `grep`/`search_code` to locate specific patterns.

2. **Plan Complex Tasks**: For multi-step work, use `plan_task` to create a structured plan. Mark steps complete with `complete_step` as you progress. This keeps you organized and shows progress.

3. **Minimal, Precise Changes**: Make the smallest edit that solves the problem. Don't refactor unrelated code. Don't add features that weren't requested.

4. **Verify Your Work**: After making changes, verify they work (run tests, check syntax, confirm the fix). Don't assume success.

5. **Efficient Tool Use**:
   - Use `batch_edit` for multiple related changes instead of sequential `edit_file` calls
   - Use `project_context` once to understand structure, not repeatedly
   - Limit web searches to 2-3 per query - synthesize results, don't over-research
   - Use `remember` to store key findings you'll need later

## Tool Selection Guide

**Understanding Code:**
- `project_context` → Get project structure + config files (do this FIRST for new projects)
- `read_file` → Read specific file contents
- `grep` / `search_code` → Find patterns, functions, classes
- `find_files` → Locate files by name/pattern

**Modifying Code:**
- `edit_file` → Single find-and-replace edit
- `batch_edit` → Multiple edits across files (prefer this for refactoring)
- `write_file` → Create new files or complete rewrites

**Execution:**
- `run` → Execute shell commands (build, test, run scripts)
- `git` → Version control operations

**Research:**
- `web_search` → Find documentation, solutions, examples (limit: 2-3 searches)
- `web_fetch` → Read specific URLs

**Organization:**
- `plan_task` → Create/view step-by-step plans
- `complete_step` → Mark plan steps done
- `remember`/`recall` → Persistent memory across sessions

## Response Style

- Be concise and direct
- Explain your reasoning briefly before acting
- When showing code changes, explain what changed and why
- If something fails, diagnose and retry with a different approach
- Ask clarifying questions only when truly ambiguous

## Constraints

- Never make destructive changes without explicit confirmation
- Never commit to git unless asked
- Never expose secrets or credentials
- Stay focused on the requested task - don't scope creep

You have a 256k token context window. Use it wisely - load relevant context, but don't dump entire codebases unnecessarily."""

# Native OpenAI-style tool definitions
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with content",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write"},
                    "content": {"type": "string", "description": "Content to write to the file"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Find and replace text in a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit"},
                    "old": {"type": "string", "description": "Text to find"},
                    "new": {"type": "string", "description": "Text to replace with"}
                },
                "required": ["path", "old", "new"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List contents of a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: current dir)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "Find files matching a glob pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern like *.py or **/*.js"},
                    "path": {"type": "string", "description": "Directory to search in (default: current dir)"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents with regex pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "File or directory to search (default: current dir)"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Smart code search for functions, classes, and patterns",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for (function name, class, etc)"},
                    "path": {"type": "string", "description": "Directory to search (default: current dir)"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command to execute"}
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git",
            "description": "Run a git command",
            "parameters": {
                "type": "object",
                "properties": {
                    "args": {"type": "string", "description": "Git arguments (e.g. 'status', 'add .', 'commit -m msg')"}
                },
                "required": ["args"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a URL and extract readable text content",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save a key-value pair to persistent memory (survives restarts)",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Memory key"},
                    "value": {"type": "string", "description": "Value to remember"}
                },
                "required": ["key", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Retrieve from persistent memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Memory key (omit to get all memories)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": "Delete a key from persistent memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Memory key to delete"}
                },
                "required": ["key"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "batch_edit",
            "description": "Apply multiple find-and-replace edits across multiple files in one operation",
            "parameters": {
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "description": "List of edits to apply",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "File path"},
                                "old": {"type": "string", "description": "Text to find"},
                                "new": {"type": "string", "description": "Replacement text"}
                            },
                            "required": ["path", "old", "new"]
                        }
                    }
                },
                "required": ["edits"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "project_context",
            "description": "Load project structure and key config files (package.json, README, etc.) to understand the codebase",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Project root directory (default: current dir)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plan_task",
            "description": "Create a step-by-step plan for a complex task, or view the current plan",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Description of the task to plan"},
                    "steps": {
                        "type": "array",
                        "description": "List of steps to complete the task (omit to view current plan)",
                        "items": {"type": "string"}
                    }
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "complete_step",
            "description": "Mark a step in the current plan as completed",
            "parameters": {
                "type": "object",
                "properties": {
                    "step_num": {"type": "integer", "description": "Step number to mark complete (1-indexed)"}
                },
                "required": ["step_num"]
            }
        }
    }
]

# =============================================================================
# SESSION & MEMORY
# =============================================================================

def ensure_agent_dir():
    """Create agent data directory"""
    AGENT_DIR.mkdir(parents=True, exist_ok=True)

def save_session(messages):
    """Save conversation to disk"""
    ensure_agent_dir()
    # Don't save system prompt
    to_save = [m for m in messages if m.get("role") != "system"]
    SESSION_FILE.write_text(json.dumps(to_save, indent=2))

def load_session():
    """Load previous session"""
    if SESSION_FILE.exists():
        try:
            return json.loads(SESSION_FILE.read_text())
        except:
            pass
    return []

def clear_session():
    """Delete saved session"""
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()

def save_memory(key, value):
    """Save persistent memory item"""
    ensure_agent_dir()
    memory = load_memory()
    memory[key] = value
    MEMORY_FILE.write_text(json.dumps(memory, indent=2))
    return f"Saved to memory: {key}"

def load_memory():
    """Load all memory"""
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except:
            pass
    return {}

def get_memory(key=None):
    """Get memory item or all memory"""
    memory = load_memory()
    if key:
        return memory.get(key, f"No memory for: {key}")
    return json.dumps(memory, indent=2) if memory else "(no memories)"

def forget_memory(key):
    """Delete memory item"""
    memory = load_memory()
    if key in memory:
        del memory[key]
        MEMORY_FILE.write_text(json.dumps(memory, indent=2))
        return f"Forgot: {key}"
    return f"No memory for: {key}"

def compact_messages(messages, client):
    """Summarize old messages based on model's context window"""
    ctx_window = get_context_window()
    current_tokens = estimate_messages_tokens(messages)

    # Compact at threshold to leave room for response
    threshold = int(ctx_window * COMPACT_THRESHOLD)

    if current_tokens <= threshold:
        return messages

    # Need to compact - figure out how many recent messages to keep
    system = messages[0]

    # Keep recent messages that fit in allocated space
    target_recent_tokens = int(ctx_window * KEEP_RECENT_PCT)
    recent = []
    recent_tokens = 0

    for m in reversed(messages[1:]):
        msg_tokens = estimate_tokens(m.get("content", "")) + 4
        if recent_tokens + msg_tokens > target_recent_tokens:
            break
        recent.insert(0, m)
        recent_tokens += msg_tokens

    # Everything else gets summarized
    old = messages[1:len(messages)-len(recent)] if recent else messages[1:]

    if not old:
        return messages  # Nothing to compact

    # Build summary of old conversation
    old_text = "\n".join([
        f"{m['role']}: {m['content'][:200]}..." if len(m['content']) > 200 else f"{m['role']}: {m['content']}"
        for m in old
    ])

    try:
        # Use LLM to summarize
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": f"Summarize this conversation in 2-3 sentences, focusing on what was accomplished and key context:\n\n{old_text[:4000]}"
            }],
            max_tokens=256,
        )
        summary = response.choices[0].message.content
    except:
        # Fallback: simple truncation
        summary = f"[Previous conversation: {len(old)} messages about coding tasks]"

    # Reconstruct with summary
    compacted = [
        system,
        {"role": "user", "content": f"[Context from earlier: {summary}]"},
        {"role": "assistant", "content": "Understood, I have context from our earlier conversation."},
    ] + recent

    old_tokens = estimate_messages_tokens(old)
    new_tokens = estimate_messages_tokens(compacted)
    print(f"\033[90m[Compacted {old_tokens:,}→{new_tokens:,} tokens | {ctx_window:,} ctx]\033[0m")
    return compacted

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

def batch_edit(edits):
    """Apply multiple edits across files"""
    results = []
    for edit in edits:
        path = edit.get("path", "")
        old = edit.get("old", "")
        new = edit.get("new", "")
        if path and old:
            result = edit_file(path, old, new)
            results.append(f"{path}: {result}")
        else:
            results.append(f"Invalid edit: {edit}")
    return "\n".join(results)

def project_context(path="."):
    """Auto-load project structure and key files"""
    p = Path(path).expanduser()
    context = []

    # Directory structure (2 levels deep)
    context.append("=== Project Structure ===")
    try:
        for item in sorted(p.iterdir()):
            if item.name.startswith('.') and item.name not in ['.env.example', '.gitignore']:
                continue
            prefix = "📁" if item.is_dir() else "📄"
            context.append(f"{prefix} {item.name}")
            if item.is_dir():
                try:
                    for sub in sorted(item.iterdir())[:10]:
                        if not sub.name.startswith('.'):
                            sp = "  📁" if sub.is_dir() else "  📄"
                            context.append(f"{sp} {sub.name}")
                except:
                    pass
    except Exception as e:
        context.append(f"Error listing: {e}")

    # Key config files
    key_files = [
        "package.json", "pyproject.toml", "setup.py", "Cargo.toml",
        "go.mod", "requirements.txt", "Makefile", "README.md",
        ".gitignore", "tsconfig.json", "vite.config.js", "webpack.config.js"
    ]

    context.append("\n=== Key Files ===")
    for kf in key_files:
        kp = p / kf
        if kp.exists():
            try:
                content = kp.read_text()[:2000]
                context.append(f"\n--- {kf} ---")
                context.append(content)
            except:
                pass

    return "\n".join(context)

def plan_task(task, steps=None):
    """Create or update a task plan"""
    ensure_agent_dir()
    plan_file = AGENT_DIR / "current_plan.json"

    if steps:
        # Save new plan
        plan = {
            "task": task,
            "steps": [{"step": s, "status": "pending"} for s in steps],
            "created": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        plan_file.write_text(json.dumps(plan, indent=2))
        return f"Plan created with {len(steps)} steps:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
    else:
        # Read existing plan
        if plan_file.exists():
            plan = json.loads(plan_file.read_text())
            lines = [f"Task: {plan['task']}", "Steps:"]
            for i, s in enumerate(plan['steps']):
                status = "✓" if s['status'] == 'done' else "○"
                lines.append(f"  {status} {i+1}. {s['step']}")
            return "\n".join(lines)
        return "No active plan. Create one with steps parameter."

def complete_step(step_num):
    """Mark a plan step as complete"""
    ensure_agent_dir()
    plan_file = AGENT_DIR / "current_plan.json"

    if not plan_file.exists():
        return "No active plan"

    plan = json.loads(plan_file.read_text())
    idx = step_num - 1
    if 0 <= idx < len(plan['steps']):
        plan['steps'][idx]['status'] = 'done'
        plan_file.write_text(json.dumps(plan, indent=2))
        return f"Step {step_num} marked complete: {plan['steps'][idx]['step']}"
    return f"Invalid step number: {step_num}"

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

TOOL_HANDLERS = {
    "read_file": lambda p: read_file(p.get("path", "")),
    "write_file": lambda p: write_file(p.get("path", ""), p.get("content", "")),
    "edit_file": lambda p: edit_file(p.get("path", ""), p.get("old", ""), p.get("new", "")),
    "batch_edit": lambda p: batch_edit(p.get("edits", [])),
    "list_dir": lambda p: list_dir(p.get("path", ".")),
    "find_files": lambda p: find_files(p.get("pattern", "*"), p.get("path", ".")),
    "grep": lambda p: grep(p.get("pattern", ""), p.get("path", ".")),
    "search_code": lambda p: search_code(p.get("query", ""), p.get("path", ".")),
    "project_context": lambda p: project_context(p.get("path", ".")),
    "run": lambda p: run(p.get("cmd", "")),
    "git": lambda p: git(p.get("args", "")),
    "web_fetch": lambda p: web_fetch(p.get("url", "")),
    "web_search": lambda p: web_search(p.get("query", "")),
    "remember": lambda p: save_memory(p.get("key", ""), p.get("value", "")),
    "recall": lambda p: get_memory(p.get("key")),
    "forget": lambda p: forget_memory(p.get("key", "")),
    "plan_task": lambda p: plan_task(p.get("task", ""), p.get("steps")),
    "complete_step": lambda p: complete_step(p.get("step_num", 0)),
}

def execute_tool_call(name, arguments):
    """Execute a native tool call"""
    try:
        if arguments is None or arguments == "":
            args = {}
        elif isinstance(arguments, str):
            args = json.loads(arguments)
        elif isinstance(arguments, dict):
            args = arguments
        else:
            args = {}
    except json.JSONDecodeError:
        return f"Error: Invalid arguments for {name}"

    if name in TOOL_HANDLERS:
        try:
            return TOOL_HANDLERS[name](args)
        except Exception as e:
            return f"Error in {name}: {e}"
    return f"Unknown tool: {name}"

# =============================================================================
# UI
# =============================================================================

def show_models():
    print("\n\033[96m=== Models ===\033[0m")
    for k, (mid, name, ctx) in MODELS.items():
        ctx_k = ctx // 1024
        print(f"  {k}) {name} ({ctx_k}k ctx)")
    print()

def select_model():
    global MODEL
    show_models()
    choice = input("Select (1-6) or model ID: ").strip()
    if choice in MODELS:
        mid, name, ctx = MODELS[choice]
        MODEL = mid
        print(f"\033[92m→ {name} ({ctx//1024}k ctx)\033[0m\n")
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

    # Load previous session
    prev_session = load_session()
    if prev_session:
        messages.extend(prev_session)
        print(f"\033[90m[Restored {len(prev_session)} messages from last session]\033[0m")

    # Load memory context for system
    memory = load_memory()
    if memory:
        mem_context = "\n".join([f"- {k}: {v}" for k, v in memory.items()])
        messages.append({"role": "user", "content": f"[Persistent memory loaded:\n{mem_context}]"})
        messages.append({"role": "assistant", "content": "I've loaded your persistent memory."})

    # Get model display name and context
    model_name = MODEL
    model_ctx = 32768
    for k, (mid, name, ctx) in MODELS.items():
        if mid == MODEL:
            model_name = name
            model_ctx = ctx
            break

    print(f"\033[96m{'═'*50}\033[0m")
    print(f"\033[96m  Termux AI Agent\033[0m")
    print(f"\033[96m{'═'*50}\033[0m")
    print(f"Model: \033[93m{model_name}\033[0m ({model_ctx//1024}k ctx)")
    print(f"Dir:   \033[90m{os.getcwd()}\033[0m")
    print(f"\nTools: read, write, edit, grep, find, run, git, web, memory")
    print(f"Cmds:  /model /clear /session /memory /quit")
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
            clear_session()
            print("Cleared session.\n")
            continue
        if user_input.lower() in ["/session", "/s"]:
            tokens = estimate_messages_tokens(messages)
            ctx = get_context_window()
            pct = (tokens / ctx) * 100
            print(f"\033[96m=== Session ===\033[0m")
            print(f"Messages: {len(messages)-1}")
            print(f"Tokens:   ~{tokens:,} / {ctx:,} ({pct:.1f}%)")
            print(f"Compact:  {'yes' if pct >= COMPACT_THRESHOLD*100 else 'no'} (at {COMPACT_THRESHOLD*100:.0f}%)")
            if SESSION_FILE.exists():
                print(f"File:     {SESSION_FILE.stat().st_size:,} bytes")
            print()
            continue
        if user_input.lower() in ["/memory", "/mem"]:
            mem = load_memory()
            if mem:
                print("\033[96m=== Memory ===\033[0m")
                for k, v in mem.items():
                    print(f"  {k}: {v[:50]}..." if len(str(v)) > 50 else f"  {k}: {v}")
            else:
                print("(no memories)")
            print()
            continue
        if user_input.lower().startswith("/forget "):
            key = user_input[8:].strip()
            result = forget_memory(key)
            print(f"{result}\n")
            continue
        if user_input.lower() in ["/model", "/m"]:
            select_model()
            continue
        if user_input.lower() in ["/models"]:
            show_models()
            continue

        messages.append({"role": "user", "content": user_input})

        # Auto-compact if needed
        messages = compact_messages(messages, client)

        # Agent loop with native tool calling + streaming
        tool_call_count = 0
        while True:
            response = None
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        tools=TOOL_DEFINITIONS,
                        tool_choice="auto",
                        max_tokens=16384,
                        stream=True,
                    )
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"\033[93m[retry {attempt+1}]\033[0m ", end="", flush=True)
                        time.sleep(2)
                    else:
                        print(f"\033[91mError: {e}\033[0m\n")

            if not response:
                break

            # Collect streamed response
            content = ""
            tool_calls = {}  # id -> {name, arguments}
            print(f"\033[94mAgent:\033[0m ", end="", flush=True)

            for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                # Stream text content
                if delta.content:
                    print(delta.content, end="", flush=True)
                    content += delta.content

                # Collect tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls:
                            tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc.id:
                            tool_calls[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls[idx]["arguments"] += tc.function.arguments

            # End streaming line if we printed content
            if content and not content.endswith("\n"):
                print()

            # Check for tool calls
            if tool_calls:
                if content:
                    print()  # Newline after any content

                # Build tool_calls list
                tc_list = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]}
                    }
                    for tc in tool_calls.values()
                ]

                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tc_list
                })

                # Execute each tool call
                for tc in tool_calls.values():
                    tool_call_count += 1
                    print(f"\033[93m[{tc['name']}]\033[0m ", end="")

                    result = execute_tool_call(tc["name"], tc["arguments"])
                    display = str(result)[:300] + "..." if len(str(result)) > 300 else result
                    print(display)

                    # Add tool result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": str(result)
                    })

                # Check tool call limit
                if tool_call_count >= MAX_TOOL_CALLS_PER_TURN:
                    print(f"\033[93m[limit: {MAX_TOOL_CALLS_PER_TURN} tools/turn]\033[0m")
                    messages.append({
                        "role": "user",
                        "content": "Tool limit reached. Please summarize what you found and respond."
                    })

                # Continue loop to let model process tool results
                continue

            # No tool calls - just a text response
            if content:
                messages.append({"role": "assistant", "content": content})
                print()

            break

        # Save session after each exchange
        save_session(messages)

if __name__ == "__main__":
    main()
