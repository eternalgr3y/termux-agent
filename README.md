# Termux AI Agent

A lightweight agentic coding assistant for Termux using OpenRouter free models.

## Setup

```bash
# Install dependency
pip install openai

# Set API key (get free key at https://openrouter.ai/keys)
export OPENROUTER_API_KEY="sk-or-your-key"

# Run
python agent.py
```

## Models

| # | Model | Best For |
|---|-------|----------|
| 1 | KAT-Coder-Pro V1 | Agentic coding (73% SWE-Bench) |
| 2 | Devstral 2 2512 | Multi-file edits (123B) |
| 3 | DeepSeek R1T2 Chimera | Reasoning tasks |
| 4 | DeepSeek R1 0528 | Complex reasoning (671B) |
| 5 | DeepSeek Chat V3 | Fast general coding |
| 6 | Grok 4 Fast | Fast responses |

## Commands

- `/model` - switch model
- `/models` - list models
- `/session` - show session info
- `/memory` - show persistent memory
- `/forget <key>` - delete memory item
- `/clear` - reset conversation
- `/quit` - exit

## Tools

**File Operations:** read_file, write_file, edit_file, list_dir, find_files

**Search:** grep, search_code

**Shell:** run, git

**Web:** web_fetch, web_search

**Memory:** remember, recall, forget

## Features

- **Session Persistence** - Conversations auto-save to `~/.termux-agent/session.json`
- **Persistent Memory** - Store key-value pairs across sessions
- **Auto-Compact** - Summarizes old messages when context gets long
- **Retry Logic** - Auto-retries failed API calls
- **Model Switching** - Switch between 6 free models on the fly
