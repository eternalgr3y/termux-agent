# Termux AI Agent

A powerful agentic coding assistant for Termux using OpenRouter free models. Features streaming responses, native tool calling, task planning, and persistent memory.

## Setup

```bash
# Install dependencies
pip install openai requests beautifulsoup4 lxml

# Set API key (get free key at https://openrouter.ai/keys)
export OPENROUTER_API_KEY="sk-or-your-key"

# Run
python agent.py
```

## Models

| # | Model | Context | Best For |
|---|-------|---------|----------|
| 1 | KAT-Coder-Pro V1 | 256k | Agentic coding (73% SWE-Bench) |
| 2 | Devstral 2 2512 | 128k | Multi-file edits |
| 3 | DeepSeek R1T2 Chimera | 64k | Reasoning tasks |
| 4 | DeepSeek R1 0528 | 128k | Complex reasoning |
| 5 | DeepSeek Chat V3 | 128k | Fast general coding |
| 6 | Grok 4 Fast | 128k | Quick responses |

## Commands

- `/model` - switch model
- `/models` - list models
- `/session` - show token usage & context %
- `/memory` - show persistent memory
- `/forget <key>` - delete memory item
- `/clear` - reset conversation
- `/quit` - exit

## Tools (18 total)

**File Operations:**
- `read_file` - read file contents
- `write_file` - create/overwrite file
- `edit_file` - find and replace in file
- `batch_edit` - multiple edits across files
- `list_dir` - list directory
- `find_files` - glob pattern search

**Search:**
- `grep` - regex search in files
- `search_code` - smart code pattern search
- `project_context` - load project structure + configs

**Shell:**
- `run` - execute shell command
- `git` - git operations

**Web:**
- `web_fetch` - fetch URL as text
- `web_search` - DuckDuckGo search

**Memory:**
- `remember` - save to persistent memory
- `recall` - retrieve from memory
- `forget` - delete from memory

**Planning:**
- `plan_task` - create step-by-step plans
- `complete_step` - mark plan steps done

## Features

- **Streaming Responses** - See output as it generates
- **Native Tool Calling** - Structured tool calls, not JSON parsing
- **Task Planning** - Break down complex tasks into steps
- **Batch Editing** - Edit multiple files in one operation
- **Project Context** - Auto-load project structure and configs
- **Session Persistence** - Conversations saved to disk
- **Persistent Memory** - Key-value store across sessions
- **Smart Auto-Compact** - Summarizes old context when near limit
- **Tool Call Limits** - Prevents infinite loops (10/turn max)

## Architecture

```
User Input
    ↓
[Load session + memory]
    ↓
API Request (stream=True, tools=[18 schemas])
    ↓
Stream response chunks to terminal
    ↓
If tool_calls: execute → feed results → loop
    ↓
Save session
```

## Data Storage

```
~/.termux-agent/
├── session.json      # Conversation history
├── memory.json       # Persistent key-value store
└── current_plan.json # Active task plan
```
