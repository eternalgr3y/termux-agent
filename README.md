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
- `/clear` - reset conversation
- `/quit` - exit

## Tools

The agent can:
- Read files
- Write files
- Run shell commands
- List directories
