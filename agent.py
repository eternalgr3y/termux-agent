#!/usr/bin/env python3
"""
Simple Agentic Coding Assistant for Termux
Uses OpenRouter free models
"""

import os
import sys
import json
import subprocess
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Installing openai...")
    subprocess.run([sys.executable, "-m", "pip", "install", "openai", "-q"])
    from openai import OpenAI

# Available models
MODELS = {
    "1": ("kwaipilot/kat-coder-pro:free", "KAT-Coder-Pro V1 (73% SWE-Bench)"),
    "2": ("mistralai/devstral-2512:free", "Devstral 2 2512 (123B agentic)"),
    "3": ("tngtech/deepseek-r1t2-chimera:free", "DeepSeek R1T2 Chimera (reasoning)"),
    "4": ("deepseek/deepseek-r1-0528:free", "DeepSeek R1 0528 (671B reasoning)"),
    "5": ("deepseek/deepseek-chat-v3-0324:free", "DeepSeek Chat V3 (fast)"),
    "6": ("x-ai/grok-4-fast:free", "Grok 4 Fast"),
}

# Config
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = os.environ.get("AGENT_MODEL", "kwaipilot/kat-coder-pro:free")

SYSTEM_PROMPT = """You are an AI coding assistant running in a terminal. You can:
1. Read files - use read_file(path)
2. Write files - use write_file(path, content)
3. Run shell commands - use run_command(cmd)
4. List directory - use list_dir(path)

When the user asks you to do something, analyze what's needed and use these tools.
Always explain what you're doing. After using a tool, analyze the result and continue.

Respond with JSON when using tools:
{"tool": "read_file", "path": "file.py"}
{"tool": "write_file", "path": "file.py", "content": "..."}
{"tool": "run_command", "cmd": "ls -la"}
{"tool": "list_dir", "path": "."}
{"tool": "done", "message": "Task completed"}

For regular responses, just reply normally without JSON."""

def read_file(path):
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"

def write_file(path, content):
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        return f"Written to {path}"
    except Exception as e:
        return f"Error: {e}"

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        output = result.stdout + result.stderr
        return output[:4000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out"
    except Exception as e:
        return f"Error: {e}"

def list_dir(path="."):
    try:
        items = os.listdir(path)
        return "\n".join(items)
    except Exception as e:
        return f"Error: {e}"

def execute_tool(tool_call):
    tool = tool_call.get("tool")
    if tool == "read_file":
        return read_file(tool_call.get("path", ""))
    elif tool == "write_file":
        return write_file(tool_call.get("path", ""), tool_call.get("content", ""))
    elif tool == "run_command":
        return run_command(tool_call.get("cmd", ""))
    elif tool == "list_dir":
        return list_dir(tool_call.get("path", "."))
    elif tool == "done":
        return None
    return "Unknown tool"

def parse_tool_call(text):
    """Try to extract JSON tool call from response"""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            json_str = text[start:end]
            return json.loads(json_str)
    except:
        pass
    return None

def show_models():
    print("\n\033[96m=== Available Models ===\033[0m")
    for key, (model_id, desc) in MODELS.items():
        print(f"  {key}) {desc}")
        print(f"     \033[90m{model_id}\033[0m")
    print()

def select_model():
    global MODEL
    show_models()
    choice = input("Select model (1-6) or enter custom model ID: ").strip()
    if choice in MODELS:
        MODEL = MODELS[choice][0]
        print(f"\033[92mSwitched to: {MODELS[choice][1]}\033[0m\n")
    elif choice:
        MODEL = choice
        print(f"\033[92mSwitched to: {MODEL}\033[0m\n")

def main():
    global MODEL

    if not OPENROUTER_API_KEY:
        print("\033[91mSet OPENROUTER_API_KEY environment variable\033[0m")
        print("Get free key at: https://openrouter.ai/keys")
        print("\nExport it:")
        print("  export OPENROUTER_API_KEY=sk-or-...")
        sys.exit(1)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Find current model name
    current_name = MODEL
    for key, (mid, desc) in MODELS.items():
        if mid == MODEL:
            current_name = desc
            break

    print(f"\033[96m{'='*50}\033[0m")
    print(f"\033[96m  Termux AI Agent\033[0m")
    print(f"\033[96m{'='*50}\033[0m")
    print(f"Model: \033[93m{current_name}\033[0m")
    print(f"Dir:   \033[90m{os.getcwd()}\033[0m")
    print()
    print("Commands: \033[90m/model\033[0m switch model | \033[90m/models\033[0m list | \033[90m/clear\033[0m reset | \033[90m/quit\033[0m exit")
    print()

    while True:
        try:
            user_input = input("\033[92mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() in ["/quit", "quit", "exit"]:
            break
        if user_input.lower() in ["/clear", "clear"]:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Cleared.\n")
            continue
        if user_input.lower() in ["/models", "/list"]:
            show_models()
            continue
        if user_input.lower() in ["/model", "/switch"]:
            select_model()
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            continue

        messages.append({"role": "user", "content": user_input})

        # Agent loop
        while True:
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=4096,
                )

                reply = response.choices[0].message.content
                messages.append({"role": "assistant", "content": reply})

                tool_call = parse_tool_call(reply)

                if tool_call:
                    tool_name = tool_call.get("tool", "")
                    print(f"\033[93m[{tool_name}]\033[0m", end=" ")

                    if tool_name == "done":
                        print(tool_call.get("message", "Done"))
                        break

                    result = execute_tool(tool_call)
                    print(f"{result[:200]}..." if len(str(result)) > 200 else result)
                    messages.append({"role": "user", "content": f"Tool result:\n{result}"})
                else:
                    print(f"\033[94mAgent:\033[0m {reply}\n")
                    break

            except Exception as e:
                print(f"\033[91mError:\033[0m {e}\n")
                break

if __name__ == "__main__":
    main()
