#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$HOME/Library/Logs/awesome-mllm-seg"
mkdir -p "$LOG_DIR"

cd "$REPO_DIR"

export AUTO_PUSH="${AUTO_PUSH:-1}"
export LLM_BACKEND="${LLM_BACKEND:-auto}"
export OLLAMA_TIMEOUT_SEC="${OLLAMA_TIMEOUT_SEC:-180}"
export OLLAMA_RETRIES="${OLLAMA_RETRIES:-2}"
export OLLAMA_BATCH_SIZE="${OLLAMA_BATCH_SIZE:-8}"
export MAX_CANDIDATES="${MAX_CANDIDATES:-48}"
export MAX_ABS_TEXT_CHARS="${MAX_ABS_TEXT_CHARS:-900}"

/usr/bin/python3 "$REPO_DIR/.local/auto_update_readme.py" >> "$LOG_DIR/agent.log" 2>&1
