#!/bin/bash

# Default model
MODEL="gpt-oss:20b"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Starting Claude Code with Ollama backend..."
echo "Model: $MODEL"
echo "Endpoint: http://localhost:11434"

# Export variables for Claude Code
export ANTHROPIC_AUTH_TOKEN=ollama
export ANTHROPIC_BASE_URL=http://129.213.76.89:11434

# Add mock_bin to PATH to suppress VS Code extension installation error
export PATH="$PWD/mock_bin:$PATH"

# Run Claude

claude --model "$MODEL"
