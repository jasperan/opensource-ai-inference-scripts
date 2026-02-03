#!/bin/bash
set -e

echo "Pulling models for Claude Code compatible setup..."

# Pull gpt-oss 20b
echo "Pulling gpt-oss:20b..."
ollama pull gpt-oss:20b

# Pull nemotron-3-nano
# Note: The search result mentioned nemotron-3-nano:30b, but sometimes 'nemotron-mini' or similar exists. 
# We'll stick to the specific request or closest match found in search if 30b is the tag.
# Search result said: "ollama pull nemotron-3-nano:30b"
echo "Pulling nemotron-3-nano:30b..."
ollama pull nemotron-3-nano:30b

# Pull qwen3-coder
echo "Pulling qwen3-coder:latest..."
ollama pull qwen3-coder:latest

echo "Models pulled successfully!"
ollama list
