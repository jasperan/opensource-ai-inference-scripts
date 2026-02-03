# Local LLM Coding Agents

This project sets up local coding agents ([OpenCode](https://opencode.ai/), [Claude Code](https://docs.anthropic.com/en/docs/claude-code), [Crush](https://github.com/charmbracelet/crush)) to run with local LLMs via multiple backends: **llama.cpp** (recommended), **vLLM**, or **Ollama**.

---

## llama.cpp + OpenCode (Recommended)

The recommended setup using [llama.cpp](https://github.com/ggml-org/llama.cpp) as the inference server and [OpenCode](https://opencode.ai/) as the coding agent. Based on the [Unsloth guide](https://unsloth.ai/docs/basics/claude-codex).

Uses the **GLM-4.7-Flash** model from [Unsloth](https://unsloth.ai/) — a 30B parameter model optimized for coding tasks with 128K context.

### Quick Start

```bash
# 1. Run full setup (installs deps, builds llama.cpp, downloads model, configures OpenCode)
./setup_llamacpp.sh
# 2. Launch OpenCode with GLM-4.7-Flash
./run_llamacpp.sh opencode
```

That's it! The script auto-starts the llama-server in background and launches OpenCode.

### Hardware Requirements

- **GPU**: 24GB+ VRAM (RTX 4090, A10, etc.) for full speed
- **CPU-only**: Works but slower; model is ~17GB
- **RAM**: 32GB+ recommended
- **Disk**: ~20GB for model + llama.cpp

### Step-by-Step Setup

**1. Install dependencies and build llama.cpp:**

```bash
./setup_llamacpp.sh deps    # Install build dependencies
./setup_llamacpp.sh build   # Clone and build llama.cpp with CUDA support
```

**2. Download the GLM-4.7-Flash model:**

```bash
./setup_llamacpp.sh model   # Downloads ~17GB GGUF from Hugging Face
```

**3. Configure OpenCode:**

```bash
./setup_llamacpp.sh config  # Adds llama.cpp provider to ~/.config/opencode/opencode.json
```

Or run everything at once:

```bash
./setup_llamacpp.sh all     # Does all steps above
```

### Usage

**Interactive session:**

```bash
./run_llamacpp.sh opencode
```

**One-shot command:**

```bash
./run_llamacpp.sh run "Create a Python script that prints hello world"
```

**Server management:**

```bash
./run_llamacpp.sh start     # Start llama-server in background
./run_llamacpp.sh stop      # Stop the server
./run_llamacpp.sh status    # Check server status
./run_llamacpp.sh logs      # View server logs
./run_llamacpp.sh test      # Test connection and run smoke tests
```

### Configuration

The OpenCode config at `~/.config/opencode/opencode.json` includes:

```json
{
  "provider": {
    "llamacpp": {
      "name": "llama.cpp (local)",
      "npm": "@ai-sdk/openai-compatible",
      "models": {
        "unsloth/GLM-4.7-Flash": {
          "name": "GLM-4.7-Flash [llama.cpp]"
        }
      },
      "options": {
        "baseURL": "http://localhost:10000/v1",
        "apiKey": "sk-EMPTY"
      }
    }
  }
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMACPP_PORT` | `10000` | Server port |
| `LLAMACPP_HOST` | `0.0.0.0` | Server bind address |
| `LLAMACPP_CTX_SIZE` | `131072` | Context window size |

---

### Router Mode (Multi-Model)

llama.cpp supports two server modes:

**Single-model mode** (default): Preloads one model at startup. Simpler, faster initial inference.

**Router mode**: Dynamic multi-model support with on-demand loading. Better for:
- Switching between models without restarting
- Running multiple models concurrently
- Memory-constrained environments (auto-unloads idle models)

#### Enabling Router Mode

```bash
# Start in router mode
LLAMACPP_ROUTER_MODE=1 ./run_llamacpp.sh start

# Or export for session
export LLAMACPP_ROUTER_MODE=1
./run_llamacpp.sh start
```

#### Router Mode Commands

```bash
# List available and loaded models
./run_llamacpp.sh models

# Load a model (by path relative to models/)
./run_llamacpp.sh load GLM-4.7-Flash-GGUF/GLM-4.7-Flash-UD-Q4_K_XL.gguf

# Unload a model to free memory
./run_llamacpp.sh unload GLM-4.7-Flash

# Check status (shows loaded models)
./run_llamacpp.sh status
```

#### Router Mode Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMACPP_ROUTER_MODE` | `0` | Set to `1` to enable router mode |
| `LLAMACPP_MODELS_MAX` | `4` | Max concurrent loaded models (LRU eviction) |
| `LLAMACPP_MODELS_AUTOLOAD` | `1` | `1`=preload models at startup, `0`=on-demand only |
| `LLAMACPP_IDLE_TIMEOUT` | `300` | Seconds before idle model unloads (`0`=never) |

#### Router Mode with Unified CLI (`./ai`)

The unified CLI (`./ai`) provides a TUI interface that automatically detects router mode and offers dynamic model management:

```bash
# Start server in router mode
LLAMACPP_ROUTER_MODE=1 ./run_llamacpp.sh start

# Launch the unified CLI
./ai
```

In the TUI, when llama.cpp is running in router mode:
- **Loaded models** (●) are ready to use immediately
- **Available models** (○) can be loaded on-demand
- **Model management** allows unloading models to free GPU memory

The interface shows:
```
Select model (● loaded, ○ available):
─── Loaded Models (ready to use) ───
  ● GLM-4.7-Flash  [16.3GB]
─── Available Models (need loading) ───
  ○ Other-Model  [8.5GB]
─────────────────────────────────────
  ⚙ Manage models (unload to free memory)
← Back
```

#### Router Mode with Command Line

For scripted usage without the TUI:

```bash
# Start server in router mode
LLAMACPP_ROUTER_MODE=1 ./run_llamacpp.sh start

# Load desired model
./run_llamacpp.sh load GLM-4.7-Flash-GGUF

# Use with OpenCode (model name routed automatically)
./run_llamacpp.sh opencode
```

#### API Endpoints (Router Mode)

Router mode exposes additional endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List loaded models (OpenAI compatible) |
| `/models` | GET | Extended model info (loaded + available) |
| `/models/load` | POST | Load model: `{"model": "path/to/model.gguf"}` |
| `/models/unload` | POST | Unload model: `{"model": "model-name"}` |
| `/health` | GET | Server health status |

---

### Troubleshooting

**Server won't start:**
- Check logs: `./run_llamacpp.sh logs`
- Verify GPU: `nvidia-smi`
- Ensure model exists: `ls -la models/GLM-4.7-Flash-GGUF/`

**Out of memory:**
- Reduce context size: `LLAMACPP_CTX_SIZE=32768 ./run_llamacpp.sh opencode`
- Use CPU-only mode (slower)

**Model not found in OpenCode:**
- Run `./setup_llamacpp.sh config` to update config
- Verify with `opencode models | grep llama`

---

## Alternative Backends

The sections below describe alternative setups using Ollama or vLLM instead of llama.cpp.

---

## Ollama + Claude Code

This setup uses [Ollama](https://ollama.com) as the backend with Claude Code.

### Prerequisites

- [Ollama](https://ollama.com) installed and running.
- `claude` CLI installed (`npm install -g @anthropic-ai/claude-code` or via curl as per docs).
- (Optional) [Crush](https://github.com/charmbracelet/crush) for the Crush pipeline (`./run_crush.sh install` or `npm install -g @charmland/crush`).

## Setup

1.  **Pull supported models**:
    Run the setup script to download `gpt-oss:20b` and `nemotron-3-nano:30b`.
    ```bash
    chmod +x setup_models.sh
    ./setup_models.sh
    ```

## Usage

Run Claude Code with the configured Ollama environment:

```bash
chmod +x run_claude.sh
./run_claude.sh --model gpt-oss:20b
```

Or for Nemotron:

```bash
./run_claude.sh --model nemotron-3-nano:30b
```

Or for Qwen3-Coder:

```bash
./run_claude.sh --model qwen3-coder:latest
```

## Troubleshooting

### "we: command not found" or "invalid tool parameters"
Ensure `run_claude.sh` is up to date and does not contain syntax errors. The script suppresses VS Code extension errors by adding a mock `code` executable to the PATH.

### "Error reading file" or Tool Use Issues
- Verify you are using a tool-capable model like `qwen3-coder:latest`.
- Ensure `ANTHROPIC_API_KEY=ollama` and `ANTHROPIC_BASE_URL=http://localhost:11434` are exported (handled by `run_claude.sh`).
- If you encounter network errors, check if Ollama is running (`ollama ps`) and bound correctly.
- Be explicit with file paths (e.g., "Read /abs/path/to/README.md").

### VS Code Extension Error
If you see errors about installing VS Code extensions, the `run_claude.sh` script automatically handles this by mocking the `code` binary. Ensure you run Claude via the script.


## Harbor CLI

This project includes `harbor`, a CLI tool to manage your local setup.

### Usage

```bash
# 1. Configure Ollama version
./harbor config set ollama.version 0.15.4

# 2. Pull resources
./harbor pull ollama           # Downloads specified Ollama version
./harbor pull glm-4.7-flash    # Pulls the model via Ollama

# 3. Start a coding agent
./harbor up opencode           # Starts Claude Code
./harbor up crush              # Starts Crush (Ollama backend, default model)
./harbor up crush --backend vllm                # Crush with vLLM
./harbor up crush --model nemotron-3-nano:30b   # Crush with specific model
```

## vLLM + OpenCode Pipeline

An alternative to the Ollama + Claude Code setup above. Uses [vLLM](https://docs.vllm.ai/) as the inference server and [OpenCode](https://opencode.ai/) as the coding agent, with a local Qwen3 model.

### Hardware tested on

- NVIDIA A10 (23 GB VRAM)
- Ubuntu, CUDA 12.9, Python 3.12
- vLLM 0.14.1, OpenCode 1.1.34

### 1. Install

```bash
pip install vllm --upgrade
# OpenCode (CLI tool, not the Python SDK)
# See https://opencode.ai for latest install method
# If already installed, verify with:
opencode --version
```

### 2. Start vLLM server

> **Tip:** You can skip this step entirely — `./run_vllm.sh crush` and `./run_vllm.sh opencode` auto-start the server in the background.

To run in a **dedicated foreground terminal** instead:

```bash
vllm serve Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.92 \
  --max-model-len 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

The first run downloads ~16 GB of model weights from Hugging Face.

**Flags explained:**

| Flag | Purpose |
|------|---------|
| `--gpu-memory-utilization 0.92` | Use 92% of VRAM (needed to fit 8B model + KV cache on 23 GB card) |
| `--max-model-len 16384` | Context window; 32K won't fit on A10 after model weights |
| `--enable-auto-tool-choice` | Let the model decide when to call tools |
| `--tool-call-parser hermes` | Parse tool calls using Hermes format (works with Qwen3) |

**Optional: disable thinking mode** for faster, shorter responses:

```bash
vllm serve Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.92 \
  --max-model-len 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

### 3. Verify the server

In a second terminal:

```bash
# Check model is loaded
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# Quick tool-call smoke test
python3 -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:8000/v1', api_key='sk-EMPTY')
r = client.chat.completions.create(
    model='Qwen/Qwen3-8B',
    messages=[{'role':'user','content':'What is 2+2? Use the calc tool.'}],
    tools=[{'type':'function','function':{'name':'calc','description':'Calculate','parameters':{'type':'object','properties':{'expr':{'type':'string'}},'required':['expr']}}}],
    tool_choice='auto'
)
print(r.choices[0].message.tool_calls)
"
```

### 4. Configure OpenCode

The config lives at `~/.config/opencode/opencode.json`. Add the vLLM provider:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "vllm": {
      "name": "vLLM (local)",
      "npm": "@ai-sdk/openai-compatible",
      "models": {
        "Qwen/Qwen3-8B": {
          "name": "Qwen3-8B [vLLM]"
        }
      },
      "options": {
        "baseURL": "http://localhost:8000/v1",
        "apiKey": "sk-EMPTY"
      }
    }
  }
}
```

Verify it appears:

```bash
opencode models          # should list vllm/Qwen/Qwen3-8B
```

### 5. Run a coding agent (single terminal)

Both `opencode` and `crush` commands **auto-start the vLLM server in the background** if it isn't already running. No second terminal needed.

**Crush** (single command):

```bash
cd ~/your-project
./run_vllm.sh crush
# Starts vLLM in background → waits for health → launches Crush
```

**OpenCode** (single command):

```bash
cd ~/your-project
./run_vllm.sh opencode
# Starts vLLM in background → waits for health → launches OpenCode
```

**OpenCode one-shot** (non-interactive):

```bash
opencode run "Create a Python file hello.py that prints 'Hello from local agent'"
```

If vLLM is already running, both commands skip the startup and launch immediately.

**Manage the background server:**

```bash
./run_vllm.sh status    # Check if running, show PID and model
./run_vllm.sh logs      # Tail the background server log
./run_vllm.sh stop      # Kill the background server
```

### 6. Test plan

Run these from a scratch directory to validate the full pipeline:

```bash
mkdir -p ~/opencode-test && cd ~/opencode-test

# Test 1: File creation
opencode run "Create hello.py that prints 'Hello from local agent'"

# Test 2: Read + edit
opencode run "Read hello.py and add a comment at the top saying 'Tested with vLLM + Qwen3'"

# Test 3: Multi-step agentic
opencode run "Initialize a git repo, create README.md with description 'Local AI coding agent test', commit it"

# Test 4: Coding task
opencode run "Write a script to fetch the current date and write it to now.txt"

# Test 5: Tool chaining
opencode run "Create backup.txt as a copy of hello.py"
```

### Switching models

To swap models, stop the vLLM server (`Ctrl+C`) and restart with a different model. Then update `opencode.json` to match.

Some alternatives that fit on a 23 GB A10:

| Model | Size | Notes |
|-------|------|-------|
| `Qwen/Qwen3-8B` | ~16 GB | General purpose, built-in thinking mode |
| `Qwen/Qwen3-8B-AWQ` | ~5 GB | 4-bit quantized, faster, more KV cache room |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | ~14 GB | Code-specialized |

### Stopping

```bash
# Find and kill the vLLM server
lsof -ti :8000 | xargs kill
```

## Crush + Ollama Pipeline

[Crush](https://github.com/charmbracelet/crush) is a terminal-based AI coding assistant from Charm. It supports local models via OpenAI-compatible APIs, making it a natural fit for Ollama and vLLM backends.

### 1. Install Crush

```bash
# Automated (picks best available method)
./run_crush.sh install

# Or manually:
npm install -g @charmland/crush     # npm
brew install charmbracelet/tap/crush # Homebrew
go install github.com/charmbracelet/crush@latest # Go
```

### 2. Run with Ollama backend

Make sure Ollama is running (`ollama serve` or check with `ollama ps`), then:

```bash
# Default: Ollama + qwen3-coder:latest
./run_crush.sh

# Specific model
./run_crush.sh --model nemotron-3-nano:30b

# Qwen 3 30B
./run_crush.sh --model qwen3:30b

# GPT-OSS
./run_crush.sh --model gpt-oss:20b
```

### 3. Run with vLLM backend

Start the vLLM server first (see [vLLM + OpenCode Pipeline](#vllm--opencode-pipeline) above), then:

```bash
./run_crush.sh --backend vllm
./run_crush.sh --backend vllm --model Qwen/Qwen3-8B-AWQ
```

### 4. Configuration details

The `run_crush.sh` script auto-generates `~/.config/crush/crush.json`. You can also write config without launching:

```bash
./run_crush.sh config --backend ollama --model qwen3-coder:latest
./run_crush.sh config --backend vllm
```

The generated config for Ollama looks like:

```json
{
  "$schema": "https://charm.land/crush.json",
  "providers": {
    "ollama": {
      "name": "Ollama (local)",
      "base_url": "http://localhost:11434/v1/",
      "type": "openai-compat",
      "models": [
        {
          "name": "Qwen3 Coder",
          "id": "qwen3-coder:latest",
          "context_window": 131072,
          "default_max_tokens": 20000
        }
      ]
    }
  }
}
```

### 5. Test connectivity

Verify the backend is reachable before launching:

```bash
./run_crush.sh test --backend ollama
./run_crush.sh test --backend vllm
```

### 6. Custom Ollama host

If Ollama runs on a remote machine or non-default port:

```bash
./run_crush.sh --ollama-host http://192.168.1.100:11434
```

### Available commands

| Command | Description |
|---------|-------------|
| `./run_crush.sh` | Launch Crush with Ollama (default) |
| `./run_crush.sh run` | Same as above |
| `./run_crush.sh config` | Write config file only |
| `./run_crush.sh install` | Install Crush |
| `./run_crush.sh test` | Test backend then launch |

### Available options

| Option | Default | Description |
|--------|---------|-------------|
| `--backend` | `ollama` | Backend: `ollama` or `vllm` |
| `--model` | auto | Model ID (auto-detected per backend) |
| `--ollama-host` | `http://localhost:11434` | Ollama server URL |
| `--vllm-host` | `http://localhost:8000` | vLLM server URL |

## Configuration (Ollama pipeline)

The `run_claude.sh` script sets the following environment variables:
- `ANTHROPIC_API_KEY=ollama`
- Unsets `ANTHROPIC_AUTH_TOKEN` to avoid conflicts.
- `ANTHROPIC_BASE_URL=http://localhost:11434`
