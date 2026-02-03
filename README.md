# Local LLM Coding Agents

Run local coding agents ([OpenCode](https://opencode.ai/), [Crush](https://github.com/charmbracelet/crush)) with local LLMs via **llama.cpp**, **vLLM**, or **Ollama**.

## Quick Start

```bash
# Install dependencies
pip install questionary rich

# Run interactive setup
python setup.py
```

That's it! The TUI guides you through:
1. Installing a backend (llama.cpp, vLLM, or Ollama)
2. Downloading a model
3. Installing a client (OpenCode or Crush)
4. Launching your coding agent

## Commands

```bash
python setup.py              # Interactive TUI (default)
python setup.py install      # Install backend + model
python setup.py run          # Select backend/model/client and launch
python setup.py status       # Show what's installed and running
```

## Supported Backends

| Backend | Port | Description |
|---------|------|-------------|
| **llama.cpp** | 10000 | Recommended - fast, 128K context, builds from source |
| **vLLM** | 8000 | Multi-model support, requires more VRAM |
| **Ollama** | 11434 | Easiest setup, pre-built binaries |

## Supported Models

### llama.cpp
| Model | Size | Context |
|-------|------|---------|
| GLM-4.7-Flash (Q4_K_XL) | 17GB | 128K |
| GLM-4.7-Flash (Q8_0) | 30GB | 128K |

### vLLM
| Model | Size | Context |
|-------|------|---------|
| GLM-4.7-Flash AWQ | 10GB | 19K |
| Qwen3-8B | 16GB | 16K |
| Qwen3-8B AWQ | 5GB | 32K |

### Ollama
| Model | Size | Context |
|-------|------|---------|
| glm-4.7-flash:latest | 9GB | 128K |
| qwen3-coder:latest | 5GB | 128K |
| gpt-oss:20b | 12GB | 32K |
| nemotron-3-nano:30b | 18GB | 32K |

## Supported Clients

| Client | Description |
|--------|-------------|
| **OpenCode** | Full-featured AI coding assistant TUI |
| **Crush** | Charm's lightweight AI chat TUI |

## Hardware Requirements

- **GPU**: 24GB+ VRAM recommended (RTX 4090, A10, etc.)
- **CPU-only**: Works but slower
- **RAM**: 32GB+ recommended
- **Disk**: 20-50GB depending on model choice

## Manual Server Management

If you need to manage servers directly:

```bash
# llama.cpp
./run_llamacpp.sh start      # Start server
./run_llamacpp.sh stop       # Stop server
./run_llamacpp.sh status     # Check status
./run_llamacpp.sh logs       # View logs

# vLLM
./run_vllm.sh serve          # Start server (foreground)
./run_vllm.sh stop           # Stop server
./run_vllm.sh status         # Check status

# Ollama
ollama serve                 # Start server
ollama list                  # List models
```

## Environment Variables

### llama.cpp
| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMACPP_PORT` | 10000 | Server port |
| `LLAMACPP_HOST` | 0.0.0.0 | Server bind address |
| `LLAMACPP_CTX_SIZE` | 131072 | Context window size |
| `LLAMACPP_ROUTER_MODE` | 0 | Set to 1 for multi-model mode |

### vLLM
| Variable | Default | Description |
|----------|---------|-------------|
| Port configured via `--port` flag | 8000 | Server port |

## Troubleshooting

### Server won't start
```bash
python setup.py status       # Check what's running
./run_llamacpp.sh logs       # Check llama.cpp logs
nvidia-smi                   # Verify GPU
```

### Out of memory
- Use a smaller quantization (Q4 instead of Q8)
- Reduce context size: `LLAMACPP_CTX_SIZE=32768`
- Try vLLM with AWQ models (smaller memory footprint)

### Model not found
```bash
python setup.py install      # Re-run installation
```

## Project Structure

```
.
├── setup.py              # Unified setup TUI (main entry point)
├── run_llamacpp.sh       # llama.cpp server management
├── run_vllm.sh           # vLLM server management
├── run_crush.sh          # Crush client configuration
├── llama.cpp/            # llama.cpp build (created by setup)
└── models/               # Downloaded models (created by setup)
```

## License

MIT
