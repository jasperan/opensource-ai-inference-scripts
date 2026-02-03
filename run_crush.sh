#!/bin/bash
set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
BACKEND="ollama"       # ollama | vllm
OLLAMA_HOST="http://localhost:11434"
VLLM_HOST="http://localhost:8000"
MODEL=""               # auto-detected per backend if empty
ACTION="run"           # run | config | install | test

usage() {
    cat <<EOF
Usage: ./run_crush.sh [command] [options]

Commands:
  run        Launch Crush TUI (default)
  config     Write Crush config and exit (does not launch)
  install    Install Crush via the best available method
  test       Verify backend connectivity then launch Crush

Options:
  --backend <name>     Backend: ollama or vllm            (default: $BACKEND)
  --model <id>         Model ID override
                         ollama default: glm-4.7-flash:latest
                         vllm   default: Qwen/Qwen3-8B
  --ollama-host <url>  Ollama base URL                    (default: $OLLAMA_HOST)
  --vllm-host <url>    vLLM base URL                      (default: $VLLM_HOST)
  -h, --help           Show this help

Examples:
  ./run_crush.sh                                    # Ollama + default model
  ./run_crush.sh --backend vllm                     # vLLM + Qwen3-8B
  ./run_crush.sh --model nemotron-3-nano:30b        # Ollama + specific model
  ./run_crush.sh config --backend vllm              # Write config only
  ./run_crush.sh install                            # Install Crush
  ./run_crush.sh test --backend ollama              # Test connection then launch
EOF
    exit 0
}

# ─── Parse args ──────────────────────────────────────────────────────────────
if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
    ACTION="$1"; shift
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)      BACKEND="$2";     shift ;;
        --model)        MODEL="$2";       shift ;;
        --ollama-host)  OLLAMA_HOST="$2"; shift ;;
        --vllm-host)    VLLM_HOST="$2";   shift ;;
        -h|--help)      usage ;;
        *)              echo "Unknown option: $1"; usage ;;
    esac
    shift
done

# ─── Resolve model defaults ─────────────────────────────────────────────────
if [[ -z "$MODEL" ]]; then
    case "$BACKEND" in
        ollama) MODEL="glm-4.7-flash:latest" ;;
        vllm)   MODEL="Qwen/Qwen3-8B"       ;;
        *)      echo "Unknown backend: $BACKEND"; exit 1 ;;
    esac
fi

# ─── Config paths ────────────────────────────────────────────────────────────
CRUSH_GLOBAL_DIR="${HOME}/.config/crush"
CRUSH_GLOBAL_CONFIG="${CRUSH_GLOBAL_DIR}/crush.json"

# ─── Helper: build model entry for the selected backend ──────────────────────
build_ollama_models() {
    # Map common Ollama model IDs to friendly names and context windows
    local id="$1"
    local name ctx
    case "$id" in
        glm-4.7-flash:*)     name="GLM-4.7 Flash";      ctx=131072 ;;
        qwen3-coder:*)       name="Qwen3 Coder";        ctx=131072 ;;
        qwen3:30b*)          name="Qwen 3 30B";          ctx=256000 ;;
        gpt-oss:20b*)        name="GPT-OSS 20B";         ctx=32768  ;;
        nemotron-3-nano:30b*)name="Nemotron-3-Nano 30B"; ctx=32768  ;;
        *)                   name="$id";                  ctx=32768  ;;
    esac
    cat <<JSON
        {
          "name": "$name",
          "id": "$id",
          "context_window": $ctx,
          "default_max_tokens": 20000
        }
JSON
}

build_vllm_models() {
    local id="$1"
    local name ctx max_tok
    case "$id" in
        cyankiwi/GLM-4.7-Flash-AWQ-4bit)  name="GLM-4.7 Flash AWQ [vLLM]";   ctx=19136;  max_tok=19136 ;;
        Qwen/Qwen3-8B)                    name="Qwen3-8B [vLLM]";            ctx=16384;  max_tok=2048 ;;
        Qwen/Qwen3-8B-AWQ)                name="Qwen3-8B-AWQ [vLLM]";        ctx=32768;  max_tok=4096 ;;
        Qwen/Qwen2.5-Coder-7B-Instruct)   name="Qwen2.5-Coder-7B [vLLM]";   ctx=16384;  max_tok=2048 ;;
        *)                                 name="$id [vLLM]";                 ctx=16384;  max_tok=2048 ;;
    esac
    cat <<JSON
        {
          "name": "$name",
          "id": "$id",
          "context_window": $ctx,
          "default_max_tokens": $max_tok
        }
JSON
}

# ─── Write crush config ─────────────────────────────────────────────────────
write_config() {
    mkdir -p "$CRUSH_GLOBAL_DIR"

    local base_url model_block

    if [[ "$BACKEND" == "ollama" ]]; then
        base_url="${OLLAMA_HOST}/v1/"
        model_block="$(build_ollama_models "$MODEL")"
    else
        base_url="${VLLM_HOST}/v1/"
        model_block="$(build_vllm_models "$MODEL")"
    fi

    cat > "$CRUSH_GLOBAL_CONFIG" <<EOF
{
  "\$schema": "https://charm.land/crush.json",
  "providers": {
    "${BACKEND}": {
      "name": "${BACKEND^} (local)",
      "base_url": "$base_url",
      "type": "openai-compat",
      "models": [
$model_block
      ]
    }
  },
  "lsp": {
    "pylsp": {
      "command": "pylsp",
      "enabled": true
    },
    "bash-language-server": {
      "command": "bash-language-server",
      "args": ["start"],
      "enabled": true
    }
  }
}
EOF

    echo "Crush config written to $CRUSH_GLOBAL_CONFIG"
    echo "  Backend: $BACKEND"
    echo "  Model:   $MODEL"
    echo "  URL:     $base_url"
}

# ─── Install crush ───────────────────────────────────────────────────────────
do_install() {
    if command -v crush &>/dev/null; then
        echo "Crush is already installed: $(command -v crush)"
        crush --version 2>/dev/null || true
        return 0
    fi

    echo "Installing Crush..."

    # Try npm first (most portable)
    if command -v npm &>/dev/null; then
        echo "Installing via npm..."
        npm install -g @charmland/crush
    elif command -v brew &>/dev/null; then
        echo "Installing via Homebrew..."
        brew install charmbracelet/tap/crush
    elif command -v go &>/dev/null; then
        echo "Installing via Go..."
        go install github.com/charmbracelet/crush@latest
    else
        echo "No supported package manager found (npm, brew, go)."
        echo "Install manually: https://github.com/charmbracelet/crush"
        exit 1
    fi

    echo ""
    if command -v crush &>/dev/null; then
        echo "Crush installed successfully!"
        crush --version 2>/dev/null || true
    else
        echo "Installation completed but 'crush' not found in PATH."
        echo "You may need to restart your shell or add the install location to PATH."
    fi
}

# ─── Test backend connectivity ───────────────────────────────────────────────
do_test() {
    local url
    if [[ "$BACKEND" == "ollama" ]]; then
        url="${OLLAMA_HOST}/v1/models"
    else
        url="${VLLM_HOST}/v1/models"
    fi

    echo "Testing $BACKEND backend at $url ..."
    if curl -sf "$url" | python3 -m json.tool 2>/dev/null; then
        echo ""
        echo "Backend is reachable and serving models."
    else
        echo "ERROR: Cannot reach $BACKEND at $url"
        echo ""
        if [[ "$BACKEND" == "ollama" ]]; then
            echo "Make sure Ollama is running:  ollama serve"
        else
            echo "Make sure vLLM is running:  ./run_vllm.sh serve"
        fi
        exit 1
    fi
}

# ─── Launch crush ────────────────────────────────────────────────────────────
do_run() {
    if ! command -v crush &>/dev/null; then
        echo "Error: 'crush' is not installed."
        echo "Run:  ./run_crush.sh install"
        exit 1
    fi

    write_config
    echo ""
    echo "Launching Crush..."
    exec crush
}

# ─── Dispatch ────────────────────────────────────────────────────────────────
case "$ACTION" in
    run)
        do_run
        ;;
    config)
        write_config
        ;;
    install)
        do_install
        ;;
    test)
        do_test
        echo ""
        do_run
        ;;
    *)
        echo "Unknown command: $ACTION"
        usage
        ;;
esac
