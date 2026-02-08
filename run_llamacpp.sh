#!/bin/bash
# Run llama.cpp server with OpenCode or Crush
# Based on: https://unsloth.ai/docs/basics/claude-codex
#
# Supports two modes:
# 1. Single-model mode (default): Preloads model at startup with -m flag
# 2. Router mode: Dynamic multi-model with on-demand loading via --models-dir
#
# Set LLAMACPP_ROUTER_MODE=1 to enable router mode
# Set LLAMACPP_CLIENT=crush to use Crush instead of OpenCode

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_CPP_DIR="${SCRIPT_DIR}/llama.cpp"
MODEL_DIR="${SCRIPT_DIR}/models"
MODEL_FILE=""          # resolved interactively or via LLAMACPP_MODEL_FILE
LLAMA_SERVER="${LLAMA_CPP_DIR}/llama-server"

# Server configuration
PORT="${LLAMACPP_PORT:-10000}"
HOST="${LLAMACPP_HOST:-0.0.0.0}"
CTX_SIZE="${LLAMACPP_CTX_SIZE:-131072}"
MODEL_ALIAS="unsloth/GLM-4.7-Flash"

# Client configuration (opencode or crush)
CLIENT="${LLAMACPP_CLIENT:-opencode}"

# Router mode configuration
ROUTER_MODE="${LLAMACPP_ROUTER_MODE:-0}"
MODELS_MAX="${LLAMACPP_MODELS_MAX:-4}"           # Max concurrent loaded models (LRU eviction)
MODELS_AUTOLOAD="${LLAMACPP_MODELS_AUTOLOAD:-1}" # 1=preload on startup, 0=on-demand only
IDLE_TIMEOUT="${LLAMACPP_IDLE_TIMEOUT:-300}"     # Seconds before idle model unloads (0=never)

# Log file for background server
LOG_FILE="${SCRIPT_DIR}/.llamacpp-server.log"
PID_FILE="${SCRIPT_DIR}/.llamacpp-server.pid"

# ─── Interactive model selection ────────────────────────────────────────────
select_gguf_model() {
    local gguf_files=()
    while IFS= read -r -d '' f; do
        gguf_files+=("$f")
    done < <(find "$MODEL_DIR" -name "*.gguf" -print0 2>/dev/null | sort -z)

    if [[ ${#gguf_files[@]} -eq 0 ]]; then
        echo "No GGUF models found in $MODEL_DIR"
        echo "Run './setup.py install' to download models."
        exit 1
    fi

    if [[ ${#gguf_files[@]} -eq 1 ]]; then
        MODEL_FILE="${gguf_files[0]}"
        echo "Auto-selected model: $(basename "$MODEL_FILE")"
        return
    fi

    echo ""
    echo "Available GGUF models:"
    echo ""
    printf "  \033[1m%-4s %-50s %s\033[0m\n" "#" "Model" "Size"
    printf "  %-4s %-50s %s\n"               "───" "──────────────────────────────────────────────────" "──────"

    for i in "${!gguf_files[@]}"; do
        local f="${gguf_files[$i]}"
        local name
        name="$(basename "$f")"
        local size_bytes
        size_bytes=$(stat --printf="%s" "$f" 2>/dev/null || stat -f "%z" "$f" 2>/dev/null || echo 0)
        local size_gb
        size_gb=$(awk "BEGIN {printf \"%.1f\", $size_bytes / 1073741824}")
        printf "  %-4s %-50s %sGB\n" "$((i+1))." "$name" "$size_gb"
    done

    echo ""
    read -rp "Select model [1]: " choice
    choice="${choice:-1}"

    if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#gguf_files[@]} )); then
        MODEL_FILE="${gguf_files[$((choice-1))]}"
    else
        echo "Invalid selection, using first model."
        MODEL_FILE="${gguf_files[0]}"
    fi

    echo "  → $(basename "$MODEL_FILE")"
    echo ""
}

# Resolve MODEL_FILE: env override > interactive > default
resolve_model_file() {
    # Allow env override for scripted use
    if [[ -n "${LLAMACPP_MODEL_FILE:-}" ]]; then
        MODEL_FILE="$LLAMACPP_MODEL_FILE"
        return
    fi

    local default_file="${MODEL_DIR}/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-UD-Q4_K_XL.gguf"

    if [[ -t 0 && -d "$MODEL_DIR" ]]; then
        select_gguf_model
    elif [[ -f "$default_file" ]]; then
        MODEL_FILE="$default_file"
    else
        # Try to find any GGUF
        MODEL_FILE=$(find "$MODEL_DIR" -name "*.gguf" 2>/dev/null | head -1)
        if [[ -z "$MODEL_FILE" ]]; then
            echo "No GGUF models found. Run './setup.py install' first."
            exit 1
        fi
    fi

    # Derive alias from filename
    local basename_noext
    basename_noext="$(basename "$MODEL_FILE" .gguf)"
    MODEL_ALIAS="$basename_noext"
}

print_header() {
    local client_display=$(echo "$CLIENT" | tr '[:lower:]' '[:upper:]')
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║  llama.cpp + ${client_display} (Unsloth GLM-4.7-Flash)                              ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"
    echo ""
}

check_prereqs() {
    if [[ ! -x "$LLAMA_SERVER" ]]; then
        echo "Error: llama-server not found at $LLAMA_SERVER"
        echo "Run './setup_llamacpp.sh' first to build llama.cpp"
        exit 1
    fi

    # In router mode, check models directory; in single mode, resolve and check model file
    if [[ "$ROUTER_MODE" == "1" ]]; then
        if [[ ! -d "$MODEL_DIR" ]]; then
            echo "Error: Models directory not found at $MODEL_DIR"
            echo "Run './setup_llamacpp.sh model' to download models"
            exit 1
        fi
        # Check if there are any GGUF files
        local gguf_count=$(find "$MODEL_DIR" -name "*.gguf" 2>/dev/null | wc -l)
        if [[ "$gguf_count" -eq 0 ]]; then
            echo "Error: No GGUF models found in $MODEL_DIR"
            echo "Run './setup_llamacpp.sh model' to download models"
            exit 1
        fi
    else
        # Resolve model file interactively if not set
        if [[ -z "$MODEL_FILE" ]]; then
            resolve_model_file
        fi
        if [[ ! -f "$MODEL_FILE" ]]; then
            echo "Error: Model file not found at $MODEL_FILE"
            echo "Run './setup_llamacpp.sh model' to download the model"
            exit 1
        fi
    fi

    # Validate client choice
    if [[ "$CLIENT" != "opencode" && "$CLIENT" != "crush" ]]; then
        echo "Error: Invalid client '$CLIENT'"
        echo "Set LLAMACPP_CLIENT to 'opencode' or 'crush'"
        exit 1
    fi

    if ! command -v "$CLIENT" &>/dev/null; then
        echo "Error: $CLIENT not found"
        if [[ "$CLIENT" == "opencode" ]]; then
            echo "Install from https://opencode.ai"
        else
            echo "Install from https://github.com/coder/crush"
        fi
        exit 1
    fi
}

is_server_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    # Also check if something is listening on the port
    if lsof -ti :$PORT &>/dev/null; then
        return 0
    fi
    return 1
}

wait_for_server() {
    local max_wait=120
    local waited=0
    echo -n "Waiting for llama-server to be ready"
    while [[ $waited -lt $max_wait ]]; do
        if curl -s "http://localhost:${PORT}/health" | grep -q "ok\|status"; then
            echo " Ready!"
            return 0
        fi
        echo -n "."
        sleep 2
        ((waited+=2))
    done
    echo " Timeout!"
    echo "Server may still be loading the model. Check logs with: ./run_llamacpp.sh logs"
    return 1
}

start_server() {
    if is_server_running; then
        echo "llama-server already running on port $PORT"
        return 0
    fi

    if [[ "$ROUTER_MODE" == "1" ]]; then
        echo "Starting llama-server in ROUTER MODE on port $PORT..."
        echo "Models directory: $MODEL_DIR"
        echo "Max concurrent models: $MODELS_MAX"
        echo "Autoload: $([ "$MODELS_AUTOLOAD" == "1" ] && echo "enabled" || echo "disabled (on-demand)")"
        [[ "$IDLE_TIMEOUT" -gt 0 ]] && echo "Idle timeout: ${IDLE_TIMEOUT}s"
    else
        echo "Starting llama-server in SINGLE MODEL MODE on port $PORT..."
        echo "Model: $MODEL_FILE"
    fi
    echo "Context size: $CTX_SIZE"
    echo ""

    # Build the command based on mode
    local cmd=(
        "$LLAMA_SERVER"
        --host "$HOST"
        --port "$PORT"
        --ctx-size "$CTX_SIZE"
        --jinja
        --temp 1.0
        --top-p 0.95
        --min-p 0.01
        --flash-attn on
        --batch-size 4096
        --ubatch-size 1024
    )

    if [[ "$ROUTER_MODE" == "1" ]]; then
        # Router mode: use --models-dir instead of -m
        cmd+=(
            --models-dir "$MODEL_DIR"
            --models-max "$MODELS_MAX"
        )

        # Configure autoload behavior
        if [[ "$MODELS_AUTOLOAD" == "1" ]]; then
            cmd+=(--models-autoload)
        else
            cmd+=(--no-models-autoload)
        fi

        # Configure idle timeout for automatic unloading
        if [[ "$IDLE_TIMEOUT" -gt 0 ]]; then
            cmd+=(--sleep-idle-seconds "$IDLE_TIMEOUT")
        fi
    else
        # Single model mode: use -m with alias
        cmd+=(
            --model "$MODEL_FILE"
            --alias "$MODEL_ALIAS"
        )
    fi

    # Add GPU-specific options if CUDA is available
    if command -v nvidia-smi &>/dev/null; then
        cmd+=(
            --n-gpu-layers 999
            --cache-type-k q8_0
            --cache-type-v q8_0
        )
    fi

    # Start in background
    nohup "${cmd[@]}" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    echo "Server started with PID $(cat $PID_FILE)"
    echo "Logs: $LOG_FILE"
}

stop_server() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping llama-server (PID $pid)..."
            kill "$pid"
            rm -f "$PID_FILE"
            sleep 2
            echo "Server stopped."
            return 0
        fi
        rm -f "$PID_FILE"
    fi

    # Kill anything on the port
    local port_pid=$(lsof -ti :$PORT 2>/dev/null)
    if [[ -n "$port_pid" ]]; then
        echo "Killing process on port $PORT (PID $port_pid)..."
        kill "$port_pid" 2>/dev/null
        sleep 2
    fi

    echo "Server stopped."
}

show_status() {
    if is_server_running; then
        echo "Status: RUNNING"
        if [[ -f "$PID_FILE" ]]; then
            echo "PID: $(cat $PID_FILE)"
        fi
        echo "Port: $PORT"
        echo "Mode: $([ "$ROUTER_MODE" == "1" ] && echo "Router (multi-model)" || echo "Single model")"

        if [[ "$ROUTER_MODE" == "1" ]]; then
            echo "Models dir: $MODEL_DIR"
            echo "Max concurrent: $MODELS_MAX"
        else
            echo "Model: $MODEL_ALIAS"
        fi

        echo ""
        echo "Health check:"
        curl -s "http://localhost:${PORT}/health" | python3 -m json.tool 2>/dev/null || echo "  (waiting for model load)"

        echo ""
        echo "Loaded models:"
        curl -s "http://localhost:${PORT}/v1/models" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    models = data.get('data', [])
    if models:
        for m in models:
            print(f\"  - {m.get('id', 'unknown')}\")
    else:
        print('  (none loaded)')
except:
    print('  (unable to fetch)')
" 2>/dev/null || echo "  (unable to fetch)"
    else
        echo "Status: STOPPED"
        echo "Mode: $([ "$ROUTER_MODE" == "1" ] && echo "Router (multi-model)" || echo "Single model")"
    fi
}

show_logs() {
    local lines="${1:-50}"
    if [[ -f "$LOG_FILE" ]]; then
        tail -n "$lines" "$LOG_FILE"
    else
        echo "No log file found at $LOG_FILE"
    fi
}

# Router mode: List available and loaded models
list_models() {
    echo "Querying models from llama-server..."
    echo ""

    local response=$(curl -s "http://localhost:${PORT}/v1/models")
    if [[ -z "$response" ]]; then
        echo "Error: Could not connect to server on port $PORT"
        return 1
    fi

    echo "Available models:"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"

    # In router mode, also show model status
    if [[ "$ROUTER_MODE" == "1" ]]; then
        echo ""
        echo "Model load status (router mode):"
        # Try to get extended model info if available
        local status=$(curl -s "http://localhost:${PORT}/models")
        if [[ -n "$status" && "$status" != "null" ]]; then
            echo "$status" | python3 -m json.tool 2>/dev/null || echo "$status"
        else
            echo "  (Extended status not available - models discovered via /v1/models)"
        fi
    fi
}

# Router mode: Load a model by name or path
load_model() {
    local model_name="$1"
    if [[ -z "$model_name" ]]; then
        echo "Usage: $0 load <model-name-or-path>"
        echo ""
        echo "Examples:"
        echo "  $0 load GLM-4.7-Flash-GGUF/GLM-4.7-Flash-UD-Q4_K_XL.gguf"
        echo "  $0 load unsloth/GLM-4.7-Flash"
        return 1
    fi

    echo "Loading model: $model_name"
    local response=$(curl -s -X POST "http://localhost:${PORT}/models/load" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$model_name\"}")

    if [[ -z "$response" ]]; then
        echo "Error: Could not connect to server on port $PORT"
        return 1
    fi

    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
}

# Router mode: Unload a model by name
unload_model() {
    local model_name="$1"
    if [[ -z "$model_name" ]]; then
        echo "Usage: $0 unload <model-name>"
        echo ""
        echo "Use '$0 models' to see loaded models"
        return 1
    fi

    echo "Unloading model: $model_name"
    local response=$(curl -s -X POST "http://localhost:${PORT}/models/unload" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$model_name\"}")

    if [[ -z "$response" ]]; then
        echo "Error: Could not connect to server on port $PORT"
        return 1
    fi

    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
}

run_test() {
    echo "Testing llama-server connection..."
    echo "Mode: $([ "$ROUTER_MODE" == "1" ] && echo "Router (multi-model)" || echo "Single model")"
    echo ""

    # Test /models endpoint
    echo "1. Checking /v1/models endpoint:"
    local models_response=$(curl -s "http://localhost:${PORT}/v1/models")
    echo "$models_response" | python3 -m json.tool 2>/dev/null || echo "  Failed to reach /v1/models"
    echo ""

    # Determine which model to use for testing
    local test_model="$MODEL_ALIAS"
    if [[ "$ROUTER_MODE" == "1" ]]; then
        # In router mode, try to get the first available model
        test_model=$(echo "$models_response" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    models = data.get('data', [])
    if models:
        print(models[0].get('id', ''))
except:
    pass
" 2>/dev/null)
        if [[ -z "$test_model" ]]; then
            echo "Warning: No models available. In router mode, load a model first:"
            echo "  $0 load <model-name>"
            return 1
        fi
        echo "Using model for test: $test_model"
    fi

    # Test a simple completion
    echo "2. Testing chat completion with model '$test_model':"
    curl -s "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$test_model"'",
            "messages": [{"role": "user", "content": "Say hello in exactly 5 words."}],
            "max_tokens": 50
        }' | python3 -m json.tool 2>/dev/null || echo "  Failed to complete chat request"
    echo ""

    # Router mode specific tests
    if [[ "$ROUTER_MODE" == "1" ]]; then
        echo "3. Router mode - checking /models endpoint:"
        curl -s "http://localhost:${PORT}/models" | python3 -m json.tool 2>/dev/null || echo "  (Extended model info not available)"
        echo ""

        echo "4. Testing with $CLIENT (listing models):"
    else
        echo "3. Testing with $CLIENT (listing models):"
    fi

    if [[ "$CLIENT" == "opencode" ]]; then
        opencode models 2>/dev/null | grep -i llama || echo "  llama.cpp provider should show in 'opencode models'"
    else
        # Crush uses different commands - try to list available models
        crush --help 2>/dev/null | head -5 || echo "  Crush CLI ready (use 'crush --help' to see options)"
    fi
}

run_client() {
    print_header
    check_prereqs

    # Start server if not running
    if ! is_server_running; then
        start_server
        wait_for_server || exit 1
    else
        echo "llama-server already running on port $PORT"
    fi

    echo ""
    echo "Launching $CLIENT with GLM-4.7-Flash..."
    echo "Model: llamacpp/$MODEL_ALIAS"
    echo ""

    # Launch the configured client
    if [[ "$CLIENT" == "opencode" ]]; then
        # OpenCode format: provider/model
        opencode -m "llamacpp/$MODEL_ALIAS" "$@"
    else
        # Crush: requires provider configured in ~/.config/crush/crush.json
        # Model format is provider/model-id (e.g., llamacpp/unsloth/GLM-4.7-Flash)
        # Interactive mode doesn't support --model flag; model can be set
        # as default in crush settings or selected in the UI
        crush "$@"
    fi
}

run_client_oneshot() {
    print_header
    check_prereqs

    # Start server if not running
    if ! is_server_running; then
        start_server
        wait_for_server || exit 1
    fi

    local prompt="$*"
    if [[ -z "$prompt" ]]; then
        echo "Usage: $0 run <prompt>"
        exit 1
    fi

    echo "Running one-shot command with $CLIENT and GLM-4.7-Flash..."
    echo ""

    if [[ "$CLIENT" == "opencode" ]]; then
        opencode run -m "llamacpp/$MODEL_ALIAS" "$prompt"
    else
        # Crush one-shot mode: crush run -m <provider/model> <prompt>
        # Model format: llamacpp/unsloth/GLM-4.7-Flash
        crush run -m "llamacpp/$MODEL_ALIAS" "$prompt"
    fi
}

# Legacy aliases for backwards compatibility
run_opencode() {
    CLIENT="opencode"
    run_client "$@"
}

run_crush() {
    CLIENT="crush"
    run_client "$@"
}

# Main command handler
main() {
    case "${1:-help}" in
        start)
            check_prereqs
            start_server
            wait_for_server
            ;;
        stop)
            stop_server
            ;;
        restart)
            stop_server
            sleep 2
            check_prereqs
            start_server
            wait_for_server
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs "${2:-50}"
            ;;
        models)
            list_models
            ;;
        load)
            shift
            load_model "$@"
            ;;
        unload)
            shift
            unload_model "$@"
            ;;
        test)
            check_prereqs
            if ! is_server_running; then
                start_server
                wait_for_server || exit 1
            fi
            run_test
            ;;
        opencode)
            shift
            CLIENT="opencode"
            run_client "$@"
            ;;
        crush)
            shift
            CLIENT="crush"
            run_client "$@"
            ;;
        run)
            shift
            run_client_oneshot "$@"
            ;;
        help|--help|-h)
            print_header
            echo "Usage: $0 <command> [options]"
            echo ""
            echo "Server Modes:"
            echo "  Single model (default): Preloads one model at startup"
            echo "  Router mode: Dynamic multi-model with on-demand loading"
            echo "               Set LLAMACPP_ROUTER_MODE=1 to enable"
            echo ""
            echo "Client Modes:"
            echo "  OpenCode (default): Full-featured coding assistant"
            echo "  Crush: Lightweight alternative client"
            echo "         Set LLAMACPP_CLIENT=crush to use Crush"
            echo ""
            echo "Commands:"
            echo "  opencode     Start server (if needed) and launch OpenCode interactively"
            echo "  crush        Start server (if needed) and launch Crush interactively"
            echo "  run <prompt> Run one-shot command (uses LLAMACPP_CLIENT, default: opencode)"
            echo "  start        Start the llama-server in background"
            echo "  stop         Stop the llama-server"
            echo "  restart      Restart the llama-server"
            echo "  status       Show server status and loaded models"
            echo "  logs [n]     Show last n lines of server log (default: 50)"
            echo "  test         Test server connectivity and client integration"
            echo ""
            echo "Router Mode Commands:"
            echo "  models       List available and loaded models"
            echo "  load <name>  Load a model by name or path"
            echo "  unload <name> Unload a model to free resources"
            echo ""
            echo "Environment Variables (Server):"
            echo "  LLAMACPP_PORT         Server port (default: 10000)"
            echo "  LLAMACPP_HOST         Server host (default: 0.0.0.0)"
            echo "  LLAMACPP_CTX_SIZE     Context size (default: 131072)"
            echo ""
            echo "Environment Variables (Client):"
            echo "  LLAMACPP_CLIENT       Client to use: 'opencode' or 'crush' (default: opencode)"
            echo ""
            echo "Environment Variables (Router Mode):"
            echo "  LLAMACPP_ROUTER_MODE     Set to 1 to enable router mode"
            echo "  LLAMACPP_MODELS_MAX      Max concurrent models (default: 4, LRU eviction)"
            echo "  LLAMACPP_MODELS_AUTOLOAD Set to 0 to disable preloading (default: 1)"
            echo "  LLAMACPP_IDLE_TIMEOUT    Seconds before idle model unloads (default: 300, 0=never)"
            echo ""
            echo "Examples (OpenCode - default):"
            echo "  $0 opencode                    # Interactive OpenCode session"
            echo "  $0 run 'Create hello.py'      # One-shot command with OpenCode"
            echo "  $0 start && $0 test           # Start and verify"
            echo ""
            echo "Examples (Crush):"
            echo "  $0 crush                       # Interactive Crush session"
            echo "  LLAMACPP_CLIENT=crush $0 run 'Create hello.py'  # One-shot with Crush"
            echo ""
            echo "Examples (Router Mode):"
            echo "  LLAMACPP_ROUTER_MODE=1 $0 start              # Start in router mode"
            echo "  $0 models                                     # List available models"
            echo "  $0 load GLM-4.7-Flash-GGUF/model.gguf        # Load a specific model"
            echo "  $0 unload GLM-4.7-Flash                      # Unload to free memory"
            echo ""
            ;;
        *)
            echo "Unknown command: $1"
            echo "Run '$0 help' for usage."
            exit 1
            ;;
    esac
}

main "$@"
