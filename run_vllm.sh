#!/bin/bash
set -euo pipefail

# Defaults
MODEL="cyankiwi/GLM-4.7-Flash-AWQ-4bit"
PORT=8000
GPU_UTIL=0.92
MAX_LEN=19136
TOOL_PARSER="glm47"
REASONING_PARSER="glm45"
THINKING=true
ACTION="serve"  # serve | stop | status | test | opencode | crush | logs

VLLM_LOG="/tmp/vllm-server.log"
VLLM_PID="/tmp/vllm-server.pid"
HEALTH_TIMEOUT=120  # seconds to wait for vLLM to become healthy

usage() {
    cat <<EOF
Usage: ./run_vllm.sh [command] [options]

Commands:
  serve      Start the vLLM server in the foreground (default)
  stop       Kill the running vLLM server
  status     Check if the server is up and which model is loaded
  test       Run a quick tool-call smoke test
  opencode   Launch OpenCode TUI (auto-starts vLLM in background if needed)
  crush      Launch Crush TUI (auto-starts vLLM in background if needed)
  logs       Tail the background vLLM server log

Options:
  --model <id>       HuggingFace model ID        (default: $MODEL)
  --port <n>         Server port                  (default: $PORT)
  --gpu-util <0-1>   GPU memory utilization       (default: $GPU_UTIL)
  --max-len <n>      Max context length           (default: $MAX_LEN)
  --parser <name>    Tool call parser             (default: $TOOL_PARSER)
  --reasoning-parser Reasoning parser             (default: $REASONING_PARSER)
  --no-thinking      Disable thinking mode
  -h, --help         Show this help

Examples:
  ./run_vllm.sh                          # start server in foreground
  ./run_vllm.sh crush                    # start vLLM in background + launch Crush
  ./run_vllm.sh opencode                 # start vLLM in background + launch OpenCode
  ./run_vllm.sh crush --no-thinking      # background vLLM (no thinking) + Crush
  ./run_vllm.sh stop                     # kill the server
  ./run_vllm.sh logs                     # tail background server log
  ./run_vllm.sh serve --model Qwen/Qwen3-8B-AWQ --max-len 32768
EOF
    exit 0
}

# --- Parse args ---
if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
    ACTION="$1"; shift
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)       MODEL="$2";      shift ;;
        --port)        PORT="$2";       shift ;;
        --gpu-util)    GPU_UTIL="$2";   shift ;;
        --max-len)     MAX_LEN="$2";    shift ;;
        --parser)            TOOL_PARSER="$2";      shift ;;
        --reasoning-parser)  REASONING_PARSER="$2"; shift ;;
        --no-thinking)       THINKING=false                ;;
        -h|--help)     usage                   ;;
        *) echo "Unknown option: $1"; usage    ;;
    esac
    shift
done

# --- Build vLLM args (shared by foreground and background) ---
build_vllm_args() {
    VLLM_ARGS=(
        vllm serve "$MODEL"
        --host 0.0.0.0
        --port "$PORT"
        --dtype auto
        --gpu-memory-utilization "$GPU_UTIL"
        --max-model-len "$MAX_LEN"
        --enable-auto-tool-choice
        --tool-call-parser "$TOOL_PARSER"
        --reasoning-parser "$REASONING_PARSER"
    )

    if [[ "$THINKING" == false ]]; then
        VLLM_ARGS+=(--default-chat-template-kwargs '{"enable_thinking": false}')
    fi
}

# --- Ensure vLLM is running (start in background if needed) ---
ensure_server() {
    # Already running? Just verify health.
    if lsof -ti :"$PORT" &>/dev/null; then
        if curl -sf "http://localhost:$PORT/v1/models" &>/dev/null; then
            echo "vLLM server already running and healthy on port $PORT."
            return 0
        fi
        echo "Port $PORT in use but /v1/models not responding — waiting..."
        wait_for_health
        return 0
    fi

    echo "Starting vLLM server in background..."
    echo "  Model:    $MODEL"
    echo "  Port:     $PORT"
    echo "  Context:  $MAX_LEN tokens"
    echo "  Thinking: $THINKING"
    echo "  Log:      $VLLM_LOG"
    echo ""

    build_vllm_args

    echo "Run (background):"
    echo "  ${VLLM_ARGS[*]}"
    echo ""

    # Launch in background, redirect output to log
    nohup "${VLLM_ARGS[@]}" > "$VLLM_LOG" 2>&1 &
    local pid=$!
    echo "$pid" > "$VLLM_PID"
    echo "vLLM PID: $pid"
    echo ""

    # Brief pause to let the process start (or fail immediately)
    sleep 2
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "ERROR: vLLM process exited immediately. Check log:"
        echo "  tail -50 $VLLM_LOG"
        exit 1
    fi

    wait_for_health
}

wait_for_health() {
    echo "Waiting for vLLM to become healthy (up to ${HEALTH_TIMEOUT}s)..."
    local elapsed=0
    local interval=3
    while [[ $elapsed -lt $HEALTH_TIMEOUT ]]; do
        if curl -sf "http://localhost:$PORT/v1/models" &>/dev/null; then
            echo "vLLM is ready!"
            echo ""
            return 0
        fi
        # Show a progress dot
        printf "."
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    echo ""
    echo "ERROR: vLLM did not become healthy within ${HEALTH_TIMEOUT}s."
    echo "Check log:  tail -100 $VLLM_LOG"
    exit 1
}

# --- Commands ---

do_serve() {
    # Kill anything already on the port
    if lsof -ti :"$PORT" &>/dev/null; then
        echo "Port $PORT in use — stopping existing process..."
        lsof -ti :"$PORT" | xargs kill 2>/dev/null || true
        sleep 2
    fi

    echo "Starting vLLM server (foreground)..."
    echo "  Model:    $MODEL"
    echo "  Port:     $PORT"
    echo "  Context:  $MAX_LEN tokens"
    echo "  Thinking: $THINKING"
    echo ""

    build_vllm_args

    echo "Run:"
    echo "  ${VLLM_ARGS[*]}"
    echo ""

    exec "${VLLM_ARGS[@]}"
}

do_stop() {
    local stopped=false

    # Kill by PID file first
    if [[ -f "$VLLM_PID" ]]; then
        local pid
        pid=$(cat "$VLLM_PID")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping vLLM (PID $pid)..."
            kill "$pid"
            stopped=true
        fi
        rm -f "$VLLM_PID"
    fi

    # Also kill anything on the port
    if lsof -ti :"$PORT" &>/dev/null; then
        echo "Stopping process on port $PORT..."
        lsof -ti :"$PORT" | xargs kill 2>/dev/null || true
        stopped=true
    fi

    if $stopped; then
        echo "Done."
    else
        echo "Nothing running on port $PORT."
    fi
}

do_status() {
    if ! lsof -ti :"$PORT" &>/dev/null; then
        echo "No server running on port $PORT."
        return 1
    fi
    echo "vLLM server is running on port $PORT."
    if [[ -f "$VLLM_PID" ]]; then
        echo "  PID: $(cat "$VLLM_PID")"
    fi
    if [[ -f "$VLLM_LOG" ]]; then
        echo "  Log: $VLLM_LOG"
    fi
    echo ""
    curl -s "http://localhost:$PORT/v1/models" | python3 -m json.tool 2>/dev/null \
        || echo "(could not reach /v1/models)"
}

do_test() {
    echo "Running tool-call smoke test against localhost:$PORT ..."
    python3 -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:$PORT/v1', api_key='sk-EMPTY')
r = client.chat.completions.create(
    model='$MODEL',
    messages=[{'role':'user','content':'Create a file called test.txt with hello inside. Use tools.'}],
    tools=[{'type':'function','function':{
        'name':'write_file','description':'Write content to a file',
        'parameters':{'type':'object','properties':{
            'filename':{'type':'string'},'content':{'type':'string'}},
            'required':['filename','content']}}}],
    tool_choice='auto'
)
msg = r.choices[0].message
if msg.tool_calls:
    for tc in msg.tool_calls:
        print(f'Tool: {tc.function.name}')
        print(f'Args: {tc.function.arguments}')
else:
    print('No tool calls returned.')
    print(f'Content: {msg.content[:200]}')
"
}

do_logs() {
    if [[ -f "$VLLM_LOG" ]]; then
        echo "Tailing $VLLM_LOG (Ctrl+C to stop)..."
        echo ""
        tail -f "$VLLM_LOG"
    else
        echo "No log file found at $VLLM_LOG."
        echo "The server may not have been started in background mode."
    fi
}

do_opencode() {
    ensure_server
    echo "Launching OpenCode (model: vllm/$MODEL) ..."
    exec opencode
}

do_crush() {
    if ! command -v crush &>/dev/null; then
        echo "Error: 'crush' is not installed."
        echo "Install it:  ./run_crush.sh install"
        echo "         or: npm install -g @charmland/crush"
        exit 1
    fi

    ensure_server

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo "Launching Crush (model: vllm/$MODEL on port $PORT) ..."
    exec "$SCRIPT_DIR/run_crush.sh" --backend vllm --model "$MODEL" --vllm-host "http://localhost:$PORT"
}

# --- Dispatch ---
case "$ACTION" in
    serve)    do_serve    ;;
    stop)     do_stop     ;;
    status)   do_status   ;;
    test)     do_test     ;;
    logs)     do_logs     ;;
    opencode) do_opencode ;;
    crush)    do_crush    ;;
    *)        echo "Unknown command: $ACTION"; usage ;;
esac
