#!/bin/bash
# Setup script for llama.cpp + Unsloth GLM-4.7-Flash model
# Based on: https://unsloth.ai/docs/basics/claude-codex

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_CPP_DIR="${SCRIPT_DIR}/llama.cpp"
MODEL_DIR="${SCRIPT_DIR}/models"

print_step() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# Step 1: Install system dependencies
install_deps() {
    print_step "Step 1: Installing system dependencies"

    if command -v apt-get &>/dev/null; then
        sudo apt-get update
        sudo apt-get install -y pciutils build-essential cmake curl libcurl4-openssl-dev git
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y pciutils gcc-c++ cmake curl libcurl-devel git
    elif command -v brew &>/dev/null; then
        brew install cmake curl git
    else
        echo "Unknown package manager. Please install: cmake, curl, git, build-essential"
        exit 1
    fi

    echo "System dependencies installed."
}

# Step 2: Build llama.cpp
build_llamacpp() {
    print_step "Step 2: Building llama.cpp"

    if [[ -d "$LLAMA_CPP_DIR" ]]; then
        echo "llama.cpp directory exists. Updating..."
        cd "$LLAMA_CPP_DIR"
        git pull
    else
        echo "Cloning llama.cpp..."
        git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
        cd "$LLAMA_CPP_DIR"
    fi

    # Detect CUDA
    CUDA_FLAG="-DGGML_CUDA=OFF"
    if command -v nvcc &>/dev/null || [[ -d "/usr/local/cuda" ]]; then
        echo "CUDA detected. Building with GPU support."
        CUDA_FLAG="-DGGML_CUDA=ON"
    else
        echo "No CUDA detected. Building CPU-only version."
    fi

    # Build
    cmake . -B build \
        -DBUILD_SHARED_LIBS=OFF \
        $CUDA_FLAG

    cmake --build build --config Release -j$(nproc) --clean-first \
        --target llama-cli llama-server llama-gguf-split

    # Copy binaries to llama.cpp root
    cp build/bin/llama-* . 2>/dev/null || true

    echo "llama.cpp built successfully."
    ./llama-server --version 2>/dev/null || echo "llama-server ready"
}

# Step 3: Download the GLM-4.7-Flash model
download_model() {
    print_step "Step 3: Downloading GLM-4.7-Flash model (Unsloth Q4_K_XL quantization)"

    mkdir -p "$MODEL_DIR"

    # Install Python dependencies if needed
    pip install --quiet huggingface_hub hf_transfer

    # Download model using Python
    python3 << 'EOF'
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download

print("Downloading GLM-4.7-Flash-GGUF (Q4_K_XL variant)...")
print("This may take a while depending on your connection...")

snapshot_download(
    repo_id="unsloth/GLM-4.7-Flash-GGUF",
    local_dir="models/GLM-4.7-Flash-GGUF",
    allow_patterns=["*UD-Q4_K_XL*", "*.md", "*.json"],
)

print("Model downloaded successfully!")
EOF

    # Verify the model file exists
    MODEL_FILE="${MODEL_DIR}/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-UD-Q4_K_XL.gguf"
    if [[ -f "$MODEL_FILE" ]]; then
        echo "Model file verified: $MODEL_FILE"
        ls -lh "$MODEL_FILE"
    else
        echo "Warning: Expected model file not found at $MODEL_FILE"
        echo "Checking available files..."
        ls -la "$MODEL_DIR/GLM-4.7-Flash-GGUF/" 2>/dev/null || true
    fi
}

# Step 4: Configure OpenCode
configure_opencode() {
    print_step "Step 4: Configuring OpenCode for llama.cpp backend"

    OPENCODE_CONFIG="${HOME}/.config/opencode/opencode.json"
    mkdir -p "$(dirname "$OPENCODE_CONFIG")"

    # Check if config exists
    if [[ -f "$OPENCODE_CONFIG" ]]; then
        echo "Existing OpenCode config found. Adding llama.cpp provider..."

        # Use Python to merge config
        python3 << EOF
import json

config_path = "$OPENCODE_CONFIG"

with open(config_path, 'r') as f:
    config = json.load(f)

# Add llama.cpp provider
if 'provider' not in config:
    config['provider'] = {}

config['provider']['llamacpp'] = {
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

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print("OpenCode config updated with llama.cpp provider.")
EOF
    else
        echo "Creating new OpenCode config..."
        cat > "$OPENCODE_CONFIG" << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
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
EOF
    fi

    echo "OpenCode config at: $OPENCODE_CONFIG"
    cat "$OPENCODE_CONFIG"
}

# Main
main() {
    cd "$SCRIPT_DIR"

    case "${1:-all}" in
        deps)
            install_deps
            ;;
        build)
            build_llamacpp
            ;;
        model)
            download_model
            ;;
        config)
            configure_opencode
            ;;
        all)
            install_deps
            build_llamacpp
            download_model
            configure_opencode
            print_step "Setup Complete!"
            echo "To start the llama.cpp server and OpenCode, run:"
            echo ""
            echo "  ./run_llamacpp.sh opencode"
            echo ""
            echo "Or for a quick test:"
            echo ""
            echo "  ./run_llamacpp.sh test"
            echo ""
            ;;
        *)
            echo "Usage: $0 [deps|build|model|config|all]"
            echo ""
            echo "  deps   - Install system dependencies"
            echo "  build  - Build llama.cpp from source"
            echo "  model  - Download GLM-4.7-Flash model"
            echo "  config - Configure OpenCode"
            echo "  all    - Run all steps (default)"
            exit 1
            ;;
    esac
}

main "$@"
