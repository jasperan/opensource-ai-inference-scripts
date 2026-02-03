#!/usr/bin/env python3
"""
Unified AI Inference Setup

Single entry point for installing and running local coding agents.
Supports llama.cpp, vLLM, and Ollama backends with OpenCode and Crush clients.

Usage:
    python setup.py              # Interactive TUI (default)
    python setup.py install      # Install backend + model + client
    python setup.py run          # Select and launch
    python setup.py status       # Show installation status
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import questionary
    from questionary import Style
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install questionary rich")
    sys.exit(1)

console = Console()
SCRIPT_DIR = Path(__file__).parent.resolve()

# Questionary style matching Rich aesthetic
STYLE = Style([
    ("qmark", "fg:cyan bold"),
    ("question", "fg:white bold"),
    ("answer", "fg:green bold"),
    ("pointer", "fg:cyan bold"),
    ("highlighted", "fg:cyan bold"),
    ("selected", "fg:green"),
    ("separator", "fg:gray"),
    ("instruction", "fg:gray italic"),
])

# =============================================================================
# Configuration Data
# =============================================================================

BACKENDS = {
    "llamacpp": {
        "name": "llama.cpp",
        "description": "Fast, 128K context, builds from source",
        "recommended": True,
        "port": 10000,
        "run_script": "run_llamacpp.sh",
        "check_paths": ["llama.cpp/llama-server"],
        "models": [
            {
                "id": "GLM-4.7-Flash-Q4_K_XL",
                "name": "GLM-4.7-Flash (Q4_K_XL)",
                "repo": "unsloth/GLM-4.7-Flash-GGUF",
                "file_pattern": "*Q4_K_XL*",
                "size_gb": 17,
                "context": 131072,
                "recommended": True,
            },
            {
                "id": "GLM-4.7-Flash-Q8_0",
                "name": "GLM-4.7-Flash (Q8_0)",
                "repo": "unsloth/GLM-4.7-Flash-GGUF",
                "file_pattern": "*Q8_0*",
                "size_gb": 30,
                "context": 131072,
            },
        ],
    },
    "vllm": {
        "name": "vLLM",
        "description": "Multi-model, requires more VRAM",
        "port": 8000,
        "run_script": "run_vllm.sh",
        "check_cmd": ["python", "-c", "import vllm"],
        "models": [
            {
                "id": "cyankiwi/GLM-4.7-Flash-AWQ-4bit",
                "name": "GLM-4.7-Flash AWQ",
                "size_gb": 10,
                "context": 19136,
                "recommended": True,
            },
            {
                "id": "Qwen/Qwen3-8B",
                "name": "Qwen3-8B",
                "size_gb": 16,
                "context": 16384,
            },
            {
                "id": "Qwen/Qwen3-8B-AWQ",
                "name": "Qwen3-8B AWQ",
                "size_gb": 5,
                "context": 32768,
            },
        ],
    },
    "ollama": {
        "name": "Ollama",
        "description": "Easiest, pre-built binaries",
        "port": 11434,
        "run_script": "run_crush.sh",
        "check_cmd": ["ollama", "--version"],
        "models": [
            {
                "id": "glm-4.7-flash:latest",
                "name": "GLM-4.7 Flash",
                "size_gb": 9,
                "context": 131072,
                "recommended": True,
            },
            {
                "id": "qwen3-coder:latest",
                "name": "Qwen3 Coder",
                "size_gb": 5,
                "context": 131072,
            },
            {
                "id": "gpt-oss:20b",
                "name": "GPT-OSS 20B",
                "size_gb": 12,
                "context": 32768,
            },
            {
                "id": "nemotron-3-nano:30b",
                "name": "Nemotron-3-Nano 30B",
                "size_gb": 18,
                "context": 32768,
            },
        ],
    },
}

CLIENTS = {
    "opencode": {
        "name": "OpenCode",
        "description": "Full-featured coding assistant TUI",
        "check_cmd": ["opencode", "--version"],
        "install_methods": [
            {"name": "npm", "check": "npm", "cmd": "npm install -g @anthropic-ai/opencode"},
            {"name": "curl", "check": "curl", "cmd": "curl -fsSL https://opencode.ai/install.sh | sh"},
        ],
    },
    "crush": {
        "name": "Crush",
        "description": "Charm's lightweight AI chat TUI",
        "check_cmd": ["crush", "--version"],
        "install_methods": [
            {"name": "npm", "check": "npm", "cmd": "npm install -g @charmland/crush"},
            {"name": "brew", "check": "brew", "cmd": "brew install charmbracelet/tap/crush"},
            {"name": "go", "check": "go", "cmd": "go install github.com/charmbracelet/crush@latest"},
        ],
    },
}

# =============================================================================
# Status Detection
# =============================================================================

def command_exists(cmd: str) -> bool:
    """Check if a command exists in PATH."""
    return shutil.which(cmd) is not None


def check_port(port: int) -> bool:
    """Check if something is listening on a port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            timeout=2,
        )
        return result.returncode == 0
    except Exception:
        return False


def fetch_url_json(url: str, timeout: float = 3.0) -> dict | None:
    """Fetch JSON from URL using curl."""
    try:
        result = subprocess.run(
            ["curl", "-sf", "--max-time", str(timeout), url],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return None


def is_backend_installed(backend_key: str) -> bool:
    """Check if a backend is installed."""
    cfg = BACKENDS[backend_key]

    # Check paths (for llama.cpp)
    if "check_paths" in cfg:
        for path in cfg["check_paths"]:
            if (SCRIPT_DIR / path).exists():
                return True
        return False

    # Check command (for vllm, ollama)
    if "check_cmd" in cfg:
        try:
            result = subprocess.run(
                cfg["check_cmd"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    return False


def is_backend_running(backend_key: str) -> bool:
    """Check if a backend server is running."""
    cfg = BACKENDS[backend_key]
    return check_port(cfg["port"])


def get_installed_models(backend_key: str) -> list[dict]:
    """Get list of installed/available models for a backend."""
    cfg = BACKENDS[backend_key]
    models = []

    if backend_key == "llamacpp":
        # Check local models directory
        models_dir = SCRIPT_DIR / "models"
        if models_dir.exists():
            for gguf in models_dir.rglob("*.gguf"):
                size_gb = gguf.stat().st_size / (1024**3)
                models.append({
                    "id": str(gguf.relative_to(models_dir)),
                    "name": gguf.stem,
                    "size_gb": size_gb,
                    "path": str(gguf),
                })

    elif backend_key == "ollama":
        # Query Ollama API or CLI
        data = fetch_url_json("http://localhost:11434/api/tags")
        if data and "models" in data:
            for m in data["models"]:
                size_gb = m.get("size", 0) / (1024**3)
                models.append({
                    "id": m.get("name", ""),
                    "name": m.get("name", ""),
                    "size_gb": size_gb,
                })
        else:
            # Try CLI fallback
            try:
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")[1:]
                    for line in lines:
                        parts = line.split()
                        if parts:
                            models.append({
                                "id": parts[0],
                                "name": parts[0],
                                "size_gb": 0,
                            })
            except Exception:
                pass

    elif backend_key == "vllm":
        # Query vLLM server
        data = fetch_url_json("http://localhost:8000/v1/models")
        if data and "data" in data:
            for m in data["data"]:
                models.append({
                    "id": m.get("id", ""),
                    "name": m.get("id", ""),
                    "size_gb": 0,
                })

    return models


def is_client_installed(client_key: str) -> bool:
    """Check if a client is installed."""
    cfg = CLIENTS[client_key]
    try:
        result = subprocess.run(
            cfg["check_cmd"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_full_status() -> dict:
    """Get complete installation and running status."""
    status = {
        "backends": {},
        "clients": {},
    }

    for key, cfg in BACKENDS.items():
        installed = is_backend_installed(key)
        running = is_backend_running(key)
        models = get_installed_models(key) if installed or running else []

        status["backends"][key] = {
            "name": cfg["name"],
            "description": cfg["description"],
            "installed": installed,
            "running": running,
            "models": models,
            "port": cfg["port"],
            "recommended": cfg.get("recommended", False),
        }

    for key, cfg in CLIENTS.items():
        status["clients"][key] = {
            "name": cfg["name"],
            "description": cfg["description"],
            "installed": is_client_installed(key),
        }

    return status


# =============================================================================
# UI Components
# =============================================================================

def print_header():
    """Print the launcher header."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]AI Inference Setup[/bold cyan]\n"
        "[dim]One-click setup for local coding agents[/dim]",
        border_style="cyan",
    ))
    console.print()


def print_status_table(status: dict):
    """Print installation status as tables."""
    # Backends table
    table = Table(
        title="[bold]Backends[/bold]",
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Backend", style="white")
    table.add_column("Status", style="white")
    table.add_column("Models", style="white")
    table.add_column("Port", style="dim")

    for key, info in status["backends"].items():
        if info["installed"]:
            if info["running"]:
                status_str = "[green]● running[/green]"
            else:
                status_str = "[yellow]● installed[/yellow]"
        else:
            status_str = "[dim]○ not installed[/dim]"

        model_count = len(info["models"])
        if model_count:
            models_str = f"{model_count} model{'s' if model_count != 1 else ''}"
        else:
            models_str = "[dim]—[/dim]"

        name = info["name"]
        if info.get("recommended"):
            name += " [cyan](recommended)[/cyan]"

        table.add_row(name, status_str, models_str, str(info["port"]))

    console.print(table)
    console.print()

    # Clients table
    table = Table(
        title="[bold]Clients[/bold]",
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Client", style="white")
    table.add_column("Status", style="white")
    table.add_column("Description", style="dim")

    for key, info in status["clients"].items():
        if info["installed"]:
            status_str = "[green]● installed[/green]"
        else:
            status_str = "[dim]○ not installed[/dim]"

        table.add_row(info["name"], status_str, info["description"])

    console.print(table)
    console.print()


# =============================================================================
# Installation Functions
# =============================================================================

def install_system_deps():
    """Install system dependencies."""
    console.print("[cyan]Installing system dependencies...[/cyan]")

    if command_exists("apt-get"):
        cmd = ["sudo", "apt-get", "update"]
        subprocess.run(cmd, check=True)
        cmd = ["sudo", "apt-get", "install", "-y",
               "pciutils", "build-essential", "cmake", "curl",
               "libcurl4-openssl-dev", "git"]
        subprocess.run(cmd, check=True)
    elif command_exists("dnf"):
        cmd = ["sudo", "dnf", "install", "-y",
               "pciutils", "gcc-c++", "cmake", "curl", "libcurl-devel", "git"]
        subprocess.run(cmd, check=True)
    elif command_exists("brew"):
        cmd = ["brew", "install", "cmake", "curl", "git"]
        subprocess.run(cmd, check=True)
    else:
        console.print("[yellow]Unknown package manager. Please install manually:[/yellow]")
        console.print("  cmake, curl, git, build-essential")
        return False

    return True


def install_llamacpp(progress_callback=None):
    """Install llama.cpp backend."""
    llama_dir = SCRIPT_DIR / "llama.cpp"

    # Clone or update
    if llama_dir.exists():
        console.print("[cyan]Updating llama.cpp...[/cyan]")
        subprocess.run(["git", "pull"], cwd=llama_dir, check=True)
    else:
        console.print("[cyan]Cloning llama.cpp...[/cyan]")
        subprocess.run([
            "git", "clone", "https://github.com/ggml-org/llama.cpp",
            str(llama_dir)
        ], check=True)

    # Detect CUDA
    cuda_flag = "-DGGML_CUDA=OFF"
    if command_exists("nvcc") or Path("/usr/local/cuda").exists():
        console.print("[green]CUDA detected. Building with GPU support.[/green]")
        cuda_flag = "-DGGML_CUDA=ON"
    else:
        console.print("[yellow]No CUDA detected. Building CPU-only version.[/yellow]")

    # Build
    console.print("[cyan]Building llama.cpp...[/cyan]")
    build_dir = llama_dir / "build"

    subprocess.run([
        "cmake", ".", "-B", "build",
        "-DBUILD_SHARED_LIBS=OFF",
        cuda_flag,
    ], cwd=llama_dir, check=True)

    subprocess.run([
        "cmake", "--build", "build", "--config", "Release",
        f"-j{os.cpu_count()}",
        "--target", "llama-cli", "llama-server", "llama-gguf-split",
    ], cwd=llama_dir, check=True)

    # Copy binaries
    for binary in (build_dir / "bin").glob("llama-*"):
        shutil.copy2(binary, llama_dir)

    console.print("[green]✓ llama.cpp built successfully[/green]")
    return True


def install_vllm():
    """Install vLLM backend."""
    console.print("[cyan]Installing vLLM...[/cyan]")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "vllm", "--upgrade"
    ], check=True)
    console.print("[green]✓ vLLM installed successfully[/green]")
    return True


def install_ollama():
    """Install Ollama backend."""
    console.print("[cyan]Installing Ollama...[/cyan]")

    if command_exists("ollama"):
        console.print("[green]✓ Ollama already installed[/green]")
        return True

    # Use official install script
    subprocess.run([
        "curl", "-fsSL", "https://ollama.com/install.sh"
    ], check=True)

    console.print("[green]✓ Ollama installed successfully[/green]")
    return True


def download_llamacpp_model(model_cfg: dict):
    """Download a model for llama.cpp."""
    console.print(f"[cyan]Downloading {model_cfg['name']}...[/cyan]")
    console.print(f"[dim]Repository: {model_cfg['repo']}[/dim]")
    console.print(f"[dim]Size: ~{model_cfg['size_gb']}GB[/dim]")

    # Ensure huggingface_hub is installed
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--quiet",
        "huggingface_hub", "hf_transfer"
    ], check=True)

    models_dir = SCRIPT_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    # Download using Python
    download_script = f'''
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="{model_cfg['repo']}",
    local_dir="models/{model_cfg['repo'].split('/')[-1]}",
    allow_patterns=["{model_cfg['file_pattern']}", "*.md", "*.json"],
)
'''
    subprocess.run([sys.executable, "-c", download_script], cwd=SCRIPT_DIR, check=True)

    console.print(f"[green]✓ {model_cfg['name']} downloaded successfully[/green]")
    return True


def download_ollama_model(model_id: str):
    """Download/pull an Ollama model."""
    console.print(f"[cyan]Pulling {model_id}...[/cyan]")
    subprocess.run(["ollama", "pull", model_id], check=True)
    console.print(f"[green]✓ {model_id} pulled successfully[/green]")
    return True


def install_client(client_key: str):
    """Install a client."""
    cfg = CLIENTS[client_key]
    console.print(f"[cyan]Installing {cfg['name']}...[/cyan]")

    for method in cfg["install_methods"]:
        if command_exists(method["check"]):
            console.print(f"[dim]Using {method['name']}...[/dim]")
            try:
                subprocess.run(method["cmd"], shell=True, check=True)
                console.print(f"[green]✓ {cfg['name']} installed successfully[/green]")
                return True
            except subprocess.CalledProcessError:
                console.print(f"[yellow]Failed with {method['name']}, trying next...[/yellow]")
                continue

    console.print(f"[red]✗ Could not install {cfg['name']}[/red]")
    console.print("[dim]No suitable package manager found (npm, brew, go)[/dim]")
    return False


def configure_client_for_backend(client_key: str, backend_key: str, model_id: str):
    """Configure client for a specific backend."""
    console.print(f"[cyan]Configuring {CLIENTS[client_key]['name']} for {BACKENDS[backend_key]['name']}...[/cyan]")

    if client_key == "crush":
        # Write crush config
        config_dir = Path.home() / ".config" / "crush"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "crush.json"

        port = BACKENDS[backend_key]["port"]

        config = {
            "$schema": "https://charm.land/crush.json",
            "providers": {
                backend_key: {
                    "name": f"{BACKENDS[backend_key]['name']} (local)",
                    "base_url": f"http://localhost:{port}/v1/",
                    "type": "openai-compat",
                    "models": [
                        {
                            "name": model_id,
                            "id": model_id,
                            "context_window": 131072,
                            "default_max_tokens": 8192,
                        }
                    ]
                }
            },
            "lsp": {
                "pylsp": {"command": "pylsp", "enabled": True},
                "bash-language-server": {
                    "command": "bash-language-server",
                    "args": ["start"],
                    "enabled": True,
                }
            }
        }

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        console.print(f"[green]✓ Crush config written to {config_file}[/green]")

    elif client_key == "opencode":
        # Write opencode config
        config_dir = Path.home() / ".config" / "opencode"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "opencode.json"

        port = BACKENDS[backend_key]["port"]

        config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                backend_key: {
                    "name": f"{BACKENDS[backend_key]['name']} (local)",
                    "npm": "@ai-sdk/openai-compatible",
                    "models": {
                        model_id: {
                            "name": f"{model_id} [{BACKENDS[backend_key]['name']}]"
                        }
                    },
                    "options": {
                        "baseURL": f"http://localhost:{port}/v1",
                        "apiKey": "sk-EMPTY"
                    }
                }
            }
        }

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        console.print(f"[green]✓ OpenCode config written to {config_file}[/green]")

    return True


# =============================================================================
# TUI Flows
# =============================================================================

def select_main_action(status: dict) -> str | None:
    """Main menu selection."""
    choices = [
        questionary.Choice(
            title="Install a backend",
            value="install_backend",
        ),
        questionary.Choice(
            title="Install a client",
            value="install_client",
        ),
        questionary.Choice(
            title="Run (select backend + client)",
            value="run",
        ),
        questionary.Choice(
            title="Exit",
            value=None,
        ),
    ]

    return questionary.select(
        "What would you like to do?",
        choices=choices,
        style=STYLE,
        qmark="",
        pointer=">",
    ).ask()


def select_backend_to_install(status: dict) -> str | None:
    """Select backend to install."""
    choices = []

    for key, cfg in BACKENDS.items():
        info = status["backends"][key]

        if info["installed"]:
            label = f"{cfg['name']:12} [green]● installed[/green]"
        else:
            rec = " [cyan](Recommended)[/cyan]" if cfg.get("recommended") else ""
            label = f"{cfg['name']:12} - {cfg['description']}{rec}"

        choices.append(questionary.Choice(title=label, value=key))

    choices.append(questionary.Choice(title="← Back", value=None))

    return questionary.select(
        "Select backend to install:",
        choices=choices,
        style=STYLE,
        qmark="",
        pointer=">",
    ).ask()


def select_model_to_download(backend_key: str) -> dict | None:
    """Select model to download for a backend."""
    models = BACKENDS[backend_key]["models"]
    choices = []

    for m in models:
        rec = " [cyan](Recommended)[/cyan]" if m.get("recommended") else ""
        ctx = f"{m['context'] // 1024}K" if m.get("context") else ""
        label = f"{m['name']:30} {m['size_gb']:>3}GB  {ctx:>6} context{rec}"
        choices.append(questionary.Choice(title=label, value=m))

    choices.append(questionary.Choice(title="← Back", value=None))

    return questionary.select(
        "Select model to download:",
        choices=choices,
        style=STYLE,
        qmark="",
        pointer=">",
    ).ask()


def select_client_to_install(status: dict) -> str | None:
    """Select client to install."""
    choices = []

    for key, cfg in CLIENTS.items():
        info = status["clients"][key]

        if info["installed"]:
            label = f"{cfg['name']:12} [green]● installed[/green]"
        else:
            label = f"{cfg['name']:12} - {cfg['description']}"

        choices.append(questionary.Choice(title=label, value=key))

    choices.append(questionary.Choice(
        title="Both         - Install both clients",
        value="both",
    ))
    choices.append(questionary.Choice(title="← Back", value=None))

    return questionary.select(
        "Select client to install:",
        choices=choices,
        style=STYLE,
        qmark="",
        pointer=">",
    ).ask()


def select_backend_to_run(status: dict) -> str | None:
    """Select backend for running."""
    choices = []

    for key, info in status["backends"].items():
        if info["running"]:
            status_icon = "[green]● running[/green]"
        elif info["installed"]:
            status_icon = "[yellow]● stopped[/yellow]"
        else:
            status_icon = "[dim]○ not installed[/dim]"

        model_count = len(info["models"])
        models_str = f"({model_count} models)" if model_count else ""

        label = f"{info['name']:12} {status_icon}  {models_str}"

        # Disable if not installed or no models
        disabled = None
        if not info["installed"]:
            disabled = "not installed"
        elif not info["models"] and not info["running"]:
            disabled = "no models"

        choices.append(questionary.Choice(title=label, value=key, disabled=disabled))

    choices.append(questionary.Choice(title="← Back", value=None))

    return questionary.select(
        "Select backend:",
        choices=choices,
        style=STYLE,
        qmark="",
        pointer=">",
    ).ask()


def select_model_to_run(backend_key: str, status: dict) -> str | None:
    """Select model to use."""
    models = status["backends"][backend_key]["models"]
    choices = []

    for m in models:
        size_str = f"[{m['size_gb']:.1f}GB]" if m.get("size_gb") else ""
        label = f"{m['name']:40} {size_str}"
        choices.append(questionary.Choice(title=label, value=m["id"]))

    # Also show available models from config that aren't installed
    for cfg_model in BACKENDS[backend_key]["models"]:
        if not any(m["id"] == cfg_model["id"] or cfg_model["name"] in m.get("name", "") for m in models):
            label = f"{cfg_model['name']:40} [dim](not downloaded)[/dim]"
            choices.append(questionary.Choice(title=label, value=cfg_model["id"], disabled="not downloaded"))

    choices.append(questionary.Choice(title="← Back", value=None))

    return questionary.select(
        "Select model:",
        choices=choices,
        style=STYLE,
        qmark="",
        pointer=">",
    ).ask()


def select_client_to_run(status: dict) -> str | None:
    """Select client to run."""
    choices = []

    for key, info in status["clients"].items():
        if info["installed"]:
            label = f"{info['name']:12} - {info['description']}"
            choices.append(questionary.Choice(title=label, value=key))
        else:
            label = f"{info['name']:12} [dim](not installed)[/dim]"
            choices.append(questionary.Choice(title=label, value=key, disabled="not installed"))

    choices.append(questionary.Choice(title="← Back", value=None))

    return questionary.select(
        "Select client:",
        choices=choices,
        style=STYLE,
        qmark="",
        pointer=">",
    ).ask()


# =============================================================================
# Launch Function
# =============================================================================

def launch(backend_key: str, model_id: str, client_key: str):
    """Launch the selected configuration."""
    console.print()
    console.print("[bold green]Launching:[/bold green]")
    console.print(f"  Backend:  [cyan]{BACKENDS[backend_key]['name']}[/cyan]")
    console.print(f"  Model:    [cyan]{model_id}[/cyan]")
    console.print(f"  Client:   [cyan]{CLIENTS[client_key]['name']}[/cyan]")
    console.print()

    # Configure client
    configure_client_for_backend(client_key, backend_key, model_id)

    # Build launch command
    script = SCRIPT_DIR / BACKENDS[backend_key]["run_script"]

    if backend_key == "llamacpp":
        cmd = [str(script), client_key]
    elif backend_key == "vllm":
        cmd = [str(script), client_key, "--model", model_id]
    elif backend_key == "ollama":
        if client_key == "crush":
            cmd = [str(script), "--backend", "ollama", "--model", model_id]
        else:
            # For opencode with ollama, use environment variables
            os.environ["ANTHROPIC_AUTH_TOKEN"] = "ollama"
            os.environ["ANTHROPIC_BASE_URL"] = "http://localhost:11434"
            cmd = ["opencode", "-m", f"ollama/{model_id}"]

    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")
    console.print()

    os.execvp(cmd[0], cmd)


# =============================================================================
# Main Flows
# =============================================================================

def flow_install_backend(status: dict):
    """Backend installation flow."""
    backend_key = select_backend_to_install(status)
    if backend_key is None:
        return

    # Install system deps first if needed
    if backend_key == "llamacpp" and not is_backend_installed("llamacpp"):
        if not install_system_deps():
            return

    # Install backend
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Installing {BACKENDS[backend_key]['name']}...", total=None)

        try:
            if backend_key == "llamacpp":
                install_llamacpp()
            elif backend_key == "vllm":
                install_vllm()
            elif backend_key == "ollama":
                install_ollama()
        except subprocess.CalledProcessError as e:
            progress.remove_task(task)
            console.print(f"[red]✗ Installation failed: {e}[/red]")
            return

        progress.remove_task(task)

    # Select and download model
    model_cfg = select_model_to_download(backend_key)
    if model_cfg is None:
        console.print("[yellow]Skipping model download.[/yellow]")
        return

    console.print()
    try:
        if backend_key == "llamacpp":
            download_llamacpp_model(model_cfg)
        elif backend_key == "ollama":
            download_ollama_model(model_cfg["id"])
        elif backend_key == "vllm":
            console.print("[dim]vLLM downloads models on first use.[/dim]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Model download failed: {e}[/red]")
        return

    console.print()
    console.print("[bold green]✓ Backend installation complete![/bold green]")
    console.print()
    console.print("Next steps:")
    console.print(f"  [cyan]python setup.py run[/cyan]  # Launch with {BACKENDS[backend_key]['name']}")


def flow_install_client(status: dict):
    """Client installation flow."""
    choice = select_client_to_install(status)
    if choice is None:
        return

    console.print()

    if choice == "both":
        install_client("opencode")
        console.print()
        install_client("crush")
    else:
        install_client(choice)

    console.print()
    console.print("[bold green]✓ Client installation complete![/bold green]")


def flow_run(status: dict):
    """Run flow - select backend, model, client and launch."""
    # Check if any backend is available
    available = [k for k, v in status["backends"].items()
                 if v["installed"] and (v["models"] or v["running"])]

    if not available:
        console.print("[yellow]No backends available with models.[/yellow]")
        console.print()
        console.print("Install a backend first:")
        console.print("  [cyan]python setup.py install[/cyan]")
        return

    # Select backend
    backend_key = select_backend_to_run(status)
    if backend_key is None:
        return

    # Refresh models for this backend if it's running
    if status["backends"][backend_key]["running"]:
        status["backends"][backend_key]["models"] = get_installed_models(backend_key)

    # Select model
    model_id = select_model_to_run(backend_key, status)
    if model_id is None:
        return

    # Check if any client is installed
    installed_clients = [k for k, v in status["clients"].items() if v["installed"]]
    if not installed_clients:
        console.print("[yellow]No clients installed.[/yellow]")
        console.print()
        choice = questionary.confirm(
            "Would you like to install a client now?",
            style=STYLE,
        ).ask()
        if choice:
            flow_install_client(status)
            status = get_full_status()  # Refresh
        else:
            return

    # Select client
    client_key = select_client_to_run(status)
    if client_key is None:
        return

    # Launch!
    launch(backend_key, model_id, client_key)


def main_interactive():
    """Main interactive TUI flow."""
    print_header()

    console.print("[dim]Checking installation status...[/dim]")
    status = get_full_status()
    console.print()

    print_status_table(status)

    while True:
        action = select_main_action(status)

        if action is None:
            console.print("[dim]Exiting.[/dim]")
            return

        console.print()

        if action == "install_backend":
            flow_install_backend(status)
            status = get_full_status()  # Refresh
            console.print()
            print_status_table(status)

        elif action == "install_client":
            flow_install_client(status)
            status = get_full_status()  # Refresh
            console.print()
            print_status_table(status)

        elif action == "run":
            flow_run(status)
            # If we return here, run was cancelled
            console.print()


def main_status():
    """Show status only."""
    print_header()
    status = get_full_status()
    print_status_table(status)


def main():
    """Main entry point."""
    args = sys.argv[1:]

    if not args or args[0] in ("--help", "-h"):
        if not args:
            # Default: interactive mode
            main_interactive()
        else:
            console.print(__doc__)
        return

    cmd = args[0]

    if cmd == "install":
        print_header()
        status = get_full_status()
        print_status_table(status)
        flow_install_backend(status)

    elif cmd == "run":
        print_header()
        status = get_full_status()
        flow_run(status)

    elif cmd == "status":
        main_status()

    else:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        console.print("Run [cyan]python setup.py --help[/cyan] for usage.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        sys.exit(0)
