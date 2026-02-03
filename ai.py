#!/usr/bin/env python3
"""
Interactive AI Inference Launcher

Arrow-key TUI to select backend, model, and frontend.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import questionary
    from questionary import Style
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
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

# Backend configurations
BACKENDS = {
    "ollama": {
        "name": "Ollama",
        "port": 11434,
        "check_url": "http://localhost:11434/api/tags",
        "models_cmd": ["ollama", "list"],
    },
    "vllm": {
        "name": "vLLM",
        "port": 8000,
        "check_url": "http://localhost:8000/v1/models",
    },
    "llamacpp": {
        "name": "llama.cpp",
        "port": 10000,
        "check_url": "http://localhost:10000/v1/models",
    },
}

FRONTENDS = {
    "opencode": {"name": "OpenCode", "desc": "AI coding assistant TUI"},
    "crush": {"name": "Crush", "desc": "Charm's AI chat TUI"},
}


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


def get_ollama_models() -> list[dict]:
    """Get models from Ollama."""
    models = []

    # Try API first
    data = fetch_url_json("http://localhost:11434/api/tags")
    if data and "models" in data:
        for m in data["models"]:
            name = m.get("name", "")
            size = m.get("size", 0)
            size_gb = size / (1024**3) if size else 0
            models.append({
                "id": name,
                "name": name,
                "size": f"{size_gb:.1f}GB" if size_gb else "",
            })
        return models

    # Fallback to CLI
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            for line in lines:
                parts = line.split()
                if parts:
                    models.append({
                        "id": parts[0],
                        "name": parts[0],
                        "size": parts[2] if len(parts) > 2 else "",
                    })
    except Exception:
        pass

    return models


def get_vllm_models() -> list[dict]:
    """Get models from vLLM server."""
    models = []
    data = fetch_url_json("http://localhost:8000/v1/models")
    if data and "data" in data:
        for m in data["data"]:
            models.append({
                "id": m.get("id", ""),
                "name": m.get("id", ""),
                "size": "",
            })
    return models


def get_llamacpp_models() -> list[dict]:
    """Get models from llama.cpp server.

    Supports both single-model mode and router mode:
    - Single mode: Returns the loaded model from /v1/models
    - Router mode: Returns models from /v1/models with status + discovers local GGUF files
    """
    models = []
    loaded_ids = set()
    known_model_ids = set()  # Track all models returned by API (loaded or not)

    # Get models from API
    data = fetch_url_json("http://localhost:10000/v1/models")
    if data and "data" in data:
        for m in data["data"]:
            model_id = m.get("id", "")
            known_model_ids.add(model_id)

            # In router mode, check status.value for load state
            status = m.get("status")
            if status and isinstance(status, dict) and "value" in status:
                # Router mode - has status field with value
                is_loaded = status.get("value") == "loaded"
            else:
                # Single model mode - if it's in the list, it's loaded
                is_loaded = True

            if is_loaded:
                loaded_ids.add(model_id)
                models.append({
                    "id": model_id,
                    "name": f"{model_id} [loaded]",
                    "size": "",
                    "loaded": True,
                })
            else:
                # Router mode: model discovered but not loaded
                models.append({
                    "id": model_id,
                    "name": f"{model_id} (available)",
                    "size": "",
                    "loaded": False,
                })

    # Scan local models directory for additional available models
    # (ones not already discovered by the router)
    models_dir = SCRIPT_DIR / "models"
    if models_dir.exists():
        for gguf in models_dir.rglob("*.gguf"):
            rel = gguf.relative_to(models_dir)
            # Create model ID in format llama.cpp expects
            model_id = str(rel)  # e.g., "GLM-4.7-Flash-GGUF/GLM-4.7-Flash-UD-Q4_K_XL.gguf"
            model_name = str(rel.parent / rel.stem)  # e.g., "GLM-4.7-Flash-GGUF/GLM-4.7-Flash-UD-Q4_K_XL"
            file_stem = rel.stem  # e.g., "GLM-4.7-Flash-UD-Q4_K_XL"

            # Skip if already known from API (either loaded or discovered by router)
            is_known = False
            for kid in known_model_ids:
                kid_lower = kid.lower()
                # Check if API model ID matches this local file
                if any(part.lower() in kid_lower for part in [file_stem, rel.parent.name] if part):
                    is_known = True
                    break
                # Check reverse: filename contains API model name parts
                if "/" in kid:
                    kid_parts = kid.split("/")[-1].lower()
                    if kid_parts in model_name.lower() or kid_parts in file_stem.lower():
                        is_known = True
                        break

            if not is_known:
                size_gb = gguf.stat().st_size / (1024**3)
                models.append({
                    "id": str(gguf),  # Full path for loading
                    "name": f"{model_name} (available)",
                    "size": f"{size_gb:.1f}GB",
                    "loaded": False,
                })

    return models


def load_llamacpp_model(model_path: str) -> bool:
    """Load a model in llama.cpp router mode.

    Args:
        model_path: Path or name of the model to load

    Returns:
        True if successful, False otherwise
    """
    try:
        result = subprocess.run(
            [
                "curl", "-sf", "-X", "POST",
                "http://localhost:10000/models/load",
                "-H", "Content-Type: application/json",
                "-d", json.dumps({"model": model_path}),
            ],
            capture_output=True,
            text=True,
            timeout=120,  # Model loading can take time
        )
        return result.returncode == 0
    except Exception:
        return False


def unload_llamacpp_model(model_name: str) -> bool:
    """Unload a model in llama.cpp router mode.

    Args:
        model_name: Name of the model to unload

    Returns:
        True if successful, False otherwise
    """
    try:
        result = subprocess.run(
            [
                "curl", "-sf", "-X", "POST",
                "http://localhost:10000/models/unload",
                "-H", "Content-Type: application/json",
                "-d", json.dumps({"model": model_name}),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_backend_status() -> dict[str, dict]:
    """Get status of all backends."""
    status = {}

    for key, cfg in BACKENDS.items():
        running = check_port(cfg["port"])
        models = []

        if running:
            if key == "ollama":
                models = get_ollama_models()
            elif key == "vllm":
                models = get_vllm_models()
            elif key == "llamacpp":
                models = get_llamacpp_models()
        elif key == "ollama":
            # Ollama CLI might work even if server not explicitly running
            models = get_ollama_models()
            if models:
                running = True  # Ollama auto-starts

        status[key] = {
            "running": running,
            "models": models,
            "port": cfg["port"],
            "name": cfg["name"],
        }

    return status


def print_header():
    """Print the launcher header."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]AI Inference Launcher[/bold cyan]\n"
        "[dim]Select backend, model, and frontend[/dim]",
        border_style="cyan",
    ))
    console.print()


def print_status_table(status: dict):
    """Print backend status as a table."""
    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("Backend", style="white")
    table.add_column("Status", style="white")
    table.add_column("Models", style="white")
    table.add_column("Port", style="dim")

    for key, info in status.items():
        status_str = "[green]running[/green]" if info["running"] else "[red]stopped[/red]"
        model_count = len(info["models"])
        models_str = f"{model_count} model{'s' if model_count != 1 else ''}" if model_count else "[dim]none[/dim]"
        table.add_row(info["name"], status_str, models_str, str(info["port"]))

    console.print(table)
    console.print()


def select_backend(status: dict) -> str | None:
    """Interactive backend selection."""
    choices = []

    for key, info in status.items():
        status_icon = "[green]●[/green]" if info["running"] else "[red]○[/red]"
        model_count = len(info["models"])
        label = f"{info['name']:12} {status_icon}  ({model_count} models)"

        # Only allow selection if running and has models
        disabled = None
        if not info["running"]:
            disabled = "not running"
        elif not info["models"]:
            disabled = "no models"

        choices.append(questionary.Choice(
            title=label,
            value=key,
            disabled=disabled,
        ))

    choices.append(questionary.Choice(title="Exit", value=None))

    return questionary.select(
        "Select backend:",
        choices=choices,
        style=STYLE,
        qmark="",
        pointer=">",
    ).ask()


def select_model(models: list[dict], backend: str = "") -> str | None:
    """Interactive model selection."""
    choices = []

    for m in models:
        size_str = f"  [{m['size']}]" if m.get("size") else ""
        label = f"{m['name']}{size_str}"
        choices.append(questionary.Choice(title=label, value=m["id"]))

    choices.append(questionary.Choice(title="← Back", value=None))

    return questionary.select(
        "Select model:",
        choices=choices,
        style=STYLE,
        qmark="",
        pointer=">",
    ).ask()


def is_llamacpp_router_mode() -> bool:
    """Check if llama.cpp server is running in router mode.

    Router mode is detected by checking if /models endpoint returns
    model status information (loaded/unloaded state).
    """
    data = fetch_url_json("http://localhost:10000/v1/models")
    if data and "data" in data:
        for m in data["data"]:
            # Router mode includes status field
            if "status" in m:
                return True
    return False


def select_llamacpp_model_with_actions(models: list[dict]) -> tuple[str | None, str]:
    """Interactive llama.cpp model selection with load/unload actions.

    Returns:
        Tuple of (model_id, action) where action is one of:
        - "use": Use the selected model
        - "load": Load the selected model first
        - "unload": Unload the selected model
        - "back": Go back
        - "manage": Go to model management
    """
    # Separate loaded and available models
    loaded = [m for m in models if m.get("loaded", False)]
    available = [m for m in models if not m.get("loaded", False)]

    choices = []

    # Add loaded models first (ready to use)
    if loaded:
        choices.append(questionary.Choice(
            title="─── Loaded Models (ready to use) ───",
            value=None,
            disabled="separator"
        ))
        for m in loaded:
            size_str = f"  [{m['size']}]" if m.get("size") else ""
            label = f"  ● {m['name'].replace(' [loaded]', '')}{size_str}"
            choices.append(questionary.Choice(title=label, value=("use", m["id"])))

    # Add available models (need loading)
    if available:
        choices.append(questionary.Choice(
            title="─── Available Models (need loading) ───",
            value=None,
            disabled="separator"
        ))
        for m in available:
            size_str = f"  [{m['size']}]" if m.get("size") else ""
            label = f"  ○ {m['name'].replace(' (available)', '')}{size_str}"
            choices.append(questionary.Choice(title=label, value=("load", m["id"])))

    # Add management options if there are loaded models
    if loaded:
        choices.append(questionary.Choice(
            title="─────────────────────────────────────",
            value=None,
            disabled="separator"
        ))
        choices.append(questionary.Choice(
            title="  ⚙ Manage models (unload to free memory)",
            value=("manage", None)
        ))

    choices.append(questionary.Choice(title="← Back", value=("back", None)))

    result = questionary.select(
        "Select model (● loaded, ○ available):",
        choices=choices,
        style=STYLE,
        qmark="",
        pointer=">",
    ).ask()

    if result is None:
        return None, "back"
    return result[1], result[0]


def manage_llamacpp_models(models: list[dict]) -> bool:
    """Model management interface for llama.cpp - unload models.

    Returns:
        True if user wants to continue, False to go back
    """
    loaded = [m for m in models if m.get("loaded", False)]

    if not loaded:
        console.print("[yellow]No models currently loaded.[/yellow]")
        return True

    choices = []
    for m in loaded:
        name = m['name'].replace(' [loaded]', '')
        choices.append(questionary.Choice(title=f"Unload: {name}", value=m["id"]))

    choices.append(questionary.Choice(title="← Back to model selection", value=None))

    model_to_unload = questionary.select(
        "Select model to unload:",
        choices=choices,
        style=STYLE,
        qmark="",
        pointer=">",
    ).ask()

    if model_to_unload is None:
        return True

    # Extract just the model name/alias for unloading
    # The ID might be a full path, but we need the alias
    model_name = Path(model_to_unload).stem if "/" in model_to_unload else model_to_unload

    console.print(f"[yellow]Unloading {model_name}...[/yellow]")

    if unload_llamacpp_model(model_name):
        console.print(f"[green]✓ Model {model_name} unloaded successfully[/green]")
    else:
        console.print(f"[red]✗ Failed to unload {model_name}[/red]")

    return True


def load_llamacpp_model_interactive(model_path: str) -> bool:
    """Load a llama.cpp model with progress feedback.

    Args:
        model_path: Path to the model to load

    Returns:
        True if loaded successfully, False otherwise
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    # Extract a friendly name from the path
    model_name = Path(model_path).stem if "/" in model_path else model_path

    console.print(f"[cyan]Loading model: {model_name}[/cyan]")
    console.print("[dim]This may take a moment for large models...[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Loading {model_name}...", total=None)

        # For router mode, we need to use the relative path from models dir
        # or the model alias that llama.cpp expects
        if model_path.startswith(str(SCRIPT_DIR)):
            # Convert absolute path to relative path from models dir
            try:
                rel_path = Path(model_path).relative_to(SCRIPT_DIR / "models")
                # Use parent directory name as the model alias (llama.cpp convention)
                load_name = str(rel_path.parent)
            except ValueError:
                load_name = model_name
        else:
            load_name = model_name

        success = load_llamacpp_model(load_name)
        progress.remove_task(task)

    if success:
        console.print(f"[green]✓ Model {model_name} loaded successfully[/green]")
        return True
    else:
        console.print(f"[red]✗ Failed to load {model_name}[/red]")
        console.print("[dim]Check server logs: ./run_llamacpp.sh logs[/dim]")
        return False


def select_frontend() -> str | None:
    """Interactive frontend selection."""
    choices = []

    for key, info in FRONTENDS.items():
        label = f"{info['name']:12} - {info['desc']}"
        choices.append(questionary.Choice(title=label, value=key))

    choices.append(questionary.Choice(title="← Back", value=None))

    return questionary.select(
        "Select frontend:",
        choices=choices,
        style=STYLE,
        qmark="",
        pointer=">",
    ).ask()


def launch(backend: str, model: str, frontend: str):
    """Launch the selected configuration."""
    console.print()
    console.print(f"[bold green]Launching:[/bold green]")
    console.print(f"  Backend:  [cyan]{backend}[/cyan]")
    console.print(f"  Model:    [cyan]{model}[/cyan]")
    console.print(f"  Frontend: [cyan]{frontend}[/cyan]")
    console.print()

    if backend == "ollama":
        script = SCRIPT_DIR / "run_crush.sh" if frontend == "crush" else SCRIPT_DIR / "run_claude.sh"
        if frontend == "crush":
            cmd = [str(script), "--backend", "ollama", "--model", model]
        else:
            cmd = [str(script), "--model", model]

    elif backend == "vllm":
        script = SCRIPT_DIR / "run_vllm.sh"
        cmd = [str(script), frontend, "--model", model]

    elif backend == "llamacpp":
        script = SCRIPT_DIR / "run_llamacpp.sh"
        if frontend == "opencode":
            cmd = [str(script), "opencode"]
        else:
            # For crush with llama.cpp, use run_crush.sh pointing to llamacpp port
            script = SCRIPT_DIR / "run_crush.sh"
            # Extract model alias if it's a path
            model_alias = Path(model).stem if "/" in model else model
            cmd = [str(script), "--backend", "vllm", "--model", model_alias, "--vllm-host", "http://localhost:10000"]

    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")
    console.print()

    os.execvp(cmd[0], cmd)


def main():
    """Main entry point."""
    print_header()

    # Get current status
    console.print("[dim]Checking backends...[/dim]")
    status = get_backend_status()
    console.print()

    print_status_table(status)

    # Check if any backend is available
    available = [k for k, v in status.items() if v["running"] and v["models"]]
    if not available:
        console.print("[yellow]No backends available with models.[/yellow]")
        console.print()
        console.print("Start a backend first:")
        console.print("  [cyan]ollama serve[/cyan]              # Start Ollama")
        console.print("  [cyan]./run_vllm.sh serve[/cyan]       # Start vLLM")
        console.print("  [cyan]./run_llamacpp.sh start[/cyan]   # Start llama.cpp")
        console.print()
        return

    # Selection loop
    while True:
        backend = select_backend(status)
        if backend is None:
            console.print("[dim]Exiting.[/dim]")
            return

        # Special handling for llama.cpp with router mode
        if backend == "llamacpp" and is_llamacpp_router_mode():
            model = None
            while model is None:
                # Refresh models list to reflect any load/unload changes
                status["llamacpp"]["models"] = get_llamacpp_models()
                models = status["llamacpp"]["models"]

                if not models:
                    console.print("[yellow]No models available.[/yellow]")
                    break

                model_id, action = select_llamacpp_model_with_actions(models)

                if action == "back":
                    break  # Back to backend selection

                elif action == "manage":
                    manage_llamacpp_models(models)
                    continue  # Refresh and show model selection again

                elif action == "load":
                    # Load the selected model
                    if load_llamacpp_model_interactive(model_id):
                        # After loading, the model becomes the selected one
                        # Get the loaded model's ID/alias
                        status["llamacpp"]["models"] = get_llamacpp_models()
                        loaded = [m for m in status["llamacpp"]["models"] if m.get("loaded")]
                        if loaded:
                            model = loaded[-1]["id"]  # Use the most recently loaded
                    continue  # Show model selection again if load failed

                elif action == "use":
                    model = model_id

            if model is None:
                continue  # Back to backend selection
        else:
            # Standard model selection for other backends
            model = select_model(status[backend]["models"], backend)
            if model is None:
                continue  # Back to backend selection

        frontend = select_frontend()
        if frontend is None:
            continue  # Back to model selection (actually backend, simpler flow)

        # Launch!
        launch(backend, model, frontend)
        break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        sys.exit(0)
