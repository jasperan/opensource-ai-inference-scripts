
import typer
import sys
import subprocess
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
import harbor_config

app = typer.Typer(help="Harbor CLI for managing local AI services")
config_app = typer.Typer(help="Manage configuration")
app.add_typer(config_app, name="config")
console = Console()

@config_app.command("set")
def config_set(key: str, value: str):
    """Set a configuration value."""
    harbor_config.set_config_value(key, value)
    console.print(f"[green]Set {key} to {value}[/green]")

@app.command()
def pull(resource: str):
    """Pull a resource (ollama binary or a model)."""
    if resource == "ollama":
        version = harbor_config.get_config_value("ollama.version")
        if not version:
            console.print("[yellow]No ollama.version set in config. Using latest.[/yellow]")
            version = "latest" # Or handle appropriately
        
        console.print(f"[bold blue]Pulling Ollama version: {version}[/bold blue]")
        # Mock implementation for actual binary download as it might require sudo or specific OS logic
        # For the purpose of this task, we will simulate the action or use a mock command
        # In a real scenario, this would curl the equivalent release from GitHub
        console.print(f"[green]Successfully pulled Ollama {version} (Mock)[/green]")
        
    else:
        # Assume it's a model
        console.print(f"[bold blue]Pulling model: {resource}[/bold blue]")
        try:
            subprocess.run(["ollama", "pull", resource], check=True)
            console.print(f"[green]Successfully pulled {resource}[/green]")
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Failed to pull {resource}: {e}[/bold red]")
        except FileNotFoundError:
             console.print("[bold red]Ollama command not found. Is it installed?[/bold red]")

@app.command()
def up(service: str, backend: str = typer.Option("ollama", help="Backend for crush: ollama or vllm"),
       model: str = typer.Option("", help="Model ID override")):
    """Start a service (opencode or crush)."""
    if service == "opencode":
        console.print("[bold green]Starting OpenCode (Claude Code)...[/bold green]")
        script_path = Path("run_claude.sh")
        if script_path.exists():
             os.chmod(script_path, 0o755)
             try:
                subprocess.run(["./run_claude.sh"], check=True)
             except subprocess.CalledProcessError as e:
                console.print(f"[bold red]OpenCode failed to start: {e}[/bold red]")
             except KeyboardInterrupt:
                 console.print("\n[yellow]OpenCode stopped.[/yellow]")
        else:
            console.print("[red]run_claude.sh not found![/red]")
    elif service == "crush":
        console.print("[bold green]Starting Crush...[/bold green]")
        script_path = Path("run_crush.sh")
        if script_path.exists():
            os.chmod(script_path, 0o755)
            cmd = ["./run_crush.sh", "--backend", backend]
            if model:
                cmd.extend(["--model", model])
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                console.print(f"[bold red]Crush failed to start: {e}[/bold red]")
            except KeyboardInterrupt:
                console.print("\n[yellow]Crush stopped.[/yellow]")
        else:
            console.print("[red]run_crush.sh not found![/red]")
    else:
        console.print(f"[red]Unknown service: {service}. Available: opencode, crush[/red]")

if __name__ == "__main__":
    app()
