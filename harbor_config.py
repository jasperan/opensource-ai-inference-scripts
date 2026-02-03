
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

CONFIG_FILE = Path("harbor_config.json")

def load_config() -> Dict[str, Any]:
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_config(config: Dict[str, Any]) -> None:
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

def get_config_value(key: str) -> Optional[Any]:
    config = load_config()
    keys = key.split(".")
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return None
    return value

def set_config_value(key: str, value: Any) -> None:
    config = load_config()
    keys = key.split(".")
    current = config
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value
    save_config(config)
