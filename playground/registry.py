"""
playground.registry
~~~~~~~~~~~~~~~~~~~~
Central game registry.  Add one line here to make a game available
everywhere: CLI, run_game.py, and any tooling that lists games.

Format
------
REGISTRY = {
    "key": ("module.path", "ClassName"),
}

The key is what you pass to --game in run_game.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playground.core import GameEnv

REGISTRY: dict[str, tuple[str, str]] = {
    "mastermind": ("games.mastermind", "MastermindGame"),
}


def get_env_class(key: str) -> type:
    """Import and return the GameEnv class for *key*."""
    import importlib
    if key not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Unknown game {key!r}.  Known games: {known}")
    module_path, class_name = REGISTRY[key]
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def list_games() -> list[str]:
    return sorted(REGISTRY)
