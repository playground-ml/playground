"""
playground
~~~~~~~~~~
A lightweight async game engine for LLM-playable text games.

    import playground as pg

Game development
----------------
Subclass pg.GameEnv in a single .py file.
See games/game_template.py for the four-section layout.

Running a game
--------------
Use run_game.py (the recommended entrypoint), or call pg.run() directly:

    import asyncio
    import playground as pg
    from openai import AsyncOpenAI
    from games.mastermind import MastermindGame

    client = AsyncOpenAI(api_key="...")

    # human plays
    asyncio.run(pg.run(MastermindGame, batch_instance=0))

    # LLM plays, watched
    asyncio.run(pg.run(MastermindGame, batch_instance=1,
                       client=client, model="gpt-4o"))

    # 16 parallel LLM episodes
    asyncio.run(pg.run(MastermindGame, batch_instance=16,
                       client=client, model="gpt-4o"))
"""

from playground.core import GameEnv, StepResult
from playground.runner import run
from playground.registry import get_env_class, list_games

__all__ = [
    "GameEnv",
    "StepResult",
    "run",
    "get_env_class",
    "list_games",
]
