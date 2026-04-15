"""
games/game_template.py
~~~~~~~~~~~~~~~~~~~~~~
Copy this file to start a new game.  Fill in every TODO.
Delete all comments once you're done.

The engine calls ONLY: name, action_schema, reset(), step().
Everything else is yours.

Quick test after filling in:
    python run_game.py --game <your_key> --batch 0     # play yourself
    python run_game.py --game <your_key> --batch 1 \   # watch LLM play
        --model gpt-4o
"""

from __future__ import annotations

import random
from typing import Any

import playground as pg


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CONFIGURATION
#
# • Class-level constants only.
# • __init__ accepts ONLY seed / difficulty — no game state here.
# • Do NOT initialise game state here; that belongs in reset().
# ──────────────────────────────────────────────────────────────────────────────

class MyGame(pg.GameEnv):

    MAX_TURNS: int = 10  # TODO

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        # Declare every piece of mutable state here with placeholder values.
        # Actual values are assigned in reset().
        self._turn: int = 0


    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 2 — IDENTITY + ACTION CONTRACT
    #
    # • name   : short string, appears in prompts and logs.
    # • action_schema : JSON Schema dict.  The engine embeds this in the system
    #   prompt and validates the model's response against it before step().
    #
    # action_schema tips:
    #   Single string: "guess": {"type":"string","description":"..."}
    #   Enum choice:   "move":  {"type":"string","enum":["north","south",...]}
    #   Integer:       "n":     {"type":"integer","minimum":1,"maximum":100}
    #   Discriminated: "type":  {"type":"string","enum":["attack","use",...]}
    #                  "value": {"type":"string","description":"..."}
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "MyGame"  # TODO

    @property
    def action_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {  # TODO: rename key(s)
                    "type": "string",
                    "description": "TODO: describe what the model should put here",
                },
            },
            "required": ["action"],  # TODO
            "additionalProperties": False,
        }


    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 3 — EPISODE LIFECYCLE
    #
    # reset()
    #   • Initialise ALL mutable state here.
    #   • Return self._build_obs() — the engine shows this to the player.
    #   • Calling reset() twice must produce a clean fresh episode.
    #
    # step(action)
    #   • action is a dict already validated against action_schema.
    #   • Raise ValueError for game-illegal moves (duplicate guess, wall, etc).
    #     The engine catches it, skips the turn counter, feeds the error back.
    #   • Return pg.StepResult with done=True on win OR loss.
    #   • success=True only on win.
    #   • Use self._build_obs(message) to produce the observation.
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> str:
        self._turn = 0
        # TODO: initialise all game state
        return self._build_obs()

    def step(self, action: dict[str, Any]) -> pg.StepResult:
        value = action["action"]  # TODO: use your key name(s)

        # TODO: raise ValueError for illegal moves
        # if <illegal>:
        #     raise ValueError("explain why this move is not allowed")

        self._turn += 1

        # TODO: check win condition
        won = False
        if won:
            return pg.StepResult(
                observation=self._build_obs("You won!"),
                done=True,
                success=True,
                info={"turns": self._turn},
            )

        # TODO: check loss condition
        lost = self._turn >= self.MAX_TURNS
        if lost:
            return pg.StepResult(
                observation=self._build_obs("Out of turns."),
                done=True,
                success=False,
            )

        return pg.StepResult(
            observation=self._build_obs(),
            done=False,
        )


    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 4 — PRIVATE INTERNALS
    #
    # _render_state()  → current board snapshot, no rules.
    # _render_rules()  → full rules, written in second person, repeated every turn.
    #
    # Add any other helpers below (_score, _deal, _check_win, …).
    # The engine never calls anything in this section.
    # ──────────────────────────────────────────────────────────────────────────

    def _render_state(self) -> str:
        # TODO: return a compact text snapshot of the current game state.
        return f"Turn {self._turn}/{self.MAX_TURNS}"

    def _render_rules(self) -> str:
        # TODO: complete rules the model needs every turn (3–6 lines).
        return (
            "TODO: describe the goal.\n"
            "TODO: describe what actions are legal.\n"
            f"You have {self.MAX_TURNS - self._turn} turn(s) remaining."
        )
