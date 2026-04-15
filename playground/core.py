"""
playground.core
~~~~~~~~~~~~~~~
GameEnv ABC and StepResult — the only types a game developer imports.

The engine (runner, agent, checkpoint writer) calls exactly four things
on a game instance:
  env.name             once, to build the system prompt
  env.action_schema    once, to build the system prompt
  env.reset()          once per episode
  env.step(action)     once per turn

Everything else is private to the game file.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepResult:
    """Returned by every call to GameEnv.step().

    Fields
    ------
    observation : str
        What the model (or human) sees at the start of the next turn.
        Must be self-contained — the model has no memory outside of it.
        Ignored when done=True.
    done : bool
        True when the episode has ended for any reason (win, loss, draw).
    success : bool
        True only on a win.  Requires done=True.
    info : dict
        Arbitrary key-value bag.  Never shown to the model.
        Forwarded verbatim into the step checkpoint record.
    """
    observation: str
    done: bool = False
    success: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class GameEnv(abc.ABC):
    """Base class for every Playground game.

    Subclass this in a single .py file.  Follow the four-section layout
    documented in games/game_template.py.

    The engine never calls _render_state, _render_rules, or any other
    private method.  Use _build_obs() to assemble observations — do not
    write observation strings manually.
    """

    # ── Section 2 — identity + action contract (implement these) ──────────────

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short display name used in system prompts and log output."""

    @property
    @abc.abstractmethod
    def action_schema(self) -> dict[str, Any]:
        """JSON Schema (draft-07) for one valid action.

        The engine:
          1. Embeds this verbatim in the LLM system prompt.
          2. Validates the model's JSON response against it.
          3. Passes the validated dict to step().

        Required keys:
          "type": "object"
          "properties": { ... }   — each with a "description" the model reads
          "required": [...]
          "additionalProperties": False
        """

    # ── Section 3 — episode lifecycle (implement these) ───────────────────────

    @abc.abstractmethod
    def reset(self) -> str:
        """Start a fresh episode.  Initialise ALL mutable state.
        Return the first observation via self._build_obs().
        """

    @abc.abstractmethod
    def step(self, action: dict[str, Any]) -> StepResult:
        """Apply *action*.  Update state.  Return a StepResult.

        action  : already validated against action_schema.
        Raise ValueError for game-illegal moves (e.g. duplicate guess).
        The engine catches ValueError, skips the turn counter, and feeds
        the error back to the model as a correction prompt.
        Build the observation via self._build_obs(message).
        """

    # ── Section 4 — private rendering (implement these) ───────────────────────

    @abc.abstractmethod
    def _render_state(self) -> str:
        """Current board / world snapshot only.
        No rules, no schema hint — just what changed this turn.
        """

    @abc.abstractmethod
    def _render_rules(self) -> str:
        """Complete rules the model needs every turn.
        Written in second person.  3–6 lines is ideal.
        Do not assume the model remembers earlier turns.
        """

    # ── Engine-provided: observation assembly ─────────────────────────────────

    def _build_obs(self, message: str | None = None) -> str:
        """Assemble a complete, self-contained observation string.

        Always call this from reset() and step() — never build observation
        strings manually.

        Structure (always in this order):
          === {name} ===
          {_render_state()}
          --- Rules ---
          {_render_rules()}
          --- Respond with this JSON ---
          {auto-generated schema hint}
          >> {message}   (only if message is not None)

        Parameters
        ----------
        message:
            Optional one-line feedback appended at the end.
            Use for per-turn results: "3/5 correct.", "Too high!", etc.
        """
        parts: list[str] = [
            f"=== {self.name} ===",
            "",
            self._render_state(),
            "",
            "--- Rules ---",
            self._render_rules(),
            "",
            "--- Respond with this JSON ---",
            self._schema_hint(),
        ]
        if message is not None:
            parts += ["", f">> {message}"]
        return "\n".join(parts)

    def _schema_hint(self) -> str:
        """Auto-generate a one-line JSON example from action_schema.

        Uses each property's "description" as the placeholder value,
        falling back to the key name.  Keeps the hint compact and in sync
        with the schema without any manual maintenance.
        """
        props: dict = self.action_schema.get("properties", {})
        example = {
            k: f"<{v.get('description', k)}>"
            for k, v in props.items()
        }
        return json.dumps(example, ensure_ascii=False)
