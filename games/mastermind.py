"""
games/mastermind.py
~~~~~~~~~~~~~~~~~~~
Mastermind — reference implementation for the Playground engine.

Rules:
  A secret permutation of A B C D E is chosen at the start.
  Each turn the player guesses a permutation.
  The game reveals how many letters are in the exact correct position.
  The player has 10 attempts to find the exact permutation.
"""

from __future__ import annotations

import random
from typing import Any

import playground as pg


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

class MastermindGame(pg.GameEnv):

    MAX_ATTEMPTS: int = 10
    LETTERS: list[str] = list("ABCDE")

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        # state placeholders — values set in reset()
        self._answer: list[str] = []
        self._attempts_left: int = 0
        self._history: list[tuple[str, int]] = []   # (guess, correct_positions)


    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 2 — IDENTITY + ACTION CONTRACT
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "Mastermind"

    @property
    def action_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "guess": {
                    "type": "string",
                    "description": "exactly 5 uppercase letters, one permutation of A B C D E — e.g. BADCE",
                    "pattern": "^[ABCDE]{5}$",
                }
            },
            "required": ["guess"],
            "additionalProperties": False,
        }


    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 3 — EPISODE LIFECYCLE
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> str:
        letters = self.LETTERS[:]
        self._rng.shuffle(letters)
        self._answer = letters
        self._attempts_left = self.MAX_ATTEMPTS
        self._history = []
        return self._build_obs()

    def step(self, action: dict[str, Any]) -> pg.StepResult:
        guess = action["guess"].strip().upper()

        # game-level validation (schema already checked character set)
        if len(set(guess)) != 5:
            raise ValueError(
                f"Each letter must appear exactly once. Got: {guess!r}"
            )
        previous_guesses = [g for g, _ in self._history]
        if guess in previous_guesses:
            raise ValueError(
                f"{guess!r} was already guessed. Try a different permutation."
            )

        correct = self._score(guess)
        self._history.append((guess, correct))
        self._attempts_left -= 1

        if correct == 5:
            return pg.StepResult(
                observation=self._build_obs(
                    f"Correct! The answer was {''.join(self._answer)}."
                ),
                done=True,
                success=True,
                info={
                    "answer": "".join(self._answer),
                    "attempts_used": self.MAX_ATTEMPTS - self._attempts_left,
                },
            )

        if self._attempts_left == 0:
            return pg.StepResult(
                observation=self._build_obs(
                    f"Out of attempts. The answer was {''.join(self._answer)}."
                ),
                done=True,
                success=False,
                info={"answer": "".join(self._answer)},
            )

        return pg.StepResult(
            observation=self._build_obs(
                f"{correct}/5 letters in the correct position."
            ),
            done=False,
        )


    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 4 — PRIVATE INTERNALS
    # ──────────────────────────────────────────────────────────────────────────

    def _render_state(self) -> str:
        attempt_no = self.MAX_ATTEMPTS - self._attempts_left + 1
        lines = [
            f"Attempt {attempt_no}/{self.MAX_ATTEMPTS}",
            "",
            "Guess history:",
        ]
        if not self._history:
            lines.append("  (none yet)")
        else:
            for i, (g, c) in enumerate(self._history, 1):
                bar = "█" * c + "░" * (5 - c)
                lines.append(f"  {i:2}. {g}   [{bar}] {c}/5")
        return "\n".join(lines)

    def _render_rules(self) -> str:
        return (
            "Guess the hidden permutation of the letters A B C D E.\n"
            "Each guess must use all five letters exactly once.\n"
            "After each guess you see how many letters are in the EXACT correct position.\n"
            "Position feedback only — you are not told WHICH letters are correct.\n"
            f"You have {self._attempts_left} attempt(s) remaining."
        )

    def _score(self, guess: str) -> int:
        return sum(1 for i in range(5) if guess[i] == self._answer[i])
