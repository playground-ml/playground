"""
games/mastermind.py
~~~~~~~~~~~~~~~~~~~
Mastermind — reference implementation for the Playground engine.

Rules:
  A secret permutation of A B C D E is chosen at the start.
  Each turn the player guesses a permutation.
  The game reveals how many letters are in the exact correct position.
  The player has 10 attempts to find the exact permutation.

Rendering notes:
  render_screen()   → the two-panel Unicode box shown in the terminal
                      (human mode + llm_watched mode)
  _render_state()   → compact plain-text snapshot for the LLM observation
  _render_rules()   → plain-text rules repeated every turn in the LLM observation

The terminal screen and the LLM observation are intentionally different:
  - the screen is wide, visual, and formatted for a human eye
  - the observation is compact and self-contained for the model
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

    # box geometry — match the original script exactly
    _L: int = 36   # left column inner width
    _R: int = 44   # right column inner width

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._answer: list[str] = []
        self._attempts_left: int = 0
        self._history: list[tuple[str, int]] = []   # (guess, correct_positions)
        self._last_message: str | None = None        # feedback from last step


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
        self._last_message = None
        return self._build_obs()

    def step(self, action: dict[str, Any]) -> pg.StepResult:
        guess = action["guess"].strip().upper()

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
            msg = f"Correct! The answer was {''.join(self._answer)}."
            self._last_message = msg
            return pg.StepResult(
                observation=self._build_obs(msg),
                done=True,
                success=True,
                info={
                    "answer": "".join(self._answer),
                    "attempts_used": self.MAX_ATTEMPTS - self._attempts_left,
                },
            )

        if self._attempts_left == 0:
            msg = f"Out of attempts. The answer was {''.join(self._answer)}."
            self._last_message = msg
            return pg.StepResult(
                observation=self._build_obs(msg),
                done=True,
                success=False,
                info={"answer": "".join(self._answer)},
            )

        msg = f"{correct}/5 letters in the correct position."
        self._last_message = msg
        return pg.StepResult(
            observation=self._build_obs(msg),
            done=False,
        )


    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 4 — PRIVATE INTERNALS
    # ──────────────────────────────────────────────────────────────────────────

    # ── terminal screen (human eye) ───────────────────────────────────────────

    def render_screen(self) -> str:
        """Two-panel Unicode box string for terminal display.

        Called by display.py in human and llm_watched modes.
        Never sent to the LLM — the model receives _build_obs() instead.
        """
        L, R = self._L, self._R
        attempt_number = self.MAX_ATTEMPTS - self._attempts_left + 1

        def pad(text: str, width: int) -> str:
            text = str(text)
            return text[:width] if len(text) > width else text + " " * (width - len(text))

        def row(left: str, right: str) -> str:
            return "│ " + pad(left, L) + " │ " + pad(right, R) + " │"

        def divider_row(lc: str = "├", mc: str = "┼", rc: str = "┤") -> str:
            return lc + "─" * (L + 2) + mc + "─" * (R + 2) + rc

        # top border with centred title
        top_fill = "─" * (L + R + 5)
        title = " Mastermind "
        insert_pos = (len(top_fill) - len(title)) // 2
        top_bar = (
            "╭"
            + top_fill[:insert_pos]
            + title
            + top_fill[insert_pos + len(title):]
            + "╮"
        )

        # progress bar
        filled = attempt_number - 1
        bar = "█" * filled + "░" * (self.MAX_ATTEMPTS - filled)
        progress_label = f"Attempt {attempt_number}/{self.MAX_ATTEMPTS}  [{bar}]"

        # left panel — all 10 history slots always shown
        history_lines: list[str] = []
        for i in range(1, self.MAX_ATTEMPTS + 1):
            if i - 1 < len(self._history):
                g, c = self._history[i - 1]
                pip = "█" * c + "░" * (5 - c)
                history_lines.append(f"  {str(i).rjust(2)}.  {g}   [{pip}] {c}/5")
            else:
                history_lines.append(f"  {str(i).rjust(2)}.  —")

        # right panel — fixed 13 rows
        right: list[str] = [
            "  Rules",
            "  Guess the shuffled order of A B C D E.",
            "  Enter exactly 5 unique letters (A–E).",
            "  10 attempts. Positions matter.",
            "",
            "  " + "─" * 41,
            "  Expected input format",
            "  {",
            '    "guess": "ABCDE"',
            "  }",
            "",
            "",
            "",
        ]

        # slot 12 (last row) shows the most recent feedback
        if self._last_message:
            right[12] = "  " + self._last_message[: R - 2]

        left: list[str] = ["  Guess history"] + history_lines + ["", ""]

        lines = [
            top_bar,
            row("  " + progress_label, ""),
            divider_row(),
        ]
        for i in range(13):
            lines.append(row(left[i], right[i]))
        lines.append("╰" + "─" * (L + 2) + "┴" + "─" * (R + 2) + "╯")

        return "\n".join(lines)

    # ── LLM observation (plain text) ──────────────────────────────────────────

    def _render_state(self) -> str:
        attempt_no = self.MAX_ATTEMPTS - self._attempts_left + 1
        lines = [f"Attempt {attempt_no}/{self.MAX_ATTEMPTS}", "", "Guess history:"]
        if not self._history:
            lines.append("  (none yet)")
        else:
            for i, (g, c) in enumerate(self._history, 1):
                pip = "█" * c + "░" * (5 - c)
                lines.append(f"  {i:2}. {g}   [{pip}] {c}/5")
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
