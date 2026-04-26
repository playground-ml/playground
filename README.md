# Playground: Python CLI Game Engine with LLM Layer

**Playground** is a Python CLI game engine built with an integrated LLM layer. It bridges the gap between traditional interactive text-based applications and autonomous model-driven environments.

## Why Playground?

Traditionally, when writing a Python game (like a simple logic puzzle or guessing game), developers rely on a `while` loop paired with `input()` prompts. The user makes a guess, the terminal reads it, and the environment responds accordingly. 

However, this severely tightly couples the game logic to human inputs (the keyboard) and standard output text formatting. This means **only a human user can play the game**. 

The objective of **Playground** is to abstract the execution of text-based iterations to enable LLM plug-and-play functionality. By developing a game on the Playground Game Engine, you empower **both a human user AND any capable LLM agent** (via OpenRouter API or others) to seamlessly play the game. Furthermore, the engine natively handles the complexity of spinning up **multiple agents to run multiple games in parallel**, making it exceptionally easy to benchmark, test, or generate execution traces.

## Comparing Implementations

To illustrate this difference, let's look at the classic "Mastermind" guessing game—where the objective is to guess a hidden permutation of letters (A, B, C, D, E).

### 1. Without Framework (Vanilla Python)
In standard Python, the game is heavily entangled with the layout, terminal rendering logic, and interactive `try/except` `input()` loops. An LLM agent would require customized scripts just to read the terminal and supply guesses.

<details>
<summary>Click to view Vanilla Python implementation</summary>

```python
import random

def print_header(attempt_number, correct_hint=None, guess_history=None):
    if guess_history is None:
        guess_history = []

    L = 36   # left column inner width
    R = 44   # right column inner width

    def pad(text, width):
        text = str(text)
        return text[:width] if len(text) > width else text + " " * (width - len(text))

    def row(left, right):
        return "│ " + pad(left, L) + " │ " + pad(right, R) + " │"

    def divider_row(lc="├", mc="┼", rc="┤"):
        return lc + "─" * (L + 2) + mc + "─" * (R + 2) + rc

    # Top border with centered title
    top_fill = "─" * (L + R + 5)
    title = " Mastermind "
    insert_pos = (len(top_fill) - len(title)) // 2
    top_bar = "╭" + top_fill[:insert_pos] + title + top_fill[insert_pos + len(title):] + "╮"

    # Progress bar
    filled = attempt_number - 1
    bar = "█" * filled + "░" * (10 - filled)
    progress_label = f"Attempt {attempt_number}/10  [{bar}]"

    # Guess history — all 10 slots, always shown
    history_lines = []
    for i in range(1, 11):
        if i - 1 < len(guess_history):
            g, c = guess_history[i - 1]
            label = "correct" if c == 1 else "correct"
            history_lines.append(f"  {str(i).rjust(2)}.  {g}   {c} correct")
        else:
            history_lines.append(f"  {str(i).rjust(2)}.  —")

    # Right panel: rules + template
    right = [
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

    left = ["  Guess history"] + history_lines + ["", ""]

    lines = [
        top_bar,
        row("  " + progress_label, ""),
        divider_row(),
    ]
    for i in range(13):
        lines.append(row(left[i], right[i]))
    lines.append("╰" + "─" * (L + 2) + "┴" + "─" * (R + 2) + "╯")

    for line in lines:
        print(line)


def validate_guess(user_input, guess_history=None):
    if guess_history is None:
        guess_history = []

    if len(user_input) != 5:
        raise ValueError("Input must be exactly 5 letters long.")

    allowed = {"A", "B", "C", "D", "E"}
    for letter in user_input:
        if not letter.isalpha():
            raise ValueError("Only alphabet letters are allowed. No numbers or symbols.")
        if letter not in allowed:
            raise ValueError("Only the letters A, B, C, D, and E are allowed.")

    seen = []
    for letter in user_input:
        if letter in seen:
            raise ValueError("Duplicate letters are not allowed. Use each letter only once.")
        seen.append(letter)

    previous = [g for g, _ in guess_history]
    if user_input in previous:
        raise ValueError(f"You already guessed {user_input}. Try a different combination.")


def main():
    letters = ["A", "B", "C", "D", "E"]
    random.shuffle(letters)
    answer = letters

    max_attempts = 10
    attempts_left = 10
    last_correct = None
    guess_history = []

    while attempts_left > 0:
        attempt_number = max_attempts - attempts_left + 1
        print("")
        print_header(attempt_number, last_correct, guess_history)
        print("")

        try:
            user_input = input("  › Your guess: ")
            user_input = user_input.strip().upper()

            validate_guess(user_input, guess_history)

            correct = sum(1 for i in range(5) if user_input[i] == answer[i])
            guess_history.append((user_input, correct))
            last_correct = correct
            attempts_left -= 1

            if correct == 5:
                print("")
                print("  ✓ Correct! The answer was " + "".join(answer) + ".")
                print("  Solved in " + str(max_attempts - attempts_left) + " attempt(s).")
                return

            print("")
            print(f"  {correct} letter(s) in the correct position.  {attempts_left} attempt(s) remaining.")

        except ValueError as error:
            print("")
            print("  ✗ Input error:", error)
            print("  Please follow the required format and try again.")

        except Exception:
            print("")
            print("  ✗ An unexpected error occurred. Please try again.")

    print("")
    print("  Out of attempts! The correct order was: " + "".join(answer))

main()
```
</details>

### 2. With Playground Framework
When developed inside the Playground constraints, the game is transformed into a formalized `GameEnv`. The `action_schema` is tightly defined to guide agent function execution automatically. The `step()` loop cleanly updates the state and returns clear text observations, freeing the game from specific rendering mechanisms—while still letting humans play it completely locally using the interactive CLI viewer.

<details>
<summary>Click to view Playground Framework implementation</summary>

```python
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
```
</details>

## Framework Features
1. **LLM Plug-and-Play Compatibility**: By defining JSON schemas for states and actions, any large-language model capable of parsing prompts and schema formatting can start playing your game automatically.
2. **Unified Action Schemas**: Clearly declare your agent API footprint using `action_schema` inside your `GameEnv` implementation. Actions are validated securely before execution.
3. **Multi-Agent Architecture**: Allows instantiation of numerous concurrent agents enabling identical parallel executions. Supercharge your benchmarks, QA bots, and synthetic data generation.
4. **Decoupled Mechanics from UI Renderers**: Implement `reset()` and `step()` cleanly on simple states, while the engine manages formatting observations for both humans (fancy CLI render tracking) and agents (lightweight context) behind the scenes.

## Setup Guide

To get started with Playground, you need to configure your environment to communicate with an agent backend. Follow these setup instructions:

1. **Clone the repository**
   ```bash
   git clone https://github.com/playground-ml/playground
   cd playground
   ```

2. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   Set up your `.env` configuration file to hook the internal agents up to OpenRouter.
   *Linux/macOS (.env file or direct export):*
   ```env
   OPENROUTER_API_KEY=API_KEY_HERE
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   OPENROUTER_MODEL=qwen/qwen3.5-35b-a3b
   ```
   *(Ensure you replace `API_KEY_HERE` with your actual OpenRouter API key).*

## Run Game

```bash
# Human plays (no LLM, no API key needed)
python run_game.py --game mastermind --batch 0

# Watch one LLM episode live
python run_game.py --game mastermind --batch 1

# Watch one LLM episode with debug console (streaming reasoning)
python run_game.py --game mastermind --batch 1 --debug

# 16 parallel silent episodes
python run_game.py --game mastermind --batch 16

# Override model / reasoning effort from CLI
python run_game.py --game mastermind --batch 4 --model openai/gpt-4o --reasoning-effort high
```

### CLI Flags

| Flag                 | Default                          | Description                                      |
|----------------------|----------------------------------|--------------------------------------------------|
| `--game`             | *(required)*                     | Game key (e.g. `mastermind`)                      |
| `--batch`            | `0`                              | `0` = human, `1` = watched LLM, `N≥2` = parallel |
| `--model`            | `OPENROUTER_MODEL` from `.env`   | Model identifier string                          |
| `--reasoning-effort` | `None`                           | `low` / `medium` / `high` — OpenRouter reasoning |
| `--seed`             | `None`                           | Base RNG seed (each parallel episode uses seed+i) |
| `--max-retries`      | `3`                              | Per-turn LLM retry budget                        |
| `--api-key`          | `OPENROUTER_API_KEY` from `.env` | API key override                                 |
| `--endpoint`         | `OPENROUTER_BASE_URL` from `.env`| Base URL override                                |
| `--provider`         | `openrouter`                     | Informational label stored in checkpoint          |
| `--debug`            | `False`                          | Open a debug console (requires `--batch 1`)       |

## Debug Mode

Debug mode opens a **second terminal window** that shows the complete LLM conversation in real-time — everything the model sees and thinks — while the main terminal displays only the game's visual output.

```bash
python run_game.py --game mastermind --batch 1 --debug
```

### What the Debug Console Shows

Each game turn is fully traced in the debug window:

1. **System Prompt** — The full system prompt sent once at episode start (game name, rules, JSON schema).
2. **Observation → LLM** — The game observation text sent to the model each turn (state, rules, schema hint, feedback).
3. **Model Reasoning** — Streaming reasoning tokens as they arrive (requires `--reasoning-effort`).
4. **Model Response** — The raw content response streamed token-by-token.
5. **Action Result** — Parsed JSON action, validation status, and latency.
6. **Retries & Errors** — Parse errors, validation errors, and retry attempts are shown in real-time.
7. **Game Errors** — When the game rejects an action (e.g. duplicate guess), the error is surfaced.
8. **Episode Result** — Final WIN/LOSS status with step count.

### How It Works

Debug mode uses a TCP socket connection between two terminal windows:
- The **main terminal** runs the game engine as normal.
- A **spawned `cmd` window** connects back to the main process and receives the debug stream.
- Reasoning and response tokens are streamed in real-time using the OpenAI streaming API.

> **Note:** `--debug` only works with `--batch 1` (watched mode). Human mode has no LLM to debug, and batch mode (N≥2) would produce interleaved output from parallel episodes.