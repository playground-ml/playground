"""
playground.display
~~~~~~~~~~~~~~~~~~
Terminal rendering for human and llm_watched modes.

Box geometry
------------
All panels share the same outer width (86 chars including borders) so that
the reasoning box, the game screen, and the action box all align visually
in the terminal.

  BATCH_INSTANCE = 0  (human)
    print_game_screen()   →  the game's render_screen() output
    (input prompt handled by runner)

  BATCH_INSTANCE = 1  (llm_watched)
    print_game_screen()   →  game screen before the LLM turn
    print_reasoning_box() →  model reasoning (if present)
    print_action_box()    →  parsed action + latency

  BATCH_INSTANCE >= 2  (batch)
    print_batch_progress() + print_batch_summary() only
"""

from __future__ import annotations

import textwrap

# ── shared geometry ───────────────────────────────────────────────────────────
# Inner content width.  Must be wide enough for the Mastermind game box
# (L + R + 5 inner chars = 36 + 44 + 5 = 85), plus 2 border chars = 87 total.
# We use W = 85 so the outer border is exactly 87 chars wide, matching the game.
_W = 85          # inner width (between the │ borders)


def _pad(text: str, width: int = _W) -> str:
    """Left-align *text*, truncate or pad to *width*."""
    if len(text) > width:
        return text[: width - 1] + "…"
    return text + " " * (width - len(text))


def _top(title: str = "", char: str = "─") -> str:
    if not title:
        return "╭" + char * (_W + 2) + "╮"
    title = f" {title} "
    pad = (_W + 2 - len(title)) // 2
    return "╭" + char * pad + title + char * (_W + 2 - pad - len(title)) + "╮"


def _bottom(char: str = "─") -> str:
    return "╰" + char * (_W + 2) + "╯"


def _mid_divider() -> str:
    return "├" + "─" * (_W + 2) + "┤"


def _row(text: str = "") -> str:
    return "│ " + _pad(text, _W) + " │"


def _wrap_rows(text: str, indent: str = "  ") -> list[str]:
    """Wrap *text* to _W, returning box rows."""
    rows: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            rows.append(_row())
            continue
        wrapped = textwrap.wrap(
            raw_line,
            width=_W - len(indent),
            subsequent_indent=indent,
        )
        for wl in wrapped:
            rows.append(_row(indent + wl))
    return rows or [_row()]


# ── episode start / end ───────────────────────────────────────────────────────

def print_episode_start(game_name: str, game_id: str, mode: str) -> None:
    print()
    print(_top())
    print(_row(f"  {game_name}"))
    print(_row(f"  mode: {mode}   id: {game_id[:8]}…"))
    print(_bottom())


def print_episode_end(success: bool, total_steps: int, game_id: str) -> None:
    status = "WIN ✓" if success else "LOSS ✗"
    char = "═"
    print()
    print("╭" + char * (_W + 2) + "╮")
    print(_row(f"  {status}   steps: {total_steps}   id: {game_id[:8]}…"))
    print("╰" + char * (_W + 2) + "╯")
    print()


# ── game screen ───────────────────────────────────────────────────────────────

def print_game_screen(env, turn: int) -> None:
    """Print the game's visual screen if it exposes render_screen(),
    otherwise fall back to printing the plain observation text.
    """
    print()
    print(f"  ── Turn {turn} " + "─" * (_W - 9))
    print()

    if hasattr(env, "render_screen"):
        # indent each line of the game box by 2 spaces so it sits inside
        # the turn divider visually without adding extra borders
        for line in env.render_screen().splitlines():
            print("  " + line)
    else:
        # fallback: plain observation (used by games that don't have render_screen)
        for line in env._build_obs().splitlines():
            print("  " + line)
    print()


# ── reasoning box ─────────────────────────────────────────────────────────────

def print_reasoning_box(reasoning: str, model_name: str = "") -> None:
    """Print the model's reasoning inside a dedicated box.

    Shown only in llm_watched mode when the model returns reasoning text.
    """
    label = f"Model reasoning  {model_name}".strip()
    print(_top(label))
    print(_row())
    for row_line in _wrap_rows(reasoning, indent="  "):
        print(row_line)
    print(_row())
    print(_bottom())
    print()


# ── action box ────────────────────────────────────────────────────────────────

def print_action_box(
    raw_response: str,
    parsed_action: dict | None,
    latency_ms: int,
    action_valid: bool = True,
    error_message: str | None = None,
) -> None:
    """Print the model's action (and raw response if action extraction failed)."""
    import json

    print(_top("Model action"))
    print(_row())

    if parsed_action is not None:
        print(_row(f"  ✓  {json.dumps(parsed_action)}"))
    else:
        # show raw response when we couldn't parse a valid action
        print(_row("  ✗  Could not extract valid action"))
        print(_row())
        for row_line in _wrap_rows(raw_response, indent="  "):
            print(row_line)

    if error_message:
        print(_row())
        print(_row(f"  ✗  {error_message}"))

    print(_row())
    print(_mid_divider())
    status = "✓  action valid" if action_valid else "✗  action invalid"
    print(_row(f"  {status}   ⏱  {latency_ms} ms"))
    print(_bottom())
    print()


# ── turn error ────────────────────────────────────────────────────────────────

def print_turn_error(error: str) -> None:
    print(_top("Game error"))
    print(_row(f"  ✗  {error}"))
    print(_bottom())
    print()


# ── batch progress / summary ─────────────────────────────────────────────────

def print_batch_progress(completed: int, total: int, success_count: int) -> None:
    bar_len = 30
    filled = int(bar_len * completed / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = completed / total * 100
    print(
        f"\r  [{bar}] {pct:5.1f}%  "
        f"{completed}/{total} episodes  "
        f"wins: {success_count}",
        end="",
        flush=True,
    )
    if completed == total:
        print()


def print_batch_summary(results: list[dict]) -> None:
    import statistics
    n = len(results)
    wins = [r for r in results if r["success"]]
    turns = [r["total_steps"] for r in results]
    print()
    print(_top("Batch results"))
    print(_row(f"  Episodes : {n}"))
    print(_row(f"  Win rate : {len(wins)/n*100:.1f}%  ({len(wins)}/{n})"))
    print(_row(
        f"  Avg turns: {statistics.mean(turns):.2f}  "
        f"median: {statistics.median(turns):.1f}"
    ))
    if wins:
        print(_row(
            f"  Avg turns (wins): "
            f"{statistics.mean(r['total_steps'] for r in wins):.2f}"
        ))
    print(_bottom())
    print()
