"""
playground.display
~~~~~~~~~~~~~~~~~~
Terminal rendering helpers.

Used by the runner in two situations:
  - BATCH_INSTANCE=0 : human plays, show the observation + prompt for input
  - BATCH_INSTANCE=1 : LLM plays, show observation + LLM action in real time

In batch mode (BATCH_INSTANCE>=2) nothing is printed to the terminal
except a one-line progress summary per completed episode.
"""

from __future__ import annotations

import textwrap


_WIDTH = 72


def _rule(char: str = "─") -> str:
    return char * _WIDTH


def _box_line(text: str, width: int = _WIDTH - 4) -> str:
    return f"│ {text:<{width}} │"


def print_episode_start(game_name: str, game_id: str, mode: str) -> None:
    print()
    print("╭" + _rule() + "╮")
    print(_box_line(f"  {game_name}"))
    print(_box_line(f"  mode: {mode}   id: {game_id[:8]}…"))
    print("╰" + _rule() + "╯")


def print_observation(observation: str, turn: int) -> None:
    print()
    print(f"── Turn {turn} " + "─" * (_WIDTH - 9))
    for line in observation.splitlines():
        print("  " + line)
    print()


def print_llm_action(raw_response: str, parsed_action: dict | None, latency_ms: int) -> None:
    print("  ▶ LLM response:")
    # indent and wrap the raw response
    for line in raw_response.splitlines():
        wrapped = textwrap.fill(line, width=_WIDTH - 6, subsequent_indent="      ")
        print("    " + wrapped)
    if parsed_action is not None:
        import json
        print(f"  ✓ Action: {json.dumps(parsed_action)}")
    print(f"  ⏱  {latency_ms} ms")
    print()


def print_turn_error(error: str) -> None:
    print(f"  ✗ {error}")
    print()


def print_episode_end(success: bool, total_steps: int, game_id: str) -> None:
    status = "WIN ✓" if success else "LOSS ✗"
    print()
    print(_rule("═"))
    print(f"  {status}   steps: {total_steps}   id: {game_id[:8]}…")
    print(_rule("═"))
    print()


def print_batch_progress(completed: int, total: int, success_count: int) -> None:
    pct = completed / total * 100
    bar_len = 30
    filled = int(bar_len * completed / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(
        f"\r  [{bar}] {pct:5.1f}%  "
        f"{completed}/{total} episodes  "
        f"wins: {success_count}",
        end="",
        flush=True,
    )
    if completed == total:
        print()  # newline on completion


def print_batch_summary(results: list[dict]) -> None:
    import statistics
    n = len(results)
    wins = [r for r in results if r["success"]]
    turns = [r["total_steps"] for r in results]
    print()
    print("╭" + _rule() + "╮")
    print(_box_line("  Batch results"))
    print(_box_line(f"  Episodes : {n}"))
    print(_box_line(f"  Win rate : {len(wins)/n*100:.1f}%  ({len(wins)}/{n})"))
    print(_box_line(f"  Avg turns: {statistics.mean(turns):.2f}  "
                    f"median: {statistics.median(turns):.1f}"))
    if wins:
        print(_box_line(f"  Avg turns (wins): {statistics.mean(r['total_steps'] for r in wins):.2f}"))
    print("╰" + _rule() + "╯")
    print()
