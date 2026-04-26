"""
playground.debug
~~~~~~~~~~~~~~~~
TCP-based debug console that shows the full LLM conversation in a
second terminal window.

Main process  →  opens TCP server on localhost:DEBUG_PORT
Spawned cmd   →  connects back and prints everything it receives.

Usage from run_game.py:

    # Main process (server side)
    from playground.debug import DebugConnection, spawn_debug_window
    spawn_debug_window(__file__)
    debug_conn = DebugConnection()
    debug_conn.start_server()          # blocks until client connects
    debug_conn.send(fmt_system_prompt(prompt))
    ...
    debug_conn.close()

    # Spawned process (client side)
    from playground.debug import debug_client_mode
    debug_client_mode()                # blocks, printing until EOF
"""

from __future__ import annotations

import os
import json
import socket
import time
from typing import Any


# ── constants ─────────────────────────────────────────────────────────────────

DEBUG_PORT = 54321


# ── ANSI helpers ──────────────────────────────────────────────────────────────

_RESET  = "\033[0m"
_DIM    = "\033[2m"
_BOLD   = "\033[1m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_BLUE   = "\033[34m"
_RED    = "\033[31m"
_MAGENTA = "\033[35m"
_WHITE  = "\033[37m"
_BG_DIM = "\033[48;5;235m"


# ── DebugConnection (server side — runs in main process) ──────────────────────

class DebugConnection:
    """TCP socket wrapper for sending debug text to the spawned terminal."""

    def __init__(self) -> None:
        self._server: socket.socket | None = None
        self._conn: socket.socket | None = None

    def start_server(self) -> None:
        """Bind, listen, and block until the debug client connects."""
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("localhost", DEBUG_PORT))
        self._server.listen(1)
        self._conn, _ = self._server.accept()

    def send(self, text: str) -> None:
        """Send text to the debug console.  Silently ignores broken pipes."""
        if self._conn is None:
            return
        try:
            self._conn.sendall(text.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def close(self) -> None:
        """Clean up sockets."""
        if self._conn:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
        if self._server:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None

    def __enter__(self) -> "DebugConnection":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def connected(self) -> bool:
        return self._conn is not None


# ── Debug client (runs in the spawned terminal) ──────────────────────────────

def debug_client_mode() -> None:
    """Entry point for the spawned debug terminal.

    Connects to the main process via TCP and prints everything it receives.
    Retries the connection for up to ~6 seconds.
    """
    print(f"{_CYAN}{_BOLD}")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║           🧠  Playground — Debug Console                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"{_RESET}")
    print(f"{_DIM}  Connecting to main process on port {DEBUG_PORT}...{_RESET}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for attempt in range(20):
        try:
            sock.connect(("localhost", DEBUG_PORT))
            break
        except ConnectionRefusedError:
            time.sleep(0.3)
    else:
        print(f"{_RED}  ✗ Could not connect to main process after 6 s.{_RESET}")
        return

    print(f"{_GREEN}  ✓ Connected!{_RESET}\n")

    try:
        while True:
            data = sock.recv(8192)
            if not data:
                break
            print(data.decode("utf-8"), end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

    print(f"\n{_DIM}  [debug session ended]{_RESET}\n")


# ── Spawner ───────────────────────────────────────────────────────────────────

def spawn_debug_window(script_path: str) -> None:
    """Open a new cmd window running this script in --debug-client mode."""
    import subprocess
    abs_path = os.path.abspath(script_path)
    subprocess.Popen(
        f'start cmd /k python "{abs_path}" --debug-client',
        shell=True,
    )


# ── Formatters ────────────────────────────────────────────────────────────────
#
# Each formatter returns a self-contained ANSI-coloured string that the
# debug console prints verbatim.

_LINE_W = 70


def _header(title: str, color: str = _CYAN) -> str:
    bar = "─" * (_LINE_W - len(title) - 2)
    return f"\n{color}{_BOLD}─── {title} {bar}{_RESET}\n"


def _footer(color: str = _CYAN) -> str:
    return f"{color}{'─' * (_LINE_W + 3)}{_RESET}\n"


def fmt_system_prompt(prompt: str) -> str:
    """Format the system prompt for the debug console."""
    lines = [_header("System Prompt", _CYAN)]
    for line in prompt.splitlines():
        lines.append(f"{_DIM}  {line}{_RESET}\n")
    lines.append(_footer(_CYAN))
    return "".join(lines)


def fmt_observation(turn: int, text: str) -> str:
    """Format the observation (user message) sent to the LLM."""
    lines = [_header(f"Turn {turn}: Observation → LLM", _GREEN)]
    for line in text.splitlines():
        lines.append(f"{_GREEN}  {line}{_RESET}\n")
    lines.append(_footer(_GREEN))
    return "".join(lines)


def fmt_reasoning_start() -> str:
    """Header printed once when reasoning tokens begin streaming."""
    return _header("Model Reasoning", _YELLOW)


def fmt_reasoning_chunk(text: str) -> str:
    """A single chunk of streaming reasoning text."""
    return f"{_DIM}{_YELLOW}{text}{_RESET}"


def fmt_reasoning_end() -> str:
    """Footer printed when reasoning is complete."""
    return "\n" + _footer(_YELLOW)


def fmt_response_start() -> str:
    """Header printed once when content tokens begin streaming."""
    return _header("Model Response", _BLUE)


def fmt_response_chunk(text: str) -> str:
    """A single chunk of streaming content text."""
    return f"{_WHITE}{text}{_RESET}"


def fmt_response_end() -> str:
    """Footer printed when the response is complete."""
    return "\n" + _footer(_BLUE)


def fmt_action_result(
    parsed: dict | None,
    valid: bool,
    error: str | None,
    latency_ms: int,
) -> str:
    """Format the action extraction / validation result."""
    lines = [_header("Action Result", _MAGENTA)]
    if parsed is not None:
        icon = "✓" if valid else "✗"
        color = _GREEN if valid else _RED
        lines.append(f"{color}  {icon}  {json.dumps(parsed)}{_RESET}\n")
    else:
        lines.append(f"{_RED}  ✗  Could not extract valid action{_RESET}\n")
    if error:
        lines.append(f"{_RED}  ✗  {error}{_RESET}\n")
    lines.append(f"{_DIM}  ⏱  {latency_ms} ms{_RESET}\n")
    lines.append(_footer(_MAGENTA))
    return "".join(lines)


def fmt_retry(attempt: int, max_retries: int, error: str) -> str:
    """Format a retry message inside the agent loop."""
    return (
        f"\n{_RED}{_BOLD}  ↻ Retry {attempt}/{max_retries}: {error}{_RESET}\n"
    )


def fmt_game_error(error: str) -> str:
    """Format a game-level rejection (ValueError from env.step)."""
    lines = [_header("Game Error", _RED)]
    lines.append(f"{_RED}  ✗  {error}{_RESET}\n")
    lines.append(_footer(_RED))
    return "".join(lines)


def fmt_episode_end(success: bool, steps: int, game_id: str) -> str:
    """Format the final episode result."""
    color = _GREEN if success else _RED
    icon = "WIN ✓" if success else "LOSS ✗"
    lines = [
        f"\n{color}{_BOLD}",
        "═" * (_LINE_W + 3) + "\n",
        f"  {icon}   steps: {steps}   id: {game_id[:8]}…\n",
        "═" * (_LINE_W + 3) + "\n",
        f"{_RESET}\n",
    ]
    return "".join(lines)


def fmt_token_usage(prompt_tokens: int, completion_tokens: int, total_tokens: int) -> str:
    """Format token usage stats from a streaming response."""
    return (
        f"{_DIM}  [tokens — prompt: {prompt_tokens} | "
        f"completion: {completion_tokens} | "
        f"total: {total_tokens}]{_RESET}\n"
    )
