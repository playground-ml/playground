"""
playground.checkpoint
~~~~~~~~~~~~~~~~~~~~~
Append-only JSONL checkpoint writer.

One file per episode:
  game_checkpoints/{game_name}/{game_id}.jsonl

Each file contains three record types written in order:
  Line 1      : game_header   (written by open())
  Lines 2..N  : step          (written by write_step() after every turn)
  Last line   : game_footer   (written by close())

All timestamps are UTC ISO-8601.  All writes are flushed immediately so
a crash mid-game loses at most one in-flight API response.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHECKPOINTS_ROOT = Path("game_checkpoints")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _new_id() -> str:
    return str(uuid.uuid4())


class CheckpointWriter:
    """Async-safe append-only writer for one game episode.

    Usage
    -----
    writer = await CheckpointWriter.open(game_name, mode, model_cfg, env_cfg, system_prompt)
    await writer.write_step(step_record)
    await writer.close(success=True, total_steps=7)
    """

    def __init__(self, path: Path, game_id: str, started_at: str) -> None:
        self._path = path
        self._game_id = game_id
        self._started_at = started_at
        self._file: Any = None
        self._lock = asyncio.Lock()

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    async def open(
        cls,
        game_name: str,
        mode: str,                        # "human" | "llm_watched" | "llm_batch"
        batch_instance_index: int,
        model_cfg: dict[str, Any] | None, # None when mode="human"
        env_cfg: dict[str, Any],
        system_prompt: str,
    ) -> "CheckpointWriter":
        game_id = _new_id()
        started_at = _now()

        folder = CHECKPOINTS_ROOT / game_name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{game_id}.jsonl"

        writer = cls(path, game_id, started_at)
        writer._file = open(path, "w", encoding="utf-8")  # noqa: WPS515

        header: dict[str, Any] = {
            "record_type": "game_header",
            "game_id": game_id,
            "game_name": game_name,
            "started_at": started_at,
            "mode": mode,
            "batch_instance_index": batch_instance_index,
            "model": model_cfg,
            "system_prompt": system_prompt,
            "env_config": env_cfg,
        }
        writer._write_line(header)
        return writer

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def game_id(self) -> str:
        return self._game_id

    async def write_step(self, record: dict[str, Any]) -> None:
        """Append one step record.  Thread-safe via asyncio lock."""
        async with self._lock:
            record["record_type"] = "step"
            record["game_id"] = self._game_id
            if "step_id" not in record:
                record["step_id"] = _new_id()
            if "timestamp" not in record:
                record["timestamp"] = _now()
            self._write_line(record)

    async def close(self, success: bool, total_steps: int) -> None:
        """Write the game_footer and flush/close the file."""
        footer: dict[str, Any] = {
            "record_type": "game_footer",
            "game_id": self._game_id,
            "ended_at": _now(),
            "total_steps": total_steps,
            "success": success,
        }
        async with self._lock:
            self._write_line(footer)
            if self._file:
                self._file.close()
                self._file = None

    # ── internal ──────────────────────────────────────────────────────────────

    def _write_line(self, obj: dict[str, Any]) -> None:
        if self._file is None:
            raise RuntimeError("CheckpointWriter is closed.")
        self._file.write(json.dumps(obj, ensure_ascii=False, default=str))
        self._file.write("\n")
        self._file.flush()


# ── convenience: build model_cfg dict from run_game config ────────────────────

def make_model_cfg(
    name: str | None,
    provider: str | None,
    endpoint: str | None,
    reasoning_effort: str | None,
) -> dict[str, Any] | None:
    if name is None:
        return None
    return {
        "name": name,
        "provider": provider,
        "endpoint": endpoint,
        "reasoning_effort": reasoning_effort,
    }
