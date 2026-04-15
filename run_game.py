"""
run_game.py
~~~~~~~~~~~
Single entrypoint for all three game modes.

Reads credentials from .env in the same directory.
CLI flags override .env values when provided.

Usage
-----
# Human plays (no LLM, no API key needed)
python run_game.py --game mastermind --batch 0

# Watch one LLM episode live (uses .env defaults)
python run_game.py --game mastermind --batch 1

# 16 parallel silent episodes
python run_game.py --game mastermind --batch 16

# Override model/endpoint from CLI
python run_game.py --game mastermind --batch 4 --model openai/gpt-4o

Optional flags
--------------
--model STR           Override OPENROUTER_MODEL from .env
--reasoning-effort    low | medium | high  (enables OpenRouter reasoning)
--seed INT            Base RNG seed (each parallel episode uses seed+i)
--max-retries INT     Per-turn LLM retry budget (default: 3)
--api-key STR         Override OPENROUTER_API_KEY from .env
--endpoint STR        Override OPENROUTER_BASE_URL from .env
--provider STR        Informational label stored in checkpoint
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# load .env before any os.environ reads
from pathlib import Path
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)

sys.path.insert(0, os.path.dirname(__file__))

import playground as pg
from playground.registry import get_env_class


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Playground — run a text game in human, watched, or batch mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--game", required=True,
                   help=f"Game key.  Available: {', '.join(pg.list_games())}")
    p.add_argument("--batch", type=int, default=0, dest="batch_instance",
                   help="0=human  1=watched LLM  N>=2=parallel batch  (default: 0)")
    p.add_argument("--model",            default=None,
                   help="Model name.  Defaults to OPENROUTER_MODEL in .env")
    p.add_argument("--provider",         default="openrouter",
                   help="Informational provider label stored in checkpoint")
    p.add_argument("--endpoint",         default=None,
                   help="Base URL.  Defaults to OPENROUTER_BASE_URL in .env")
    p.add_argument("--api-key",          default=None, dest="api_key",
                   help="API key.  Defaults to OPENROUTER_API_KEY in .env")
    p.add_argument("--reasoning-effort", default=None,
                   choices=["low", "medium", "high"],
                   dest="reasoning_effort",
                   help="Enable OpenRouter reasoning with given effort level")
    p.add_argument("--max-retries",      type=int, default=3, dest="max_retries")
    p.add_argument("--seed",             type=int, default=None)
    return p.parse_args()


async def _main() -> None:
    args = _parse_args()

    # ── resolve game class ────────────────────────────────────────────────────
    try:
        env_cls = get_env_class(args.game)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── resolve config: CLI flags > .env ─────────────────────────────────────
    api_key  = args.api_key  or os.environ.get("OPENROUTER_API_KEY")
    endpoint = args.endpoint or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model    = args.model    or os.environ.get("OPENROUTER_MODEL")

    # ── env factory ───────────────────────────────────────────────────────────
    def env_factory(i: int = 0):
        if args.seed is not None:
            return env_cls(seed=args.seed + i)
        return env_cls()

    # ── build AsyncOpenAI client (skip for human mode) ───────────────────────
    client = None
    if args.batch_instance >= 1:
        if not model:
            print(
                "Error: model required.  Pass --model or set OPENROUTER_MODEL in .env",
                file=sys.stderr,
            )
            sys.exit(1)
        if not api_key:
            print(
                "Error: API key required.  Pass --api-key or set OPENROUTER_API_KEY in .env",
                file=sys.stderr,
            )
            sys.exit(1)
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url=endpoint)

    # ── single episode (human or watched) ────────────────────────────────────
    if args.batch_instance < 2:
        results = await pg.run(
            env_factory      = env_factory,
            batch_instance   = args.batch_instance,
            client           = client,
            model            = model,
            provider         = args.provider,
            endpoint         = endpoint,
            reasoning_effort = args.reasoning_effort,
            max_retries      = args.max_retries,
        )
        r = results[0]
        status = "WIN" if r["success"] else "LOSS"
        print(f"\n{status}  steps={r['total_steps']}  game_id={r['game_id']}")
        return

    # ── parallel batch ────────────────────────────────────────────────────────
    from playground.agent import LLMAgent
    from playground.checkpoint import make_model_cfg
    from playground import runner as _runner, display as _display

    model_cfg = make_model_cfg(model, args.provider, endpoint, args.reasoning_effort)
    agent = LLMAgent(
        client=client,
        model=model,
        max_retries=args.max_retries,
        reasoning_effort=args.reasoning_effort,
    )
    tasks = [
        _runner._run_llm_batch(
            (lambda idx=i: env_factory(idx)), {}, agent, model_cfg, i
        )
        for i in range(args.batch_instance)
    ]
    results = await _runner._gather_with_progress(tasks, len(tasks))
    _display.print_batch_summary(results)


if __name__ == "__main__":
    asyncio.run(_main())
