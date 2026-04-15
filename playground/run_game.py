"""
run_game.py
~~~~~~~~~~~
Single entrypoint for all three game modes.

Usage
-----
# Human plays
python run_game.py --game mastermind --batch 0

# Watch one LLM episode live
python run_game.py --game mastermind --batch 1 \
    --model gpt-4o --provider openai

# Run 16 parallel LLM episodes silently
python run_game.py --game mastermind --batch 16 \
    --model gpt-4o --provider openai

# With a custom endpoint (e.g. DeepInfra, OpenRouter)
python run_game.py --game mastermind --batch 8 \
    --model meta-llama/Meta-Llama-3-70B-Instruct \
    --provider deepinfra \
    --endpoint https://api.deepinfra.com/v1/openai \
    --api-key $DEEPINFRA_API_KEY

# With reasoning effort (o-series models)
python run_game.py --game mastermind --batch 1 \
    --model o3-mini --reasoning-effort medium

Optional flags
--------------
--seed INT          Fix the RNG seed (applied to all episodes; each episode
                    increments by 1 so they're distinct but reproducible).
--max-retries INT   Per-turn LLM retry budget (default: 3).
--api-key STR       API key.  Falls back to OPENAI_API_KEY env var.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# make sure repo root is on the path when called from anywhere
sys.path.insert(0, os.path.dirname(__file__))

import playground as pg
from playground.registry import get_env_class


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Playground — run a text game in human, watched, or batch mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--game",    required=True,
                   help=f"Game key.  Available: {', '.join(pg.list_games())}")
    p.add_argument("--batch",   type=int, default=0, dest="batch_instance",
                   help="0=human  1=watched LLM  N>=2=parallel batch (default: 0)")

    # LLM config (required when --batch >= 1)
    p.add_argument("--model",            default=None)
    p.add_argument("--provider",         default=None,
                   help="Informational label stored in checkpoint (e.g. openai, deepinfra)")
    p.add_argument("--endpoint",         default=None,
                   help="Custom base URL for OpenAI-compatible endpoints")
    p.add_argument("--api-key",          default=None, dest="api_key")
    p.add_argument("--reasoning-effort", default=None,
                   choices=["low", "medium", "high"],
                   dest="reasoning_effort")
    p.add_argument("--max-retries",      type=int, default=3, dest="max_retries")

    # episode config
    p.add_argument("--seed", type=int, default=None,
                   help="Base RNG seed.  Each parallel episode uses seed+i.")
    return p.parse_args()


async def _main() -> None:
    args = _parse_args()

    # ── resolve game class ────────────────────────────────────────────────────
    try:
        env_cls = get_env_class(args.game)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── build env_factory ─────────────────────────────────────────────────────
    # Each parallel episode gets seed+i so episodes are distinct but
    # reproducible.  When no seed is given, each episode is random.
    if args.seed is not None:
        def env_factory(i: int = 0):
            return env_cls(seed=args.seed + i)
    else:
        def env_factory(i: int = 0):
            return env_cls()

    # Wrap so that batch workers can pass their index as seed offset
    if args.batch_instance >= 2:
        factories = [
            (lambda idx=i: env_factory(idx))
            for i in range(args.batch_instance)
        ]
    else:
        factories = [env_factory]

    # ── build AsyncOpenAI client ──────────────────────────────────────────────
    client = None
    if args.batch_instance >= 1:
        if not args.model:
            print("Error: --model is required when --batch >= 1", file=sys.stderr)
            sys.exit(1)
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(
                "Error: API key required.  Pass --api-key or set OPENAI_API_KEY.",
                file=sys.stderr,
            )
            sys.exit(1)
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=args.endpoint,  # None = default OpenAI endpoint
        )

    # ── dispatch ──────────────────────────────────────────────────────────────
    if args.batch_instance < 2:
        # single episode — factory doesn't need an index
        results = await pg.run(
            env_factory      = env_factory,
            batch_instance   = args.batch_instance,
            client           = client,
            model            = args.model,
            provider         = args.provider,
            endpoint         = args.endpoint,
            reasoning_effort = args.reasoning_effort,
            max_retries      = args.max_retries,
        )
    else:
        # parallel batch — run each factory concurrently
        from playground.agent import LLMAgent, build_system_prompt
        from playground.checkpoint import make_model_cfg
        from playground import runner as _runner, display as _display

        model_cfg = make_model_cfg(
            args.model, args.provider, args.endpoint, args.reasoning_effort
        )
        agent = LLMAgent(
            client=client,
            model=args.model,
            max_retries=args.max_retries,
            reasoning_effort=args.reasoning_effort,
        )

        tasks = [
            _runner._run_llm_batch(
                f, {}, agent, model_cfg, idx
            )
            for idx, f in enumerate(factories)
        ]
        results = await _runner._gather_with_progress(tasks, len(tasks))
        _display.print_batch_summary(results)

    # ── done ──────────────────────────────────────────────────────────────────
    if args.batch_instance <= 1:
        r = results[0]
        status = "WIN" if r["success"] else "LOSS"
        print(f"\n{status}  steps={r['total_steps']}  game_id={r['game_id']}")


if __name__ == "__main__":
    asyncio.run(_main())
