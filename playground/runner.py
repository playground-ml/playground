"""
playground.runner
~~~~~~~~~~~~~~~~~
Async episode runner.  Handles all three modes via BATCH_INSTANCE:

  BATCH_INSTANCE = 0   Human plays in the terminal.
                       LLM is not involved.  model_cfg must be None.

  BATCH_INSTANCE = 1   LLM plays one episode, rendered live in the terminal.
                       Developer watches every observation and action.

  BATCH_INSTANCE >= 2  LLM plays N episodes in parallel (asyncio.gather).
                       Nothing is printed except a progress bar and summary.

All modes write a JSONL checkpoint file.

When a debug_conn (DebugConnection) is provided in watched mode, the full
LLM conversation — observations, streaming reasoning, responses, retries,
and game errors — is sent to the debug console while the main terminal
shows only the game's visual render_screen output.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from openai import AsyncOpenAI

from playground.core import GameEnv, StepResult
from playground.agent import LLMAgent, AgentResult, build_system_prompt
from playground.checkpoint import CheckpointWriter, make_model_cfg
from playground import display


# ── public entry point ────────────────────────────────────────────────────────

async def run(
    env_factory,            # callable() → GameEnv
    batch_instance: int,    # 0 = human, 1 = watched LLM, >=2 = batch LLM
    client: AsyncOpenAI | None = None,
    model: str | None = None,
    provider: str | None = None,
    endpoint: str | None = None,
    reasoning_effort: str | None = None,
    max_retries: int = 3,
    env_kwargs: dict | None = None,
    debug_conn: Any = None,
) -> list[dict]:
    """Run one or more episodes and return a list of summary dicts.

    Parameters
    ----------
    env_factory       : zero-arg callable that returns a fresh GameEnv.
                        For seeded games pass a lambda: lambda: MyGame(seed=i)
    batch_instance    : 0 = human, 1 = watched, >=2 = parallel batch count
    client            : AsyncOpenAI client (required when batch_instance >= 1)
    model             : model identifier (required when batch_instance >= 1)
    provider          : informational, stored in checkpoint
    endpoint          : informational, stored in checkpoint
    reasoning_effort  : "low" | "medium" | "high" | None
    max_retries       : per-turn LLM retry budget
    env_kwargs        : passed to env_factory if it accepts kwargs
    debug_conn        : DebugConnection or None — when set, streams the full
                        LLM conversation to the debug console (batch 1 only)
    """
    env_kwargs = env_kwargs or {}

    if batch_instance == 0:
        result = await _run_human(env_factory, env_kwargs)
        return [result]

    if client is None or model is None:
        raise ValueError("client and model are required when batch_instance >= 1")

    agent = LLMAgent(
        client=client,
        model=model,
        max_retries=max_retries,
        reasoning_effort=reasoning_effort,
    )
    model_cfg = make_model_cfg(model, provider, endpoint, reasoning_effort)

    if batch_instance == 1:
        result = await _run_llm_watched(
            env_factory, env_kwargs, agent, model_cfg, debug_conn=debug_conn,
        )
        return [result]

    # batch mode
    tasks = [
        _run_llm_batch(env_factory, env_kwargs, agent, model_cfg, idx)
        for idx in range(batch_instance)
    ]
    results = await _gather_with_progress(tasks, batch_instance)
    display.print_batch_summary(results)
    return results


# ── mode implementations ──────────────────────────────────────────────────────

async def _run_human(env_factory, env_kwargs: dict) -> dict:
    env: GameEnv = env_factory(**env_kwargs)
    writer = await CheckpointWriter.open(
        game_name            = env.name,
        mode                 = "human",
        batch_instance_index = 0,
        model_cfg            = None,
        env_cfg              = env_kwargs,
        system_prompt        = "",
    )
    display.print_episode_start(env.name, writer.game_id, "human")

    observation = env.reset()
    step_index = 0
    success = False

    while True:
        step_index += 1
        display.print_game_screen(env, step_index)

        # get human input
        action_schema = env.action_schema
        hint = _schema_one_liner(action_schema)
        print(f"  Format: {hint}")
        try:
            raw = input("  › Your action (JSON): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  [interrupted]")
            break

        # parse
        try:
            action = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"  ✗ Not valid JSON: {exc}")
            step_index -= 1  # don't count malformed input as a turn
            continue

        # game-level validation
        action_valid = True
        error_message = None
        result: StepResult | None = None
        try:
            result = env.step(action)
        except ValueError as exc:
            action_valid = False
            error_message = str(exc)
            print(f"  ✗ Invalid move: {exc}")
            step_index -= 1
            continue

        # write step
        await writer.write_step({
            "step_index"    : step_index,
            "observation"   : observation,
            "llm"           : None,
            "human_input"   : raw,
            "parsed_action" : action,
            "action_valid"  : action_valid,
            "error_message" : error_message,
            "step_result"   : {
                "done"    : result.done,
                "success" : result.success,
                "info"    : result.info,
            },
        })

        observation = result.observation

        if result.done:
            success = result.success
            display.print_game_screen(env, step_index)
            display.print_episode_end(success, step_index, writer.game_id)
            break

    await writer.close(success=success, total_steps=step_index)
    return {"game_id": writer.game_id, "success": success, "total_steps": step_index}


async def _run_llm_watched(
    env_factory, env_kwargs: dict, agent: LLMAgent,
    model_cfg: dict, debug_conn: Any = None,
) -> dict:
    env: GameEnv = env_factory(**env_kwargs)
    system_prompt = build_system_prompt(env.name, env.action_schema)
    writer = await CheckpointWriter.open(
        game_name            = env.name,
        mode                 = "llm_watched",
        batch_instance_index = 0,
        model_cfg            = model_cfg,
        env_cfg              = env_kwargs,
        system_prompt        = system_prompt,
    )
    display.print_episode_start(env.name, writer.game_id, "llm_watched")

    result = await _episode_loop(
        env, agent, system_prompt, writer, verbose=True, debug_conn=debug_conn,
    )
    display.print_episode_end(result["success"], result["total_steps"], writer.game_id)
    return result


async def _run_llm_batch(
    env_factory, env_kwargs: dict, agent: LLMAgent,
    model_cfg: dict, instance_index: int,
) -> dict:
    env: GameEnv = env_factory(**env_kwargs)
    system_prompt = build_system_prompt(env.name, env.action_schema)
    writer = await CheckpointWriter.open(
        game_name            = env.name,
        mode                 = "llm_batch",
        batch_instance_index = instance_index,
        model_cfg            = model_cfg,
        env_cfg              = env_kwargs,
        system_prompt        = system_prompt,
    )
    return await _episode_loop(env, agent, system_prompt, writer, verbose=False)


# ── shared episode loop ───────────────────────────────────────────────────────

async def _episode_loop(
    env: GameEnv,
    agent: LLMAgent,
    system_prompt: str,
    writer: CheckpointWriter,
    verbose: bool,
    debug_conn: Any = None,
) -> dict:
    """Core turn loop shared by watched and batch modes.

    When debug_conn is provided the full LLM conversation is streamed to
    the debug console: system prompt, observations, reasoning tokens,
    response tokens, action results, retries, and game errors.
    """
    # ── send system prompt to debug console ───────────────────────────────
    if debug_conn is not None:
        from playground.debug import fmt_system_prompt
        debug_conn.send(fmt_system_prompt(system_prompt))

    observation = env.reset()
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    step_index = 0
    success = False

    while True:
        step_index += 1
        if verbose:
            display.print_game_screen(env, step_index)

        messages.append({"role": "user", "content": observation})

        # ── send observation to debug console ─────────────────────────────
        if debug_conn is not None:
            from playground.debug import fmt_observation
            debug_conn.send(fmt_observation(step_index, observation))

        # call LLM (streaming goes to debug console automatically)
        t_step = time.perf_counter()
        agent_result: AgentResult = await agent.call(
            messages, env.action_schema, debug_conn=debug_conn,
        )
        step_elapsed_ms = int((time.perf_counter() - t_step) * 1000)

        # ── send action result to debug console ───────────────────────────
        if debug_conn is not None:
            from playground.debug import fmt_action_result
            debug_conn.send(fmt_action_result(
                parsed     = agent_result.parsed_action,
                valid      = agent_result.action_valid,
                error      = agent_result.error_message,
                latency_ms = agent_result.latency_ms,
            ))

        if verbose:
            if agent_result.reasoning:
                model_name = getattr(agent, "_model", "")
                display.print_reasoning_box(agent_result.reasoning, model_name)
            display.print_action_box(
                raw_response   = agent_result.raw_response,
                parsed_action  = agent_result.parsed_action,
                latency_ms     = agent_result.latency_ms,
                action_valid   = agent_result.action_valid,
                error_message  = agent_result.error_message,
            )

        # append assistant message to history
        messages.append({"role": "assistant", "content": agent_result.raw_response})

        # step the game (only if we have a valid action)
        action_valid = agent_result.action_valid
        error_message = agent_result.error_message
        step_result: StepResult | None = None

        if agent_result.parsed_action is not None:
            try:
                step_result = env.step(agent_result.parsed_action)
                action_valid = True
                error_message = None
            except ValueError as exc:
                action_valid = False
                error_message = str(exc)
                if verbose:
                    display.print_turn_error(f"Game rejected action: {exc}")
                # ── send game error to debug console ──────────────────────
                if debug_conn is not None:
                    from playground.debug import fmt_game_error
                    debug_conn.send(fmt_game_error(str(exc)))
                # feed game error back to model
                messages.append({
                    "role": "user",
                    "content": f"Invalid move: {exc}  Please try again.",
                })
        else:
            if verbose:
                display.print_turn_error(agent_result.error_message or "No valid action.")

        # write step checkpoint — always, even on error
        await writer.write_step({
            "step_index"    : step_index,
            "observation"   : observation,
            "llm": {
                "messages_sent"  : agent_result.messages_sent,
                "reasoning"      : agent_result.reasoning,
                "raw_response"   : agent_result.raw_response,
                "action_raw"     : agent_result.action_raw,
                "latency_ms"     : agent_result.latency_ms,
            },
            "human_input"   : None,
            "parsed_action" : agent_result.parsed_action,
            "action_valid"  : action_valid,
            "error_message" : error_message,
            "step_result": {
                "done"    : step_result.done    if step_result else False,
                "success" : step_result.success if step_result else False,
                "info"    : step_result.info    if step_result else {},
            },
        })

        if step_result is None:
            # couldn't get a valid action after all retries — forfeit
            break

        observation = step_result.observation

        if step_result.done:
            success = step_result.success
            break

    # ── send episode end to debug console ─────────────────────────────────
    if debug_conn is not None:
        from playground.debug import fmt_episode_end
        debug_conn.send(fmt_episode_end(success, step_index, writer.game_id))

    await writer.close(success=success, total_steps=step_index)
    return {
        "game_id"     : writer.game_id,
        "success"     : success,
        "total_steps" : step_index,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

async def _gather_with_progress(tasks, total: int) -> list[dict]:
    results: list[dict] = []
    success_count = 0

    async def _tracked(coro, idx):
        nonlocal success_count
        result = await coro
        results.append(result)
        if result["success"]:
            success_count += 1
        display.print_batch_progress(len(results), total, success_count)
        return result

    await asyncio.gather(*[_tracked(t, i) for i, t in enumerate(tasks)])
    return results


def _schema_one_liner(schema: dict) -> str:
    props = schema.get("properties", {})
    example = {k: f"<{v.get('description', k)}>" for k, v in props.items()}
    return json.dumps(example, ensure_ascii=False)
