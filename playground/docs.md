# Playground Engine — Module Documentation

This document describes every module in the `playground/` package: what it does, its key classes and functions, and how the modules connect.

---

## `__init__.py`

**Purpose:** Package entry point. Re-exports the public API so users can write `import playground as pg`.

**Public exports:**
- `pg.GameEnv` — base class for games (from `core.py`)
- `pg.StepResult` — dataclass returned by `GameEnv.step()` (from `core.py`)
- `pg.run()` — the main async entry point to run episodes (from `runner.py`)
- `pg.get_env_class(key)` — look up a game class by registry key (from `registry.py`)
- `pg.list_games()` — list all registered game keys (from `registry.py`)

---

## `core.py`

**Purpose:** Defines the two foundational types every game implements against.

### `StepResult` (dataclass)
Returned by every call to `GameEnv.step()`.

| Field         | Type            | Description                                         |
|---------------|-----------------|-----------------------------------------------------|
| `observation` | `str`           | Text the model sees at the next turn's start        |
| `done`        | `bool`          | `True` when the episode has ended                   |
| `success`     | `bool`          | `True` only on a win (requires `done=True`)         |
| `info`        | `dict[str,Any]` | Arbitrary metadata — never shown to the model       |

### `GameEnv` (ABC)
Abstract base class for all Playground games. Subclasses must implement:

| Method / Property   | Returns  | Description                                           |
|---------------------|----------|-------------------------------------------------------|
| `name`              | `str`    | Short display name for prompts and logs               |
| `action_schema`     | `dict`   | JSON Schema (draft-07) for one valid action           |
| `reset()`           | `str`    | Start a fresh episode, return the first observation   |
| `step(action)`      | `StepResult` | Apply an action, update state, return the result  |
| `_render_state()`   | `str`    | Current board/world snapshot (no rules)               |
| `_render_rules()`   | `str`    | Complete rules text sent every turn                   |

Engine-provided methods (do not override):
- `_build_obs(message=None)` — Assembles the full observation string from `_render_state()`, `_render_rules()`, and the auto-generated schema hint.
- `_schema_hint()` — Auto-generates a one-line JSON example from `action_schema`.

---

## `agent.py`

**Purpose:** Async LLM agent that wraps the OpenAI `AsyncClient` to call the model, extract JSON actions, validate them, and retry on failure.

### `AgentResult` (dataclass)
Everything produced by one LLM turn.

| Field           | Type            | Description                                      |
|-----------------|-----------------|--------------------------------------------------|
| `raw_response`  | `str`           | Complete text returned by the API                |
| `reasoning`     | `str \| None`   | Reasoning text (o-series / reasoning models)     |
| `action_raw`    | `str \| None`   | JSON string extracted from the response          |
| `parsed_action` | `dict \| None`  | Validated dict, or `None` on failure             |
| `action_valid`  | `bool`          | Whether the action passed validation             |
| `error_message` | `str \| None`   | Error description if action is invalid           |
| `latency_ms`    | `int`           | Wall-clock time for the API call(s)              |
| `messages_sent` | `list[dict]`    | Full messages array sent to the API              |

### `LLMAgent` class
| Method                      | Description                                              |
|-----------------------------|----------------------------------------------------------|
| `__init__(client, model, max_retries, reasoning_effort)` | Configure the agent        |
| `call(messages, action_schema, debug_conn=None)` | Make one LLM call with retries, return `AgentResult`. When `debug_conn` is provided, uses streaming and sends tokens to the debug console in real-time. |
| `_api_call(messages)` | Internal non-streaming API call                                    |
| `_api_call_streaming(messages, debug_conn)` | Internal streaming API call — sends reasoning and content tokens to the debug console as they arrive |

### Standalone functions
- `build_system_prompt(game_name, action_schema)` — Builds the system prompt the engine sends once per episode.
- `_extract_json(text)` — Extracts the first JSON object from model output (prefers fenced blocks, falls back to bare braces).
- `_validate(action, schema)` — Validates a dict against a JSON Schema.

---

## `runner.py`

**Purpose:** Async episode runner. Handles all three play modes and writes JSONL checkpoints.

### `run()` (public entry point)
```python
async def run(
    env_factory, batch_instance, client=None, model=None,
    provider=None, endpoint=None, reasoning_effort=None,
    max_retries=3, env_kwargs=None, debug_conn=None,
) -> list[dict]
```

| `batch_instance` | Mode         | Description                                    |
|-------------------|-------------|------------------------------------------------|
| `0`               | Human       | Human plays in the terminal, no LLM            |
| `1`               | Watched     | LLM plays, rendered live in terminal            |
| `≥ 2`             | Batch       | N parallel LLM episodes, progress bar only      |

### Internal functions
| Function                 | Description                                                      |
|--------------------------|------------------------------------------------------------------|
| `_run_human()`           | Human-interactive episode with `input()` prompts                 |
| `_run_llm_watched()`     | Single watched LLM episode with optional debug console           |
| `_run_llm_batch()`       | Single silent LLM episode (used in parallel)                     |
| `_episode_loop()`        | Core turn loop shared by watched and batch modes. When `debug_conn` is provided, streams system prompt, observations, action results, game errors, and episode end to the debug console. |
| `_gather_with_progress()`| Runs N tasks with a live progress bar                            |

---

## `display.py`

**Purpose:** Terminal rendering for human and `llm_watched` modes. All visual output in the main terminal passes through this module.

### Geometry
All panels share a fixed outer width (87 chars including borders) so game screens, reasoning boxes, and action boxes align visually.

### Functions
| Function                  | Used in       | Description                                    |
|---------------------------|---------------|------------------------------------------------|
| `print_episode_start()`   | All modes     | Game name + mode + ID box at episode start     |
| `print_episode_end()`     | Watched/Human | WIN/LOSS summary box                           |
| `print_game_screen()`     | Watched/Human | Renders `env.render_screen()` or `_build_obs()`|
| `print_reasoning_box()`   | Watched       | Model reasoning inside a bordered box          |
| `print_action_box()`      | Watched       | Parsed action + latency                        |
| `print_turn_error()`      | Watched       | Game error box                                 |
| `print_batch_progress()`  | Batch         | Live progress bar with win count               |
| `print_batch_summary()`   | Batch         | Final stats table (win rate, avg turns, etc.)  |

---

## `checkpoint.py`

**Purpose:** Append-only JSONL checkpoint writer. One file per episode at `game_checkpoints/{game_name}/{game_id}.jsonl`.

### Record types (written in order)
1. `game_header` — Written by `open()`. Contains game ID, name, mode, model config, system prompt, env config.
2. `step` — Written by `write_step()` after every turn. Contains observation, LLM call details, parsed action, validation, step result.
3. `game_footer` — Written by `close()`. Contains final success status and total step count.

### `CheckpointWriter` class
| Method                    | Description                                           |
|---------------------------|-------------------------------------------------------|
| `open()` (classmethod)    | Factory — creates the checkpoint file and writes the header |
| `write_step(record)`      | Append one step record (async-safe via lock)          |
| `close(success, steps)`   | Write the footer and close the file                   |

### `make_model_cfg()`
Convenience function that builds the `model_cfg` dict from CLI/config values.

---

## `registry.py`

**Purpose:** Central game registry. Maps CLI game keys to `(module_path, class_name)` tuples.

### `REGISTRY` dict
```python
REGISTRY = {
    "mastermind": ("games.mastermind", "MastermindGame"),
}
```

### Functions
- `get_env_class(key)` — Import and return the `GameEnv` class for a given key.
- `list_games()` — Return sorted list of all registered game keys.

To add a new game: add one line to `REGISTRY`.

---

## `debug.py`

**Purpose:** TCP-based debug console that opens a second terminal window showing the full LLM conversation — observations, streaming reasoning tokens, responses, retries, and game errors — while the main terminal displays only the game's visual output.

### Architecture
- Main process opens a TCP server on `localhost:54321`.
- A second `cmd` window is spawned running `--debug-client` mode.
- The spawned window connects back via TCP and prints everything it receives.

### `DebugConnection` class (server side — main process)
| Method/Property   | Description                                                  |
|-------------------|--------------------------------------------------------------|
| `start_server()`  | Bind, listen, block until the debug client connects          |
| `send(text)`      | Send UTF-8 text to the debug console (swallows broken pipes) |
| `close()`         | Clean up sockets                                             |
| `connected`       | `True` when a client is connected                            |

Supports context manager protocol (`with DebugConnection() as dc: ...`).

### `debug_client_mode()`
Entry point for the spawned debug terminal. Connects to the main process, prints everything received, retries connection for up to ~6 seconds.

### `spawn_debug_window(script_path)`
Opens a new `cmd` window running the given script with `--debug-client`.

### Formatting functions
All return ANSI-coloured strings ready to `send()` to the debug console:

| Function                | Color   | Shows                                        |
|-------------------------|---------|----------------------------------------------|
| `fmt_system_prompt()`   | Cyan    | Full system prompt at episode start           |
| `fmt_observation()`     | Green   | Game observation text sent to the LLM         |
| `fmt_reasoning_start()` | Yellow  | Header when reasoning tokens begin            |
| `fmt_reasoning_chunk()` | Yellow  | Individual streaming reasoning token          |
| `fmt_reasoning_end()`   | Yellow  | Footer when reasoning is complete             |
| `fmt_response_start()`  | Blue    | Header when content tokens begin              |
| `fmt_response_chunk()`  | White   | Individual streaming content token            |
| `fmt_response_end()`    | Blue    | Footer when response is complete              |
| `fmt_action_result()`   | Magenta | Parsed action, validity, latency              |
| `fmt_retry()`           | Red     | Retry attempt with error message              |
| `fmt_game_error()`      | Red     | Game-level action rejection                   |
| `fmt_episode_end()`     | Green/Red | Final WIN/LOSS status                       |
| `fmt_token_usage()`     | Dim     | Token usage stats from streaming response     |
