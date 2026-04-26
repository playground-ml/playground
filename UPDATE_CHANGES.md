# Changelog

All notable changes to the Playground engine are documented here.

---

## [2026-04-26] — Debug Console & Streaming Support

### Added
- **`playground/debug.py`** — New module: TCP-based debug console that spawns a second terminal showing the full LLM conversation (system prompt, observations, streaming reasoning, responses, retries, game errors) in real-time.
- **`--debug` CLI flag** in `run_game.py` — Opens the debug console when used with `--batch 1`.
- **`--debug-client` internal flag** — Used by the spawned debug terminal to run in client mode.
- **Streaming API support** in `agent.py` — New `_api_call_streaming()` method sends reasoning and content tokens to the debug console as they arrive.
- **`playground/docs.md`** — Module-level documentation for every file in the `playground/` package.

### Modified
- **`playground/agent.py`** — `LLMAgent.call()` accepts optional `debug_conn` parameter; uses streaming when provided, sends retry errors to debug console.
- **`playground/runner.py`** — `run()`, `_run_llm_watched()`, and `_episode_loop()` accept `debug_conn`; observations, action results, game errors, and episode end are sent to the debug console at each step.
- **`run_game.py`** — Added `--debug`/`--debug-client` flags, spawns debug window, validates `--debug` only works with `--batch 1`, passes `debug_conn` through to `pg.run()`.
- **`README.md`** — Expanded "Run Game" section with full CLI flag table; added "Debug Mode" section documenting the feature and its output.
