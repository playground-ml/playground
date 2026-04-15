"""
playground.agent
~~~~~~~~~~~~~~~~
Async LLM agent.  Wraps the openai AsyncClient to:
  1. Call the model with the full conversation history.
  2. Extract a JSON action from the raw response (even if the model
     wraps it in prose or markdown fences).
  3. Validate the extracted dict against the game's action_schema.
  4. Retry up to max_retries times, feeding parse errors back to the model.

Returns an AgentResult with everything the checkpoint writer needs.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from openai import AsyncOpenAI

try:
    import jsonschema as _js
    _HAVE_JSONSCHEMA = True
except ImportError:
    _HAVE_JSONSCHEMA = False


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """Everything produced by one LLM turn."""
    raw_response: str               # complete text returned by the API
    reasoning: str | None           # reasoning field if present (o-series models)
    action_raw: str | None          # JSON string extracted from raw_response
    parsed_action: dict | None      # validated dict, or None on failure
    action_valid: bool
    error_message: str | None
    latency_ms: int
    messages_sent: list[dict]       # full messages array sent to the API


# ── JSON extraction ───────────────────────────────────────────────────────────

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON   = re.compile(r"(\{[^{}]*\})", re.DOTALL)

def _extract_json(text: str) -> str | None:
    """Extract the first JSON object from *text*, ignoring surrounding prose."""
    # prefer fenced block
    m = _JSON_FENCE.search(text)
    if m:
        return m.group(1).strip()
    # fall back to bare braces
    m = _BARE_JSON.search(text)
    if m:
        return m.group(1).strip()
    return None


def _validate(action: dict, schema: dict) -> str | None:
    """Return an error string, or None if valid."""
    if _HAVE_JSONSCHEMA:
        try:
            _js.validate(instance=action, schema=schema)
            return None
        except _js.ValidationError as exc:
            return exc.message
    # fallback: required-key check
    missing = [k for k in schema.get("required", []) if k not in action]
    if missing:
        return f"Missing required keys: {missing}"
    return None



def _assistant_msg(content: str, reasoning_details: Any) -> dict:
    """Build an assistant message, preserving reasoning_details for multi-turn."""
    msg: dict = {"role": "assistant", "content": content}
    if reasoning_details is not None:
        msg["reasoning_details"] = reasoning_details
    return msg

# ── agent ─────────────────────────────────────────────────────────────────────

class LLMAgent:
    """Async agent that drives a GameEnv for one episode.

    Parameters
    ----------
    client        : AsyncOpenAI (or compatible async client)
    model         : model identifier string
    max_retries   : how many times to retry a bad response per turn
    reasoning_effort : "low" | "medium" | "high" | None
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        max_retries: int = 3,
        reasoning_effort: str | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._max_retries = max_retries
        self._reasoning_effort = reasoning_effort

    async def call(
        self,
        messages: list[dict],
        action_schema: dict,
    ) -> AgentResult:
        """Make one LLM call and return an AgentResult.

        Retries internally on parse/validation failure.  On retry the assistant
        message includes reasoning_details so the model continues its
        chain-of-thought unbroken (OpenRouter multi-turn spec).
        """
        local_msgs = list(messages)
        last_raw = ""
        last_reasoning: str | None = None
        last_rd: Any = None
        latency_ms = 0

        for attempt in range(1, self._max_retries + 1):
            t0 = time.perf_counter()
            raw, reasoning, rd = await self._api_call(local_msgs)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            last_raw, last_reasoning, last_rd = raw, reasoning, rd

            action_raw = _extract_json(raw)
            if action_raw is None:
                err = "No JSON object found in your response."
                if attempt < self._max_retries:
                    local_msgs += [
                        _assistant_msg(raw, rd),
                        {"role": "user", "content": f"Parse error: {err}  Please respond with only the JSON object."},
                    ]
                continue

            try:
                parsed = json.loads(action_raw)
            except json.JSONDecodeError as exc:
                err = f"Invalid JSON: {exc}"
                if attempt < self._max_retries:
                    local_msgs += [
                        _assistant_msg(raw, rd),
                        {"role": "user", "content": f"Parse error: {err}  Please respond with only the JSON object."},
                    ]
                continue

            err = _validate(parsed, action_schema)
            if err:
                if attempt < self._max_retries:
                    local_msgs += [
                        _assistant_msg(raw, rd),
                        {"role": "user", "content": f"Validation error: {err}  Please correct your JSON."},
                    ]
                continue

            return AgentResult(
                raw_response  = raw,
                reasoning     = reasoning,
                action_raw    = action_raw,
                parsed_action = parsed,
                action_valid  = True,
                error_message = None,
                latency_ms    = latency_ms,
                messages_sent = local_msgs,
            )

        return AgentResult(
            raw_response  = last_raw,
            reasoning     = last_reasoning,
            action_raw    = _extract_json(last_raw),
            parsed_action = None,
            action_valid  = False,
            error_message = "Max retries exceeded without a valid action.",
            latency_ms    = latency_ms,
            messages_sent = local_msgs,
        )

    # ── internal ──────────────────────────────────────────────────────────────

    async def _api_call(
        self, messages: list[dict]
    ) -> tuple[str, str | None, Any]:
        """One raw API call.  Returns (content, reasoning_text, reasoning_details).

        reasoning_text    : human-readable string for logging / display.
        reasoning_details : raw object passed back unmodified in multi-turn
                            messages so the model continues its chain-of-thought.
        """
        kwargs: dict[str, Any] = dict(model=self._model, messages=messages)

        if self._reasoning_effort:
            # kwargs["extra_body"] = {
            #     "reasoning": {"enabled": True, "effort": self._reasoning_effort}
            # }
            kwargs["extra_body"] = {
                "reasoning": {"enabled": True, "effort": "low"}
            }

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = choice.message.content or ""

        # OpenRouter returns reasoning_details as a list of block objects
        rd = getattr(choice.message, "reasoning_details", None)
        # reasoning_text: str | None = None
        # if rd:
        #     parts = []
        #     for block in rd:
        #         text = getattr(block, "summary", None) or getattr(block, "text", None)
        #         if text:
        #             parts.append(str(text))
        #     reasoning_text = "\n".join(parts) if parts else None
        reasoning_text = response.reasoning

        return content, reasoning_text, rd


def build_system_prompt(game_name: str, action_schema: dict) -> str:
    """Build the system prompt the engine sends once per episode."""
    schema_str = json.dumps(action_schema, indent=2, ensure_ascii=False)
    return (
        f"You are playing the text-based game: {game_name}.\n\n"
        "Each turn you will receive the current game state as a message.\n"
        "You must respond with a JSON object that conforms EXACTLY to this schema:\n\n"
        f"{schema_str}\n\n"
        "Think through your reasoning, then output ONLY the JSON object — "
        "no extra prose, no markdown fences."
    )
