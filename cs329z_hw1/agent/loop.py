"""The agent loop: alternate model replies and tool calls."""

from __future__ import annotations

import copy

from cs329z_hw1.agent.protocol import parse_response, render_tools
from cs329z_hw1.agent.registry import Registry
from cs329z_hw1.agent.tools import build_tools
from cs329z_hw1.llm import OutputTruncated
from cs329z_hw1.tokens import count_message_tokens, count_tokens
from cs329z_hw1.types import AgentConfig, AgentResult, ToolSpec

_MALFORMED_STATUSES = ("unknown_tool", "invalid_args")
_MALFORMED_LIMIT = 3


class Session:
    """One conversation. Each send continues the same transcript and context."""

    def __init__(self, lm, tools: list[ToolSpec] | None, config: AgentConfig) -> None:
        self.lm = lm
        self.aux_lm = lm if config.aux_lm is None else config.aux_lm
        self.config = config
        self.mode = config.mode
        self.registry = Registry(build_tools(tools, config))
        self.transcript: list[dict] = []
        self.messages: list[dict] = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that answers the user and calls tools when needed. "
                    "A reply with no tool call is the final answer.\n\n"
                    + render_tools(self.registry.schemas())
                ),
            }
        ]

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def send(self, user_message: str) -> AgentResult:
        self._append({"type": "user", "content": user_message})
        self.messages.append({"role": "user", "content": user_message})
        turns = 0
        tokens = 0
        streak = 0
        while True:
            # Budgets are checked before the call, so a final answer that
            # crosses the token budget on its own call is still "done".
            if turns >= self.config.max_turns:
                return self._finish("max_turns", "Stopped because this message reached the turn limit.")
            if tokens >= self.config.token_budget:
                return self._finish("token_budget", "Stopped because this message reached the token budget.")
            try:
                reply = self.lm(self.messages)
            except OutputTruncated as exc:
                partial = exc.text or ""
                turns += 1
                tokens += count_message_tokens(self.messages) + count_tokens(partial)
                feedback = "The model reply was cut off before it finished. Send one complete reply."
                self._append({"type": "assistant", "content": partial})
                self._append({"type": "error", "content": feedback})
                self.messages.append({"role": "assistant", "content": partial})
                self.messages.append({"role": "user", "content": feedback})
                streak += 1
                if streak >= _MALFORMED_LIMIT:
                    return self._finish("malformed", "Stopped because three replies in a row were malformed.")
                continue
            turns += 1
            tokens += count_message_tokens(self.messages) + count_tokens(reply)
            self._append({"type": "assistant", "content": reply})
            self.messages.append({"role": "assistant", "content": reply})
            parsed = parse_response(reply)
            if parsed.error:
                feedback = parsed.error.strip() or "The reply could not be parsed."
                self._append({"type": "error", "content": feedback})
                self.messages.append({"role": "user", "content": feedback})
                streak += 1
                if streak >= _MALFORMED_LIMIT:
                    return self._finish("malformed", "Stopped because three replies in a row were malformed.")
                continue
            if parsed.tool_call is None:
                text = parsed.text.strip() or reply.strip() or "Done."
                return self._finish("done", text)
            call = parsed.tool_call
            result = self.registry.execute(call.name, call.args)
            self._append({"type": "tool_call", "name": call.name, "args": copy.deepcopy(call.args)})
            self._append(
                {
                    "type": "tool_result",
                    "name": call.name,
                    "status": result.status,
                    "content": result.content,
                }
            )
            # The next model call must contain this string verbatim.
            self.messages.append({"role": "user", "content": result.content})
            if result.status in _MALFORMED_STATUSES:
                streak += 1
                if streak >= _MALFORMED_LIMIT:
                    return self._finish("malformed", "Stopped because three replies in a row were malformed.")
            else:
                streak = 0

    def _append(self, event: dict) -> None:
        self.transcript.append(event)

    def _finish(self, status: str, text: str) -> AgentResult:
        if not text.strip():
            text = "Done."
        return AgentResult(text=text, status=status, transcript=copy.deepcopy(self.transcript))
