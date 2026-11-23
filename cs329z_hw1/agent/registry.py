"""Tool registry: validate a call, then run it."""

from __future__ import annotations

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from cs329z_hw1.types import ToolResult, ToolSpec


class Registry:
    """Tools in registration order. Validation never calls a tool."""

    def __init__(self, tools: list[ToolSpec]) -> None:
        self._tools = list(tools)
        self._by_name = {tool.name: tool for tool in self._tools}

    def schemas(self) -> list[dict]:
        """One dict per tool, in registration order."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools
        ]

    def validate(self, name: str, args: dict) -> ToolResult | None:
        """The rejection execute would return, or None when the call can run."""
        tool = self._by_name.get(name)
        if tool is None:
            return ToolResult("unknown_tool", f"Unknown tool: {name!r}.")
        error = _schema_error(tool.parameters, args)
        if error is not None:
            return ToolResult("invalid_args", error)
        return None

    def execute(self, name: str, args: dict) -> ToolResult:
        """Validate, then call fn(**args). Always returns a ToolResult."""
        rejected = self.validate(name, args)
        if rejected is not None:
            return rejected
        tool = self._by_name[name]
        try:
            value = tool.fn(**args)
        except Exception as exc:
            return ToolResult("tool_error", str(exc))
        return ToolResult("ok", str(value))


def _schema_error(schema: dict, args: dict) -> str | None:
    """None when args matches the JSON Schema. Otherwise a non-empty message."""
    try:
        Draft202012Validator(schema).validate(args)
    except ValidationError as exc:
        return exc.message or "Arguments do not match the tool schema."
    except Exception as exc:
        return str(exc) or "Arguments do not match the tool schema."
    return None
