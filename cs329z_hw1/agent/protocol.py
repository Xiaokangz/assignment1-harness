"""Text protocol for one tool call: a fenced JSON object.

A call looks like this:

    Looking.
    ```tool
    {"name": "add", "arguments": {"a": 17, "b": 25}}
    ```

The fence markers sit on their own lines. Arguments are JSON, so 1, 1.0,
true, and "1" stay distinct. A fence that appears inside a string argument
is just characters: the JSON parser consumes it with the string.
"""

from __future__ import annotations

import json

from cs329z_hw1.types import ParsedResponse, ToolCall

_OPEN = "```tool"
_CLOSE = "```"
_DECODER = json.JSONDecoder()


def render_tools(schemas: list[dict]) -> str:
    """Prompt text: every tool, its parameters, and the call format."""
    parts = [
        "To call a tool, add exactly one fenced block to your reply. "
        "The block contains one JSON object and nothing else:",
        "",
        "```tool",
        '{"name": "echo", "arguments": {"text": "hello"}}',
        "```",
        "",
        "Rules:",
        "- At most one tool block. A second block is an error.",
        "- Put the opening ```tool and the closing ``` on their own lines.",
        "- arguments is a JSON object. Write numbers, true, false, and null as JSON values, not as strings.",
        "- Use the parameter names listed for that tool. Include every required parameter.",
        "- A tool with no parameters uses \"arguments\": {}.",
        "- When you are not calling a tool, do not write a tool block.",
        "",
        "Tools:",
    ]
    for schema in schemas:
        parts.append(_describe_tool(schema))
    return "\n".join(parts)


def format_tool_call(text: str, call: ToolCall) -> str:
    """Write text and one call. Does not modify call."""
    payload = json.dumps(
        {"name": call.name, "arguments": call.args},
        ensure_ascii=False,
    )
    block = f"{_OPEN}\n{payload}\n{_CLOSE}"
    prose = text.strip()
    if not prose:
        return block
    return f"{prose}\n{block}"


def parse_response(text: str) -> ParsedResponse:
    """Parse one reply. Never raises."""
    try:
        return _parse(text if isinstance(text, str) else str(text))
    except Exception as exc:
        return ParsedResponse(text=str(text).strip(), error=f"Could not parse the reply: {exc}")


def _parse(reply: str) -> ParsedResponse:
    blocks: list[tuple[int, int, ToolCall]] = []
    pos = 0
    while True:
        opened = _find_open(reply, pos)
        if opened is None:
            break
        parsed = _read_block(reply, opened)
        if parsed is None:
            # A fence that does not finish, including one cut inside its
            # arguments, is a malformed call. Do not return a partial call.
            return ParsedResponse(
                text=_without(reply, blocks).strip(),
                error="The tool call is incomplete or is not valid JSON.",
            )
        end, call = parsed
        blocks.append((opened, end, call))
        pos = end
    if len(blocks) > 1:
        return ParsedResponse(text=reply.strip(), error="A reply can contain only one tool call.")
    if not blocks:
        return ParsedResponse(text=reply.strip())
    opened, end, call = blocks[0]
    prose = (reply[:opened] + reply[end:]).strip()
    return ParsedResponse(text=prose, tool_call=call)


def _read_block(reply: str, opened: int) -> tuple[int, ToolCall] | None:
    """A finished block as (end index, call), or None when it does not close."""
    cursor = opened + len(_OPEN)
    while cursor < len(reply) and reply[cursor] in " \t\r\n":
        cursor += 1
    try:
        obj, json_end = _DECODER.raw_decode(reply, cursor)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    arguments = obj.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    cursor = json_end
    while cursor < len(reply) and reply[cursor] in " \t\r\n":
        cursor += 1
    if not reply.startswith(_CLOSE, cursor):
        return None
    # The closer is a line of its own. ```tool starts with ```, so require
    # that the next character ends the line.
    after = cursor + len(_CLOSE)
    if after < len(reply) and reply[after] != "\n":
        return None
    if after < len(reply):
        after += 1
    return after, ToolCall(name=name, args=arguments)


def _find_open(reply: str, start: int) -> int | None:
    """Index of the next ```tool marker that begins a line."""
    token = _OPEN
    idx = start
    while True:
        found = reply.find(token, idx)
        if found < 0:
            return None
        at_line_start = found == 0 or reply[found - 1] == "\n"
        after = found + len(token)
        boundary = after == len(reply) or reply[after] in " \t\r\n"
        if at_line_start and boundary:
            return found
        idx = found + 1


def _without(reply: str, blocks: list[tuple[int, int, ToolCall]]) -> str:
    if not blocks:
        return reply
    pieces = []
    pos = 0
    for opened, end, _call in blocks:
        pieces.append(reply[pos:opened])
        pos = end
    pieces.append(reply[pos:])
    return "".join(pieces)


def _describe_tool(schema: dict) -> str:
    name = str(schema.get("name", ""))
    description = str(schema.get("description", ""))
    parameters = schema.get("parameters") or {}
    properties = parameters.get("properties") or {}
    required = set(parameters.get("required") or [])
    lines = [f"Tool {name}: {description}"]
    if not properties:
        lines.append("Parameters: none. Call it with \"arguments\": {}.")
        return "\n".join(lines)
    lines.append("Parameters:")
    for param_name, param_schema in properties.items():
        need = "required" if param_name in required else "optional"
        lines.append(f"- {param_name} ({need}): {json.dumps(param_schema, ensure_ascii=False)}")
    return "\n".join(lines)
