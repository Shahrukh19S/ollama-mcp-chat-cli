"""
agent/tool_manager.py — bridge between the model's tool calls and the MCP servers.

WHAT THIS FILE DOES
-------------------
Two jobs:
  1. get_all_tools: collect every MCP server's tools and convert them into the
     OpenAI "function" tool-definition shape that litellm expects.
  2. execute_tool_requests: when the model asks to call tools, run each one through
     the right MCP client and package the outputs as OpenAI "tool" result messages.

FORMAT NOTE (Anthropic -> OpenAI/litellm) — this is the part that changed most:
  * Tool DEFINITION:  Anthropic used {"name","description","input_schema"}.
                      OpenAI uses {"type":"function","function":{name,description,parameters}}.
  * Tool REQUEST:     Anthropic put tool_use blocks in message.content.
                      OpenAI puts them in message.tool_calls[], each with .id,
                      .function.name, and .function.arguments — a JSON STRING.
  * Tool RESULT:      Anthropic used a user message with a tool_result block.
                      OpenAI uses a separate message: {"role":"tool","tool_call_id",content}.
"""

import json
from typing import Optional
from mcp.types import Tool, TextContent
from mcp_client import MCPClient


class ToolManager:
    @classmethod
    async def get_all_tools(cls, clients: dict[str, MCPClient]) -> list[dict]:
        """
        Gather tools from every connected MCP client and return them as OpenAI
        function-tool definitions (the shape litellm/the model understands).

        Each MCP Tool gives us name, description, and inputSchema (already a JSON
        schema), which map straight onto the OpenAI 'parameters' field.
        """
        tools = []
        for client in clients.values():
            tool_models = await client.list_tools()
            tools += [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        # inputSchema is already a JSON schema describing the args.
                        "parameters": t.inputSchema,
                    },
                }
                for t in tool_models
            ]
        return tools

    @classmethod
    async def _find_client_with_tool(
        cls, clients: list[MCPClient], tool_name: str
    ) -> Optional[MCPClient]:
        """
        Find the first client whose server exposes `tool_name`.

        With multiple MCP servers, a tool name alone doesn't say which server owns
        it — so we ask each client for its tools and match by name.
        """
        for client in clients:
            tools = await client.list_tools()
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                return client
        return None

    @classmethod
    def _build_tool_result_message(cls, tool_call_id: str, text: str) -> dict:
        """
        Build one OpenAI 'tool' result message.

        `tool_call_id` MUST match the id from the assistant's tool_calls entry — that
        link is how the model knows which call this result answers. Unlike Anthropic,
        there is no separate is_error flag; if the tool failed we simply put the
        error text in `content` so the model can read and react to it.
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": text,
        }

    @classmethod
    async def execute_tool_requests(
        cls, clients: dict[str, MCPClient], message
    ) -> list[dict]:
        """
        Run every tool the model requested in `message.tool_calls` and return a list
        of 'tool' result messages (one per call), in the same order.

        `message` is the assistant message object from litellm. Its `.tool_calls`
        list holds the requested calls; each call's `.function.arguments` is a JSON
        STRING we must json.loads before handing the dict to the MCP client.
        """
        # The model may emit zero tool calls; default to an empty list defensively.
        tool_requests = message.tool_calls or []
        tool_result_messages: list[dict] = []

        for tool_request in tool_requests:
            tool_call_id = tool_request.id
            tool_name = tool_request.function.name

            # arguments arrives as a JSON string, e.g. '{"doc_id": "welcome.md"}'.
            # Parse it; if the model emitted malformed JSON, fall back to empty args
            # rather than crashing the whole loop.
            try:
                tool_input = json.loads(tool_request.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_input = {}

            # Locate the server that owns this tool.
            client = await cls._find_client_with_tool(
                list(clients.values()), tool_name
            )
            if not client:
                tool_result_messages.append(
                    cls._build_tool_result_message(
                        tool_call_id, f"Could not find tool '{tool_name}'."
                    )
                )
                continue

            try:
                # Call the tool over MCP. The result's `.content` is a list of parts;
                # we keep the text parts and JSON-encode them as the result content.
                tool_output = await client.call_tool(tool_name, tool_input)
                items = tool_output.content if tool_output else []
                content_list = [
                    item.text for item in items if isinstance(item, TextContent)
                ]
                # If the server flagged an error, surface it; else send the text(s).
                if tool_output and tool_output.isError:
                    result_text = json.dumps({"error": content_list})
                else:
                    result_text = json.dumps(content_list)
                tool_result_messages.append(
                    cls._build_tool_result_message(tool_call_id, result_text)
                )
            except Exception as e:
                # Never let one bad tool call kill the loop — report it as a result.
                error_message = f"Error executing tool '{tool_name}': {e}"
                print(error_message)
                tool_result_messages.append(
                    cls._build_tool_result_message(
                        tool_call_id, json.dumps({"error": error_message})
                    )
                )

        return tool_result_messages
