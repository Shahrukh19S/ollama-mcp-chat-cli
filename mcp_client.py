import sys
import json
import asyncio
from typing import Optional, Any
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(
        self,
        command: str,
        args: list[str],
        env: Optional[dict] = None,
    ):
        self._command = command
        self._args = args
        self._env = env
        self._session: Optional[ClientSession] = None
        self._exit_stack: AsyncExitStack = AsyncExitStack()

    async def connect(self):
        server_params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env,
        )
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        _stdio, _write = stdio_transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(_stdio, _write)
        )
        await self._session.initialize()

    def session(self) -> ClientSession:
        if self._session is None:
            raise ConnectionError(
                "Client session not initialized or cache not populated. Call connect_to_server first."
            )
        return self._session

    async def list_tools(self) -> list[types.Tool]:
        """
        Ask the server for every tool it exposes.

        `session().list_tools()` returns a ListToolsResult whose `.tools` field is
        the actual list of Tool objects (each has name, description, inputSchema).
        The agent loop turns these into the tool definitions it offers the model.
        """
        result = await self.session().list_tools()
        return result.tools

    async def call_tool(
        self, tool_name: str, tool_input: dict
    ) -> types.CallToolResult | None:
        """
        Run one tool on the server and return its result.

        `tool_input` is a plain dict of arguments. IMPORTANT: the model hands us
        tool arguments as a JSON *string*; the caller (agent/tool_manager.py) json.loads it
        into this dict before calling here. The returned CallToolResult carries the
        output content plus an `isError` flag.
        """
        return await self.session().call_tool(tool_name, tool_input)

    async def list_prompts(self) -> list[types.Prompt]:
        """
        Ask the server for every prompt (the /commands like /format, /summarize).

        Returns the `.prompts` list. The CLI uses these to power /-command
        autocompletion and to know each prompt's argument names.
        """
        result = await self.session().list_prompts()
        return result.prompts

    async def get_prompt(self, prompt_name, args: dict[str, str]):
        """
        Fetch a specific prompt with its arguments filled in.

        e.g. get_prompt("summarize", {"doc_id": "customer-feedback.txt"}). The server returns a
        GetPromptResult; we return its `.messages` (the chat messages to prepend to
        the conversation so the model carries out that command).
        """
        result = await self.session().get_prompt(prompt_name, args)
        return result.messages

    async def read_resource(self, uri: str) -> Any:
        """
        Read a resource by URI and return its parsed contents.

        A resource read comes back as a list of content parts; we use the first.
        We branch on the declared mime type: JSON resources (like
        docs://documents, our id list) are parsed into a Python object, while
        everything else (a doc's text) is returned as a plain string. This is why
        list_docs_ids() upstream gets a real list while get_doc_content() gets text.
        """
        result = await self.session().read_resource(uri)
        resource = result.contents[0]

        # TextResourceContents carries `.text` and `.mimeType`.
        if isinstance(resource, types.TextResourceContents):
            if resource.mimeType == "application/json":
                return json.loads(resource.text)
            return resource.text

        # Fallback for any non-text (blob) resource — return the raw part.
        return resource

    async def cleanup(self):
        await self._exit_stack.aclose()
        self._session = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()


# For testing
async def main():
    async with MCPClient(
        # If using Python without UV, update command to 'python' and remove "run" from args.
        command="uv",
        args=["run", "mcp_server.py"],
    ) as _client:
        pass


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
