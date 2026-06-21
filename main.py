import asyncio
import sys
import os
from dotenv import load_dotenv
from contextlib import AsyncExitStack

from mcp_client import MCPClient
from agent.llm_service import LLMService

from agent.cli_agent import CliChat
from agent.terminal_ui import CliApp

load_dotenv()

# Provider config (replaces the old Anthropic-only setup).
# LLM_PROVIDER selects which credentials/endpoint LLMService loads; LLM_MODEL is
# the litellm model string whose prefix routes the call (ollama_chat/ | groq/ | openai/).
# Switch providers by editing these two lines in .env — no code change needed.
llm_provider = os.getenv("LLM_PROVIDER", "")
llm_model = os.getenv("LLM_MODEL", "")

assert llm_provider, "Error: LLM_PROVIDER cannot be empty. Update .env"
assert llm_model, "Error: LLM_MODEL cannot be empty. Update .env"


async def main():
    # One provider-agnostic service for the whole app. For Ollama this needs no key;
    # for the Groq/GitHub fallbacks LLMService reads the key from .env.
    llm_service = LLMService(model=llm_model, provider=llm_provider)

    server_scripts = sys.argv[1:]
    clients = {}

    command, args = (
        ("uv", ["run", "mcp_server.py"])
        if os.getenv("USE_UV", "0") == "1"
        else ("python", ["mcp_server.py"])
    )

    async with AsyncExitStack() as stack:
        doc_client = await stack.enter_async_context(
            MCPClient(command=command, args=args)
        )
        clients["doc_client"] = doc_client

        for i, server_script in enumerate(server_scripts):
            client_id = f"client_{i}_{server_script}"
            client = await stack.enter_async_context(
                MCPClient(command="uv", args=["run", server_script])
            )
            clients[client_id] = client

        chat = CliChat(
            doc_client=doc_client,
            clients=clients,
            llm_service=llm_service,
        )

        cli = CliApp(chat)
        await cli.initialize()
        await cli.run()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Force UTF-8 on our console output. Models — especially the cloud ones
    # (Groq/GitHub) — routinely emit characters outside Windows' default cp1252
    # codepage (em dashes, the narrow no-break space U+202F, etc.). Without this,
    # printing such a reply raises UnicodeEncodeError and crashes the CLI.
    # errors="replace" is a safety net so an odd glyph never takes the app down.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        # reconfigure() exists on Python 3.7+ text streams; ignore if unavailable.
        pass

    asyncio.run(main())
