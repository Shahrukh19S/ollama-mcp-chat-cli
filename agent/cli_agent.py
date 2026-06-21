from typing import List, Tuple
from mcp.types import Prompt, PromptMessage

from agent.agent_loop import Chat
from agent.llm_service import LLMService
from mcp_client import MCPClient


class CliChat(Chat):
    def __init__(
        self,
        doc_client: MCPClient,
        clients: dict[str, MCPClient],
        llm_service: LLMService,
    ):
        super().__init__(clients=clients, llm_service=llm_service)

        self.doc_client: MCPClient = doc_client

    async def list_prompts(self) -> list[Prompt]:
        return await self.doc_client.list_prompts()

    async def list_docs_ids(self) -> list[str]:
        return await self.doc_client.read_resource("docs://documents")

    async def get_doc_content(self, doc_id: str) -> str:
        return await self.doc_client.read_resource(f"docs://documents/{doc_id}")

    async def get_prompt(
        self, command: str, doc_id: str
    ) -> list[PromptMessage]:
        return await self.doc_client.get_prompt(command, {"doc_id": doc_id})

    async def _extract_resources(self, query: str) -> str:
        mentions = [word[1:] for word in query.split() if word.startswith("@")]

        doc_ids = await self.list_docs_ids()
        mentioned_docs: list[Tuple[str, str]] = []

        for doc_id in doc_ids:
            if doc_id in mentions:
                content = await self.get_doc_content(doc_id)
                mentioned_docs.append((doc_id, content))

        return "".join(
            f'\n<document id="{doc_id}">\n{content}\n</document>\n'
            for doc_id, content in mentioned_docs
        )

    async def _process_command(self, query: str) -> bool:
        if not query.startswith("/"):
            return False

        words = query.split()
        command = words[0].replace("/", "")

        # A prompt command needs a document id, e.g. "/summarize customer-feedback.txt".
        # The autocompletion makes it easy to submit just "/summarize": pressing
        # Enter on the menu both SELECTS the command and SUBMITS the line. Without
        # this guard, words[1] below raised IndexError and crashed the whole CLI.
        # Instead of crashing, add a normal user turn (so the loop stays valid and
        # never calls the model with empty messages) telling the user the usage.
        if len(words) < 2:
            doc_ids = await self.list_docs_ids()
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        f"I typed the command '/{command}' but did not include a "
                        f"document id. Tell me the correct usage, like "
                        f"'/{command} <doc_id>'. The available document ids are: "
                        f"{', '.join(doc_ids)}."
                    ),
                }
            )
            return True

        messages = await self.doc_client.get_prompt(
            command, {"doc_id": words[1]}
        )

        self.messages += convert_prompt_messages_to_message_params(messages)
        return True

    async def _process_query(self, query: str):
        if await self._process_command(query):
            return

        added_resources = await self._extract_resources(query)

        prompt = f"""
        The user has a question:
        <query>
        {query}
        </query>

        The following context may be useful in answering their question:
        <context>
        {added_resources}
        </context>

        Note the user's query might contain references to documents like "@welcome.md". The "@" is only
        included as a way of mentioning the doc. The actual name of the document would be "welcome.md".
        If the document content is included in this prompt, you don't need to use an additional tool to read the document.
        Answer the user's question directly and concisely. Start with the exact information they need. 
        Don't refer to or mention the provided context in any way - just use it to inform your answer.
        """

        self.messages.append({"role": "user", "content": prompt})


def convert_prompt_message_to_message_param(
    prompt_message: "PromptMessage",
) -> dict:
    """
    Convert one MCP PromptMessage into a plain OpenAI-shaped message dict
    {"role": ..., "content": ...}.

    This is the ONLY thing that changed for the provider swap: OpenAI/litellm want
    flat string content, not Anthropic's typed content blocks. An MCP prompt message
    carries a single TextContent (or, rarely, a list of them); we pull the text out
    and join it into a string.
    """
    # MCP prompt roles are "user"/"assistant"; pass them straight through.
    role = "user" if prompt_message.role == "user" else "assistant"

    content = prompt_message.content

    # Common case: a single TextContent object with .type == "text" and .text.
    if getattr(content, "type", None) == "text":
        return {"role": role, "content": content.text}

    # Rare case: a list of content blocks — keep the text ones and join them.
    if isinstance(content, list):
        text = "".join(
            getattr(block, "text", "")
            for block in content
            if getattr(block, "type", None) == "text"
        )
        return {"role": role, "content": text}

    # Anything unexpected -> empty content (keeps the conversation well-formed).
    return {"role": role, "content": ""}


def convert_prompt_messages_to_message_params(
    prompt_messages: List[PromptMessage],
) -> List[dict]:
    return [
        convert_prompt_message_to_message_param(msg) for msg in prompt_messages
    ]
