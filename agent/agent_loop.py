"""
agent/agent_loop.py — the agentic tool-calling loop.

WHAT THIS FILE DOES
-------------------
`Chat.run(query)` drives one user question to a final answer. It may take several
model round-trips: the model can ask to call a tool, we run it, feed the result
back, and ask again — looping until the model stops requesting tools and gives a
final text answer. This is the heart of an "agent".

THE LOOP, IN WORDS
------------------
  1. Add the user's query to the running message list.
  2. Send the whole conversation + the available tools to the model.
  3. Append the assistant's reply to the conversation.
  4. If finish_reason == "tool_calls": run the requested tools, append their
     results, and loop back to step 2 so the model can use them.
  5. Otherwise: that reply is the final answer — return its text.

ORDER MATTERS: we append the assistant message that HOLDS the tool_calls BEFORE the
tool-result messages. The API pairs each result to a tool_call by id, and it only
works if the assistant tool_calls message comes first.
"""

from mcp_client import MCPClient
from agent.llm_service import LLMService
from agent.tool_manager import ToolManager


class Chat:
    def __init__(self, llm_service: LLMService, clients: dict[str, MCPClient]):
        """
        Args:
            llm_service: the provider-agnostic LLM wrapper (Ollama/Groq/GitHub).
            clients:     the connected MCP clients, keyed by id. Their tools are
                         offered to the model each round.
        `self.messages` is the growing OpenAI-shaped conversation (user/assistant/
        tool messages) that we re-send in full every round so the model has context.
        """
        self.llm_service: LLMService = llm_service
        self.clients: dict[str, MCPClient] = clients
        self.messages: list = []

    async def _process_query(self, query: str):
        """Add the raw user query as a user message. (CliChat overrides this to also
        handle @-mentions and /commands before the query reaches the model.)"""
        self.messages.append({"role": "user", "content": query})

    async def run(self, query: str) -> str:
        """Run the full loop for one query and return the final answer text."""
        final_text_response = ""

        await self._process_query(query)

        while True:
            # Send the conversation so far plus every tool the servers expose.
            response = self.llm_service.chat(
                messages=self.messages,
                tools=await ToolManager.get_all_tools(self.clients),
            )
            choice = response.choices[0]
            message = choice.message

            # Append the assistant message FIRST. If it carries tool_calls, those
            # must precede their results in the list (see module docstring).
            self.llm_service.add_assistant_message(self.messages, message)

            if choice.finish_reason == "tool_calls":
                # The model sometimes emits explanatory text alongside its tool
                # call — show it so the user sees the reasoning as it happens.
                thinking_text = self.llm_service.text_from_message(response)
                if thinking_text:
                    print(thinking_text)

                # Execute the requested tools and append one result message each,
                # AFTER the assistant message we just added.
                tool_result_messages = await ToolManager.execute_tool_requests(
                    self.clients, message
                )
                self.messages.extend(tool_result_messages)
                # Loop again so the model can answer using the tool results.
            else:
                # No tool requested -> this is the final answer.
                final_text_response = self.llm_service.text_from_message(response)
                break

        return final_text_response
