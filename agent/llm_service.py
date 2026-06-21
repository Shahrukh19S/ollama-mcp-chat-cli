"""
agent/llm_service.py — the provider-agnostic LLM layer.

WHAT THIS FILE IS
-----------------
This is the single place the rest of the app talks to a language model. Rather than
binding to a single vendor's API, the `LLMService` class speaks to THREE providers
behind one method:

    * Ollama        — a local model (qwen2.5:7b) running on this machine, $0, offline.
    * Groq          — a free cloud fallback with strong/fast tool calling.
    * GitHub Models  — a second free cloud fallback (small input cap; last resort).

HOW IT FITS THE WHOLE
---------------------
`main.py` reads two env vars (LLM_PROVIDER + LLM_MODEL), builds ONE `LLMService`,
and hands it to the chat loop. Every model round-trip in the agent loop
(`agent/agent_loop.py`) goes through `LLMService.chat(...)`. Because all three providers
return the SAME OpenAI-shaped response, the loop has a single code path.

WHY litellm
-----------
`litellm` is a thin routing layer. We give it a model string whose prefix tells it
which provider to call:
    * "ollama_chat/qwen2.5:7b"   -> Ollama's NATIVE API (where tool calling works).
                                    NOTE: the "/v1" OpenAI-compatible Ollama endpoint
                                    BREAKS tool calling, so we must use ollama_chat/.
    * "groq/openai/gpt-oss-120b" -> Groq.
    * "openai/gpt-4o-mini"       -> GitHub Models (via its OpenAI-compatible endpoint).
Whatever the provider, litellm hands back a response shaped like OpenAI's, so we
always read text from `response.choices[0].message.content`.
"""

import os

import litellm

# --- litellm global behaviour -------------------------------------------------
# drop_params=True: if a provider doesn't support a parameter we pass (e.g. a model
# that ignores `temperature`), litellm silently drops it instead of erroring. This
# keeps one call site working across three providers with different capabilities.
litellm.drop_params = True
# Keep the console clean — we don't want litellm's debug banners in a teaching CLI.
litellm.suppress_debug_info = True


class LLMService:
    """
    A single, provider-agnostic wrapper around `litellm.completion`.

    The constructor figures out WHICH provider we're using and loads the right
    credentials/endpoint for it. After that, `chat(...)` is identical no matter
    the provider — that's the whole point.
    """

    def __init__(self, model: str, provider: str):
        """
        Args:
            model:    The litellm model string, e.g. "ollama_chat/qwen2.5:7b".
                      This string's prefix is what tells litellm which API to hit.
            provider: One of "ollama" | "groq" | "github". This selects which
                      credentials/endpoint we load from the environment.

        We store an `api_base` and `api_key` chosen by provider; `chat()` passes
        whichever are set to litellm. Ollama needs only a base URL (no key); the
        cloud providers need a key (and GitHub Models also needs a base URL).
        """
        self.model = model
        self.provider = provider

        # Defaults: nothing. We fill these in per provider below.
        self.api_base: str | None = None
        self.api_key: str | None = None

        if provider == "ollama":
            # Local server. No API key exists or is needed; just point at the host.
            self.api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

        elif provider == "groq":
            # Cloud fallback 1. Needs a key; litellm already knows Groq's endpoint,
            # so we do NOT set api_base here.
            self.api_key = os.getenv("GROQ_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "LLM_PROVIDER=groq but GROQ_API_KEY is empty. "
                    "Add your Groq key to .env (it stays local; .env is git-ignored)."
                )

        elif provider == "github":
            # Cloud fallback 2. Needs both a key (a GitHub PAT with models:read)
            # and the GitHub Models endpoint, since it's an OpenAI-compatible route.
            self.api_key = os.getenv("GITHUB_MODELS_API_KEY")
            self.api_base = os.getenv("GITHUB_MODELS_API_BASE")
            if not self.api_key or not self.api_base:
                raise ValueError(
                    "LLM_PROVIDER=github needs GITHUB_MODELS_API_KEY and "
                    "GITHUB_MODELS_API_BASE in .env."
                )

        else:
            # Fail loudly on a typo rather than silently behaving like one provider.
            raise ValueError(
                f"Unknown LLM_PROVIDER '{provider}'. "
                "Use one of: ollama, groq, github."
            )

    def add_user_message(self, messages: list, message):
        """
        Append a user turn to the running `messages` list.

        `message` is usually a plain string (the user's text). We also accept an
        already-formed message object/dict and read its `.content`, so callers can
        pass either. OpenAI-shaped messages are just {"role", "content"} dicts.
        """
        messages.append(
            {
                "role": "user",
                # If someone handed us a message-like object, pull its content;
                # otherwise treat it as the literal text.
                "content": getattr(message, "content", message),
            }
        )

    def add_assistant_message(self, messages: list, message):
        """
        Append an assistant turn to the running `messages` list.

        Two shapes show up here:
          * a plain string (a final text answer), or
          * the assistant message object returned by litellm — which may carry
            `tool_calls`. We must preserve that object as-is so the follow-up
            tool-result messages line up with the tool_call ids the model emitted.
        The agent loop (agent/agent_loop.py) relies on this preservation.
        """
        if isinstance(message, str):
            messages.append({"role": "assistant", "content": message})
        else:
            # A litellm message object — append it directly so tool_calls survive.
            messages.append(message)

    def text_from_message(self, response) -> str:
        """
        Pull the assistant's text out of a litellm completion response.

        OpenAI/litellm shape: response.choices[0].message.content. When the model
        asks for a tool instead of answering, `content` can be None, so we coalesce
        to an empty string to keep callers simple.
        """
        return response.choices[0].message.content or ""

    def chat(
        self,
        messages,
        tools=None,
        system=None,
        temperature=1.0,
        max_tokens=4096,
    ):
        """
        Send one request to the model and return litellm's raw response.

        Args:
            messages:    The OpenAI-shaped conversation so far (list of role/content
                         dicts, plus any assistant tool_call / tool result messages).
            tools:       Optional list of tool definitions in OpenAI "function" shape
                         (built in agent/tool_manager.py). Omitted entirely when None so a
                         plain chat call carries no tools key.
            system:      Optional system prompt. UNLIKE Anthropic (where `system` is a
                         top-level param), OpenAI/litellm expect the system prompt as
                         the FIRST message with role "system" — so we prepend it here.
            temperature: Sampling temperature.
            max_tokens:  Cap on output tokens.

        Returns:
            The litellm completion response (OpenAI-shaped). Read text via
            `text_from_message`, and inspect `.choices[0].finish_reason` /
            `.choices[0].message.tool_calls` to detect a tool request.
        """
        # Prepend the system prompt as a message rather than passing it as a param.
        final_messages = (
            [{"role": "system", "content": system}, *messages]
            if system
            else messages
        )

        # Assemble the call. We always send model + messages; api_base/api_key are
        # added only when this provider uses them (Ollama has no key, Groq has no
        # custom base, etc.), so each provider gets exactly what it needs.
        params = {
            "model": self.model,
            "messages": final_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.api_base:
            params["api_base"] = self.api_base
        if self.api_key:
            params["api_key"] = self.api_key
        if tools:
            params["tools"] = tools

        # One call, three providers — litellm routes by the model-string prefix.
        return litellm.completion(**params)
