"""
agent/terminal_ui.py — the terminal user interface.

WHAT THIS FILE DOES
-------------------
This is the front desk of the app: it draws the `> ` prompt, reads what you type,
offers autocompletion, and prints the model's answer. It is built on the
third-party `prompt_toolkit` library (a normal pip dependency — we just *use* it).

Two kinds of autocompletion are provided:
  * `@name`  -> completes **document ids** (so you can inject a doc into your query).
  * `/name`  -> completes **prompt commands** (e.g. /summarize), and then the
                document-id argument that command takes.

It exposes one class, `CliApp`, that `main.py` builds and runs. `CliApp` talks to
the chat "agent" (`CliChat`) through three methods only: `agent.run(text)`,
`agent.list_docs_ids()`, and `agent.list_prompts()`.

NOTE ON THE ENTER KEY (intentional UX): when the `/` completion menu is open,
pressing Enter **accepts the highlighted command and keeps you editing** (it does
NOT submit the line). That lets you pick `/summarize`, then type the document id,
then press Enter again to actually run it. See `_handle_enter` below.
"""

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

from agent.cli_agent import CliChat


class DocumentCompleter(Completer):
    """
    Supplies the autocompletion entries for the prompt.

    It holds two small caches the CLI refreshes from the MCP server at startup:
      * `_doc_ids`  — the list of document ids (for `@` mentions and command args).
      * `_prompts`  — the list of prompt objects (for `/` command names).

    `get_completions` is called by prompt_toolkit on every keystroke (because we
    enable complete-while-typing) and yields the matching suggestions.
    """

    def __init__(self):
        self._doc_ids: list[str] = []
        self._prompts: list = []

    def set_doc_ids(self, doc_ids: list[str]) -> None:
        """Refresh the document-id suggestions (called by CliApp.refresh_resources)."""
        self._doc_ids = doc_ids

    def set_prompts(self, prompts: list) -> None:
        """Refresh the prompt-command suggestions (called by CliApp.refresh_prompts)."""
        self._prompts = prompts

    def _complete_doc_ids(self, prefix: str):
        """Yield every document id that starts with `prefix` (case-insensitive)."""
        for doc_id in self._doc_ids:
            if doc_id.lower().startswith(prefix.lower()):
                # start_position=-len(prefix) tells prompt_toolkit to replace the
                # text the user already typed for the mention/argument.
                yield Completion(
                    doc_id,
                    start_position=-len(prefix),
                    display=doc_id,
                    display_meta="document",
                )

    def get_completions(self, document, complete_event):
        # We only look at the text to the LEFT of the cursor — that's what the
        # user is currently typing.
        text = document.text_before_cursor

        # --- Case 1: an "@" mention anywhere in the line -> complete document ids.
        if "@" in text:
            prefix = text[text.rfind("@") + 1:]
            # Only while still typing the mention itself (no space yet).
            if " " not in prefix:
                yield from self._complete_doc_ids(prefix)
                return

        # --- Case 2: a "/command" -> complete the command name, then its doc-id arg.
        if text.startswith("/"):
            parts = text[1:].split()

            # 2a. Still typing the command word (no trailing space) -> command names.
            if len(parts) <= 1 and not text.endswith(" "):
                cmd_prefix = parts[0] if parts else ""
                for prompt in self._prompts:
                    if prompt.name.startswith(cmd_prefix):
                        yield Completion(
                            prompt.name,
                            start_position=-len(cmd_prefix),
                            display=f"/{prompt.name}",
                            display_meta=prompt.description or "",
                        )
                return

            # 2b. Command chosen, now typing its argument -> complete document ids.
            #     If we're right after the trailing space, the prefix is empty and
            #     we list all docs; otherwise we filter by what's been typed.
            arg_prefix = "" if text.endswith(" ") else parts[-1]
            yield from self._complete_doc_ids(arg_prefix)
            return


def _handle_enter(buffer) -> None:
    """
    Decide what Enter does, based on whether the completion menu is open.

    This is the key UX rule of this file:
      * Menu open AND a completion is highlighted -> accept it into the line and
        add a trailing space, then KEEP EDITING (do not submit). This lets the user
        pick `/summarize` and continue typing the document id.
      * Menu open but nothing highlighted -> just close the menu, keep editing.
      * No menu open -> submit the line normally (the default Enter behavior).

    Pulling the logic into a plain function (instead of burying it in the key
    binding) makes it directly unit-testable without a live terminal.
    """
    state = buffer.complete_state

    if state is not None and state.current_completion is not None:
        # Lock in the highlighted completion...
        buffer.apply_completion(state.current_completion)
        # ...and add a space so the next thing typed is the argument, not glued on.
        if not buffer.text.endswith(" "):
            buffer.insert_text(" ")
        return

    if state is not None:
        # Menu showing but the user hasn't picked anything — close it, stay editing.
        buffer.complete_state = None
        return

    # Nothing to complete -> this Enter means "run my line".
    buffer.validate_and_handle()


class CliApp:
    """
    The interactive terminal app: prompt loop, autocompletion, history.

    `main.py` constructs this with the chat agent, then calls `initialize()` once
    (to load doc ids + prompt commands for autocompletion) and `run()` to start the
    read-eval-print loop.
    """

    def __init__(self, agent: CliChat):
        self.agent = agent
        self.doc_ids: list[str] = []
        self.prompts: list = []

        # The completer is shared with the PromptSession; we refresh its caches in
        # initialize() once the MCP server is connected.
        self.completer = DocumentCompleter()

        # Custom key bindings — currently just our Enter behavior (see _handle_enter).
        bindings = KeyBindings()

        @bindings.add("enter")
        def _(event):
            _handle_enter(event.current_buffer)

        # A little colour for the completion menu; purely cosmetic.
        style = Style.from_dict(
            {
                "completion-menu.completion": "bg:#222222 #ffffff",
                "completion-menu.completion.current": "bg:#444444 #ffffff",
            }
        )

        self.session = PromptSession(
            completer=self.completer,
            history=InMemoryHistory(),
            key_bindings=bindings,
            style=style,
            # Show suggestions as the user types, computed off the main thread so
            # the UI stays responsive.
            complete_while_typing=True,
            complete_in_thread=True,
        )

    async def initialize(self) -> None:
        """Load the autocompletion data once, before the prompt loop starts."""
        await self.refresh_resources()
        await self.refresh_prompts()

    async def refresh_resources(self) -> None:
        """Pull the current document ids from the server into the completer."""
        try:
            self.doc_ids = await self.agent.list_docs_ids()
            self.completer.set_doc_ids(self.doc_ids)
        except Exception as e:
            # Non-fatal: the app still runs, you just won't get @-completion.
            print(f"Error refreshing resources: {e}")

    async def refresh_prompts(self) -> None:
        """Pull the current prompt commands from the server into the completer."""
        try:
            self.prompts = await self.agent.list_prompts()
            self.completer.set_prompts(self.prompts)
        except Exception as e:
            # Non-fatal: the app still runs, you just won't get /-completion.
            print(f"Error refreshing prompts: {e}")

    async def run(self) -> None:
        """
        The main read-eval-print loop.

        Read a line, hand it to the agent (which does @/command handling + the
        tool-calling loop), and print the answer. Ctrl+C (or Ctrl+D) exits cleanly.
        """
        while True:
            try:
                user_input = await self.session.prompt_async("> ")

                # Ignore blank lines so an accidental Enter doesn't call the model.
                if not user_input.strip():
                    continue

                response = await self.agent.run(user_input)
                print(f"\nResponse:\n{response}\n")

            except (KeyboardInterrupt, EOFError):
                # Ctrl+C / Ctrl+D -> leave the loop and let the app shut down.
                break
