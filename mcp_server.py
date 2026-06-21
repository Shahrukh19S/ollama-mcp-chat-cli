"""
mcp_server.py — the MCP server that exposes our "documents" to the model.

WHAT THIS FILE IS
-----------------
An MCP (Model Context Protocol) server built with FastMCP. MCP is a standard way to
give a model three kinds of capabilities over some data:

    * TOOLS     — actions the MODEL can choose to call (read a doc, edit a doc).
    * RESOURCES — data the APP fetches by URI (list of doc ids, one doc's content).
    * PROMPTS   — reusable, parameterized instructions the USER triggers (/format, /summarize).

IMPORTANT — "documents" are NOT real files. They are just the in-memory `docs`
dict below: filename -> content string. No filesystem, no uploads. Editing a doc
mutates this dict, so an edit only persists for the life of this process.

HOW IT FITS THE WHOLE
---------------------
`main.py` launches this file as a subprocess and talks to it over stdio. The
MCP client (`mcp_client.py`) calls list_tools / call_tool / list_prompts /
get_prompt / read_resource against the capabilities registered here. The agent
loop then offers the tools to the model and runs whichever ones it asks for.
"""

from mcp.server.fastmcp import FastMCP

# FastMCP's prompt helpers: `base.UserMessage(...)` / `base.AssistantMessage(...)`
# build the chat-message objects a prompt must return.
from mcp.server.fastmcp.prompts import base

# pydantic.Field lets us attach human-readable descriptions to tool arguments.
# Those descriptions become part of the tool's JSON schema, which the model reads
# to understand what each argument means — so good descriptions = better tool use.
from pydantic import Field

# The server. The name shows up in the Inspector; log_level="ERROR" keeps the
# stdio channel quiet (stray logs on stdout would corrupt the MCP protocol stream).
mcp = FastMCP("DocumentMCP", log_level="ERROR")


# The "documents". Filename -> content. This dict IS our entire data store.
docs = {
    "welcome.md": "Brightwater Coffee Co. is a small neighborhood roastery founded in 2021. We serve single-origin pour-overs, house blends, and seasonal cold brew, and we roast every batch on-site.",
    "supplier-notes.txt": "raw notes - green beans from three farms: la esperanza (colombia, washed), mountain mist (ethiopia, natural), rio verde (brazil, pulped natural). la esperanza ships monthly, mountain mist is seasonal (oct-feb), rio verde on demand. need to confirm 2026 contracts before september.",
    "weekly-plan.md": "## Week plan\n- Monday: receive the Colombia shipment, cup and log it.\n- Wednesday: rotate the seasonal menu to the winter blend.\n- Friday: staff training on the new espresso machine.\n- Weekend: run the pop-up stall at the farmers market.",
    "equipment.pdf": "The new espresso machine is a dual-boiler unit with PID temperature control. The recommended brew temperature is 93C, and the group head should be flushed for two seconds before each shot.",
    "customer-feedback.txt": "Feedback this month: many regulars love the Ethiopia natural pour-over and the friendly morning staff. A few asked for oat milk to be the default option and for more seating during the weekday rush. One customer noted the cold brew was occasionally sold out by noon.",
    "expansion.docx": "Draft idea: open a second, smaller location near the university by late 2027. It would focus on fast espresso drinks and bagged retail beans rather than full pour-over service. Funding would come from reinvested profit, not outside investors.",
}


# ---------------------------------------------------------------------------
# TOOLS — actions the MODEL can decide to call during the agent loop.
# ---------------------------------------------------------------------------


@mcp.tool(
    name="read_document",
    description="Read the full contents of a document given its id (its filename).",
)
def read_document(
    doc_id: str = Field(description="The id (filename) of the document to read, e.g. 'welcome.md'."),
) -> str:
    """
    Return the content string for `doc_id`.

    Raises a clear ValueError if the id is unknown — MCP turns a raised exception
    into a tool error the model can see and react to, which is far better than
    silently returning nothing.
    """
    if doc_id not in docs:
        raise ValueError(f"Doc with id '{doc_id}' not found.")
    return docs[doc_id]


@mcp.tool(
    name="edit_document",
    description="Edit a document by replacing an exact string with a new string.",
)
def edit_document(
    doc_id: str = Field(description="The id (filename) of the document to edit."),
    old_str: str = Field(description="The exact text to find and replace. Must match exactly, including whitespace."),
    new_str: str = Field(description="The new text to insert in place of old_str."),
) -> str:
    """
    Replace `old_str` with `new_str` inside the document and persist it in `docs`.

    Because `docs` is a module-level dict, the change persists for the rest of this
    server process's life (i.e. the whole CLI session). Returns a short confirmation
    string so the model gets feedback that the edit happened.
    """
    if doc_id not in docs:
        raise ValueError(f"Doc with id '{doc_id}' not found.")
    # Plain string replace — replaces every occurrence of old_str.
    docs[doc_id] = docs[doc_id].replace(old_str, new_str)
    return f"Successfully edited '{doc_id}'."


# ---------------------------------------------------------------------------
# RESOURCES — data the APP fetches by URI (the model doesn't call these).
# ---------------------------------------------------------------------------


@mcp.resource(
    "docs://documents",
    mime_type="application/json",
)
def list_docs() -> list[str]:
    """
    Return every document id as a JSON array.

    The CLI reads this at startup to power @-mention autocompletion (the list of
    docs you can reference). We declare mime_type application/json so the client
    knows to JSON-parse the body into a Python list.
    """
    return list(docs.keys())


@mcp.resource(
    "docs://documents/{doc_id}",
    mime_type="text/plain",
)
def fetch_doc(doc_id: str) -> str:
    """
    Return the content of one document, addressed by a templated URI.

    `{doc_id}` in the URI is filled in by the caller, e.g.
    "docs://documents/welcome.md". This is how the CLI injects a doc's text
    when you write "@welcome.md".
    """
    if doc_id not in docs:
        raise ValueError(f"Doc with id '{doc_id}' not found.")
    return docs[doc_id]


# ---------------------------------------------------------------------------
# PROMPTS — reusable instructions the USER triggers with /commands.
# A prompt returns a list of chat messages that get prepended to the conversation.
# ---------------------------------------------------------------------------


@mcp.prompt(
    name="format",
    description="Reformat a document into clean, well-structured markdown.",
)
def format_document(
    doc_id: str = Field(description="The id (filename) of the document to reformat."),
) -> list[base.Message]:
    """
    Build the instruction that asks the model to rewrite a doc as tidy markdown.

    We return a single user message. Note we tell the model to USE the
    edit_document tool to apply its changes — so /format becomes an agentic action,
    not just a printed suggestion.
    """
    prompt = f"""
    Your goal is to reformat the document with id '{doc_id}' into clean, well-structured markdown.

    Use the 'read_document' tool to get the current contents, decide on improved
    markdown formatting (headings, lists, emphasis where helpful), and then apply
    your changes using the 'edit_document' tool. Make as many edits as needed.
    Do not change the meaning of the text — only its formatting.
    """
    return [base.UserMessage(prompt)]


@mcp.prompt(
    name="summarize",
    description="Summarize the contents of a document.",
)
def summarize_document(
    doc_id: str = Field(description="The id (filename) of the document to summarize."),
) -> list[base.Message]:
    """
    Build the instruction that asks the model to summarize a doc.

    Again a single user message; the model is told to read the doc first (via the
    read_document tool) and then produce a concise summary.
    """
    prompt = f"""
    Summarize the document with id '{doc_id}'.

    First use the 'read_document' tool to get its contents, then write a short,
    clear summary of the key points. Keep it concise.
    """
    return [base.UserMessage(prompt)]


if __name__ == "__main__":
    # Run as a stdio server: the parent process (main.py) speaks MCP over the
    # subprocess's stdin/stdout. This is why nothing else may print to stdout.
    mcp.run(transport="stdio")
