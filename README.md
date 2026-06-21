# MCP Chat CLI

MCP Chat is a command-line chat application that talks to AI models through the **MCP
(Model Context Protocol)** architecture, with document retrieval, command-based prompts, and
tool integrations. It runs on a **free local model via [Ollama](https://ollama.com)
(`qwen2.5:7b`)** by default, with **Groq** and **GitHub Models** as cloud fallbacks — all
behind one provider-agnostic layer powered by [litellm](https://github.com/BerriAI/litellm).

## ONE-LINER:
A Claude like mcp cli chat application on simulated set of document profiles built on all free AI models

## Features

- **Local-first:** runs `qwen2.5:7b` on your machine — $0, offline, private.
- **One-line provider switch:** Ollama ↔ Groq ↔ GitHub Models via two `.env` variables.
- **MCP server** (`mcp_server.py`) exposing tools, resources, and prompts over an in-memory
  document store.
- **Agentic tool-calling loop** — the model can read and edit documents to answer questions.
- `@document` mentions and `/command` prompts with autocompletion.

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.com) running, with the model pulled: `ollama pull qwen2.5:7b`
- Node.js / `npx` (only needed for the optional MCP Inspector)
- *(Optional)* a Groq API key and/or a GitHub PAT for the cloud fallbacks

## Setup

### 1. Configure environment variables

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

```
LLM_PROVIDER=ollama
LLM_MODEL=ollama_chat/qwen2.5:7b
OLLAMA_API_BASE=http://localhost:11434

# Optional cloud fallbacks:
GROQ_API_KEY=your-groq-api-key-here
GITHUB_MODELS_API_KEY=your-github-pat-with-models-read-scope
GITHUB_MODELS_API_BASE=https://models.github.ai/inference

USE_UV=1
```

> `.env` is git-ignored — your keys never leave your machine.

### 2. Install dependencies

```bash
uv sync
```

### 3. Run

```bash
uv run main.py
```

> Run in a real terminal (Windows Terminal / a normal shell). The CLI uses `prompt_toolkit`,
> which needs an interactive console.

## Usage

- **Basic chat:** type a message and press Enter.
- **Document retrieval:** mention a document with `@`:
  ```
  > What is in @welcome.md?
  ```
- **Commands:** use `/` to run a server prompt (include the document id):
  ```
  > /summarize customer-feedback.txt
  ```
- **Tools:** ask the model to read or edit a document and it will call the MCP tools:
  ```
  > Using your tools, read equipment.pdf and tell me the recommended brew temperature.
  ```

## Switching providers

Edit `.env` and restart:

| Provider | `LLM_PROVIDER` | `LLM_MODEL` |
|---|---|---|
| Ollama (local, default) | `ollama` | `ollama_chat/qwen2.5:7b` |
| Groq (cloud fallback) | `groq` | `groq/openai/gpt-oss-120b` |
| GitHub Models (cloud fallback) | `github` | `openai/gpt-4o-mini` |

## Inspecting the MCP server

```bash
uv run mcp dev mcp_server.py
```

Opens the MCP Inspector in your browser to browse and call the server's tools, resources, and
prompts.

## Project layout

| File | Role |
|---|---|
| `main.py` | Entry point: config, build services, start the CLI. |
| `agent/llm_service.py` | Provider-agnostic LLM adapter (Ollama/Groq/GitHub via litellm). |
| `agent/agent_loop.py` | The agentic tool-calling loop. |
| `agent/cli_agent.py` | `@`-mention injection and `/command` handling. |
| `agent/terminal_ui.py` | Terminal UI and autocompletion. |
| `agent/tool_manager.py` | MCP-tool ↔ OpenAI-tool conversion and execution. |
| `mcp_client.py` | MCP client over stdio. |
| `mcp_server.py` | MCP server over the in-memory document store. |

## Development

- **Add documents:** edit the `docs` dictionary in `mcp_server.py`. (Documents are an
  in-memory dict, not real files — edits reset when the app restarts.)

## License

Released under the [MIT License](./LICENSE) — © 2026 Abdullah Ansari (Shahrukh19S).

## Credits & acknowledgements

This project started from the starter scaffold in Anthropic's
course. I rebuilt it to run on a **free local model (Ollama)** with **Groq** and **GitHub
Models** cloud fallbacks via **[litellm](https://github.com/BerriAI/litellm)**, implemented the
MCP server and client and the agentic tool-calling loop, and replaced the example content and
the terminal UI with my own.

Thanks to **Anthropic** for the scaffold, and to **Claude** (via Claude Code) for pairing on the
build — the commit history keeps the `Co-Authored-By: Claude` trailers for provenance.
