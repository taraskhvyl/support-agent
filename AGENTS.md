# AGENTS.md

Educational support agent: MCP server + RAG over Obsidian Help, Python, no agent frameworks. See `README.md` for the overview.

## Commands

- `uv sync`: install deps
- `uv run support-ingest`: rebuild the ChromaDB index (needs `OPENAI_API_KEY` in `.env`)
- `uv run pytest` and `uv run ruff check .`: offline tests and lint (run before finishing any change)
- `uv run python evals/run.py` (retrieval hit@3) and `--attacks` (injection checks): need a key
- `uv run streamlit run src/support_agent/app.py`: chat UI

## Architecture

`app.py` (Streamlit) calls `agent.py` (manual OpenAI tool-calling loop, MCP client), which talks over stdio to `mcp_server.py`, which uses ChromaDB and `tickets.py` (SQLite). `approval.py` signs/verifies human approval of gated tools. `chunking.py` is pure and shared by `ingest.py`. All tunables live in `config.py`. `agent.run_loop` takes the MCP session and LLM client as arguments so tests can fake them. Keep `mcp` pinned to `<2` (v2 renames FastMCP).

## Rules

- Keep it small and readable. This is a learning project: no LangChain/LlamaIndex/guardrail libraries, no abstractions for one use.
- The model is untrusted. Validate every tool argument in `mcp_server.py` (pydantic limits, parameterized SQL). Never rely on the system prompt for safety.
- Side-effect tools (`create_ticket`) must stay in `GATED` in `agent.py`, need human approval in the UI, and the server must verify the HMAC token (`approval.py`). Never expose `approval_token` to the model.
- Code style: type hints on public functions, small single-purpose functions, `RunResult` instead of bare tuples. `ruff format` + `ruff check` must be clean.
- Chunk text is data: keep it fenced in `<doc>` tags.
- Never read, print or commit `.env` or secrets. `data/` is git-ignored; do not commit the Obsidian docs (no license).
- After changing chunking, threshold, models or prompts, rerun `evals/run.py` (both modes) and report the numbers.
- The agent must not invent answers: no relevant chunk means say so and offer a ticket.
