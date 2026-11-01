"""MCP server (stdio): search_knowledge_base, create_ticket, get_ticket.

All validation lives here: the model is never trusted.
"""

import os
from functools import lru_cache
from typing import Annotated

import chromadb
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from pydantic import Field

from . import approval, config, tickets

mcp = FastMCP("support")


@lru_cache
def _openai() -> OpenAI:
    return OpenAI()


@lru_cache
def _collection() -> chromadb.Collection:
    return chromadb.PersistentClient(str(config.CHROMA_DIR)).get_collection(config.COLLECTION)


def _session_id() -> str:
    return os.environ.get("SESSION_ID", "default")  # set by the agent, not by the model


def _format_hit(doc: str, meta: dict, distance: float) -> str:
    """Render one chunk as a fenced <doc> block. Chunk text is DATA, not instructions."""
    relevance = "relevant" if distance <= config.RELEVANCE_MAX_DISTANCE else "NOT_RELEVANT"
    doc = doc.replace("</doc>", "<\\/doc>")  # a chunk must not be able to close its own fence
    return (
        f'<doc source="{meta["source"]}" heading="{meta["heading"]}" '
        f'distance="{distance:.2f}" {relevance}>\n{doc}\n</doc>'
    )


@mcp.tool()
def search_knowledge_base(
    query: Annotated[str, Field(min_length=1, max_length=300)],
    k: Annotated[int, Field(ge=1, le=5)] = 4,
) -> str:
    """Search the Obsidian help docs. Returns chunks with source and relevance."""
    embedding = _openai().embeddings.create(model=config.EMBED_MODEL, input=query).data[0].embedding
    hits = _collection().query(query_embeddings=[embedding], n_results=k)
    rows = zip(hits["documents"][0], hits["metadatas"][0], hits["distances"][0], strict=True)
    return "\n".join(_format_hit(*row) for row in rows)


@mcp.tool()
def create_ticket(
    subject: Annotated[str, Field(min_length=3, max_length=120)],
    description: Annotated[str, Field(min_length=3, max_length=2000)],
    approval_token: Annotated[str, Field(max_length=128)] = "",
) -> str:
    """Create a support ticket. Requires user approval (a token the system adds, not the model)."""
    args = {"subject": subject, "description": description}
    secret = os.environ.get("APPROVAL_SECRET", "")
    if not approval.verify(secret, _session_id(), "create_ticket", args, approval_token):
        return "Error: this action requires user approval."
    if tickets.count(_session_id()) >= config.MAX_TICKETS_PER_SESSION:
        return "Error: ticket limit for this session reached."
    return f"Ticket #{tickets.create(_session_id(), subject, description)} created."


@mcp.tool()
def get_ticket(ticket_id: Annotated[int, Field(ge=1)]) -> str:
    """Get a ticket created in this session."""
    t = tickets.get(_session_id(), ticket_id)
    return f"#{t.id} [{t.status}] {t.subject}: {t.description}" if t else "Ticket not found."


def main() -> None:
    mcp.run()  # stdio


if __name__ == "__main__":
    main()
