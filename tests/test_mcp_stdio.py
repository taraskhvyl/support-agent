"""Spawn the real MCP server over stdio. No network or API key needed."""

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from support_agent import approval, config


def test_server_lists_tools_and_serves_tickets():
    async def go():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "support_agent.mcp_server"],
            env={
                "SESSION_ID": "t",
                "SUPPORT_AGENT_DB": str(config.DB_PATH),
                "APPROVAL_SECRET": "k",
            },
        )
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            args = {"subject": "abc", "description": "def"}
            unsigned = await s.call_tool("create_ticket", args)
            tok = approval.sign("k", "t", "create_ticket", args)
            created = await s.call_tool("create_ticket", {**args, "approval_token": tok})
            fetched = await s.call_tool("get_ticket", {"ticket_id": 1})
            bad = await s.call_tool("get_ticket", {"ticket_id": -5})
            return (
                names,
                unsigned.content[0].text,
                created.content[0].text,
                fetched.content[0].text,
                bad.isError,
            )

    names, unsigned, created, fetched, bad_is_error = asyncio.run(go())
    assert "requires user approval" in unsigned
    assert names == {"search_knowledge_base", "create_ticket", "get_ticket"}
    assert "Ticket #1" in created and "abc" in fetched and bad_is_error
