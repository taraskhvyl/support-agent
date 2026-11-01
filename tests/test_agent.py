import asyncio
import json
from types import SimpleNamespace as NS

from support_agent import agent, approval, config


class FakeSession:
    def __init__(self, error=False):
        self.calls, self.error = [], error

    async def list_tools(self):
        t = NS(name="create_ticket", description="d", inputSchema={"type": "object"})
        return NS(tools=[t, NS(name="search_knowledge_base", description="d", inputSchema={})])

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if self.error:
            raise ValueError("bad args")
        return NS(content=[NS(text=f"result of {name}")])


class FakeClient:
    """Replays scripted assistant messages; records what the model was sent."""

    def __init__(self, *script):
        self.script, self.sent = list(script), []
        self.chat = NS(completions=NS(create=self._create))

    async def _create(self, **kw):
        self.sent.append(kw)
        return NS(choices=[NS(message=self.script.pop(0))])


def tool_msg(name, args, id="c1"):
    return NS(
        content=None, tool_calls=[NS(id=id, function=NS(name=name, arguments=json.dumps(args)))]
    )


def text_msg(t):
    return NS(content=t, tool_calls=None)


def run(s, client, messages, decision=None):
    return asyncio.run(agent.run_loop(s, client, messages, decision, session_id="sid", secret="k"))


USER = {"role": "user", "content": "hi"}
TICKET = {"subject": "Bug", "description": "details"}


def test_plain_answer_has_no_tools():
    s = FakeSession()
    msgs, trace, pending = run(s, FakeClient(text_msg("hello")), [dict(USER)])
    assert msgs[-1]["content"] == "hello" and trace == [] and pending is None


def test_search_tool_runs_without_approval():
    s = FakeSession()
    client = FakeClient(tool_msg("search_knowledge_base", {"query": "q"}), text_msg("answer"))
    msgs, trace, pending = run(s, client, [dict(USER)])
    assert s.calls == [("search_knowledge_base", {"query": "q"})]
    assert pending is None and msgs[-1]["content"] == "answer" and len(trace) == 1


def test_create_ticket_is_gated_not_executed():
    s = FakeSession()
    _msgs, trace, pending = run(s, FakeClient(tool_msg("create_ticket", TICKET)), [dict(USER)])
    assert pending is not None and pending["function"]["name"] == "create_ticket"
    assert s.calls == [] and trace == []


def test_approve_executes_ticket_and_continues():
    s = FakeSession()
    msgs, _, pending = run(s, FakeClient(tool_msg("create_ticket", TICKET)), [dict(USER)])
    msgs, trace, pending = run(s, FakeClient(text_msg("done")), msgs, decision=True)
    token = approval.sign("k", "sid", "create_ticket", TICKET)
    assert s.calls == [("create_ticket", {**TICKET, "approval_token": token})] and pending is None
    assert msgs[-1]["content"] == "done" and trace[0]["result"] == "result of create_ticket"
    assert "approval_token" not in trace[0]["args"]  # token never shown in the UI trace


def test_reject_never_calls_tool():
    s = FakeSession()
    msgs, _, _ = run(s, FakeClient(tool_msg("create_ticket", TICKET)), [dict(USER)])
    msgs, trace, _ = run(s, FakeClient(text_msg("ok, no ticket")), msgs, decision=False)
    assert s.calls == [] and "declined" in trace[0]["result"]


def test_tool_error_is_returned_to_model_not_raised():
    s = FakeSession(error=True)
    client = FakeClient(tool_msg("search_knowledge_base", {"query": "q"}), text_msg("sorry"))
    msgs, trace, _ = run(s, client, [dict(USER)])
    assert trace[0]["result"].startswith("Error:") and msgs[-1]["content"] == "sorry"


def test_step_limit_stops_runaway_loop():
    s = FakeSession()
    client = FakeClient(
        *[tool_msg("search_knowledge_base", {"query": "q"}, f"c{i}") for i in range(20)]
    )
    msgs, _, pending = run(s, client, [dict(USER)])
    assert len(s.calls) == config.MAX_AGENT_STEPS and pending is None
    assert "Step limit" in msgs[-1]["content"]


def test_system_prompt_sent_and_parallel_calls_disabled():
    client = FakeClient(text_msg("x"))
    run(FakeSession(), client, [dict(USER)])
    sent = client.sent[0]
    assert (
        sent["messages"][0]["role"] == "system" and agent.CANARY in sent["messages"][0]["content"]
    )
    assert sent["parallel_tool_calls"] is False


def test_reasoning_effort_sent_only_when_configured(monkeypatch):
    monkeypatch.setattr(config, "REASONING_EFFORT", "none")
    c = FakeClient(text_msg("x"))
    run(FakeSession(), c, [dict(USER)])
    assert c.sent[0]["reasoning_effort"] == "none"
    monkeypatch.setattr(config, "REASONING_EFFORT", "")
    c = FakeClient(text_msg("x"))
    run(FakeSession(), c, [dict(USER)])
    assert "reasoning_effort" not in c.sent[0]


def test_approval_token_param_hidden_from_model():
    tool = NS(
        name="create_ticket",
        description="d",
        inputSchema={
            "type": "object",
            "properties": {"subject": {}, "approval_token": {}},
            "required": ["subject", "approval_token"],
        },
    )
    params = agent.to_openai(tool)["function"]["parameters"]
    assert "approval_token" not in params["properties"] and params["required"] == ["subject"]
    assert "approval_token" in tool.inputSchema["properties"]  # original untouched


def test_cost_cap_sent_to_model():
    c = FakeClient(text_msg("x"))
    run(FakeSession(), c, [dict(USER)])
    assert c.sent[0]["max_completion_tokens"] == config.MAX_COMPLETION_TOKENS


def test_history_trim_keeps_tool_calls_paired(monkeypatch):
    monkeypatch.setattr(config, "MAX_HISTORY_MESSAGES", 3)
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "r"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    out = agent.trim_history(msgs)
    assert out[0]["role"] == "user" and out == [msgs[-1]]  # no orphaned tool message


def test_obedient_model_still_cannot_create_ticket_from_injected_doc():
    """Worst case: the model follows an instruction hidden in a retrieved chunk."""
    s = FakeSession()
    client = FakeClient(
        tool_msg("search_knowledge_base", {"query": "sync"}),
        tool_msg("create_ticket", {"subject": "pwned", "description": "from doc"}),
    )
    _msgs, _trace, pending = run(s, client, [dict(USER)])
    assert pending is not None  # stopped for human approval
    assert all(name != "create_ticket" for name, _ in s.calls)  # nothing executed
