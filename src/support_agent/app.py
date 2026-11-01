"""Streamlit chat UI. Shows every tool call under the answer and asks before creating tickets."""

import asyncio
import json
import uuid

import streamlit as st

from support_agent import agent, config

state = st.session_state


def init_state() -> None:
    state.setdefault("session_id", uuid.uuid4().hex)
    state.setdefault("messages", [])
    state.setdefault("traces", {})  # index of the final assistant message -> tool calls of its turn
    state.setdefault("turn_trace", [])  # tool calls of the turn in progress
    state.setdefault("pending", None)  # gated tool call waiting for approval


def advance(decision: bool | None = None) -> None:
    """Run the agent; keep the turn's tool calls and attach them to the final answer."""
    result = asyncio.run(agent.run(state.messages, state.session_id, decision))
    state.messages, state.pending = result.messages, result.pending
    state.turn_trace += result.trace
    if not result.pending:
        state.traces[len(result.messages) - 1] = state.turn_trace
        state.turn_trace = []


def render_history() -> None:
    for i, message in enumerate(state.messages):
        if message["role"] not in ("user", "assistant") or not message.get("content"):
            continue
        with st.chat_message(message["role"]):
            st.write(message["content"])
            for call in state.traces.get(i, []):
                with st.expander(f"🔧 {call['tool']}({call['args']})"):
                    st.code(call["result"])


def render_approval() -> None:
    args = json.loads(state.pending["function"]["arguments"])
    st.warning(f"Create ticket?\n\n**{args.get('subject')}**\n\n{args.get('description')}")
    approve, reject = st.columns(2)
    if approve.button("Approve"):
        advance(True)
        st.rerun()
    if reject.button("Reject"):
        advance(False)
        st.rerun()


def handle_prompt(prompt: str) -> None:
    sent = sum(m["role"] == "user" for m in state.messages)
    if sent >= config.MAX_USER_MESSAGES_PER_SESSION:
        st.error("Session message limit reached. Reload the page to start a new session.")
    elif len(prompt) > config.MAX_MESSAGE_CHARS:
        st.error(f"Message too long (max {config.MAX_MESSAGE_CHARS} chars).")
    else:
        state.messages.append({"role": "user", "content": prompt})
        advance()
        st.rerun()


def main() -> None:
    st.title("Obsidian Support Agent")
    init_state()
    render_history()
    if state.pending:
        render_approval()
    elif prompt := st.chat_input("Ask about Obsidian"):
        handle_prompt(prompt)


main()
