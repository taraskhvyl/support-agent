"""Smoke test: the Streamlit app renders without errors (no network, no API key)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).parents[1] / "src" / "support_agent" / "app.py"


def test_app_renders_empty_chat():
    at = AppTest.from_file(str(APP)).run(timeout=30)
    assert not at.exception
    assert at.title[0].value == "Obsidian Support Agent"
    assert len(at.chat_input) == 1


def test_overlong_message_is_rejected_before_calling_the_agent():
    at = AppTest.from_file(str(APP)).run(timeout=30)
    at.chat_input[0].set_value("x" * 5000).run()
    assert "too long" in at.error[0].value
    assert at.session_state["messages"] == []
