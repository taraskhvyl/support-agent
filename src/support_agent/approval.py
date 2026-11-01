"""HMAC proof that a human approved a specific tool call.
The agent signs after the user clicks Approve; the MCP server refuses unsigned gated calls.
The model never sees the secret or the token, so it cannot forge approval."""

import hashlib
import hmac
import json


def sign(secret: str, session_id: str, name: str, args: dict) -> str:
    msg = json.dumps({"s": session_id, "n": name, "a": args}, sort_keys=True).encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


def verify(secret: str, session_id: str, name: str, args: dict, token: str) -> bool:
    return bool(secret) and hmac.compare_digest(sign(secret, session_id, name, args), token or "")
