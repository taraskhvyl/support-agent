import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")  # OPENAI_API_KEY lives only in .env / env; never printed

DOCS_DIR = ROOT / "data" / "obsidian-help" / "en"
CHROMA_DIR = ROOT / "data" / "chroma"
DB_PATH = Path(os.environ.get("SUPPORT_AGENT_DB", ROOT / "data" / "tickets.db"))

COLLECTION = "knowledge_base"  # chroma requires >= 3 chars
CHAT_MODEL = os.environ.get(
    "CHAT_MODEL", "gpt-6-luna"
)  # cheapest current model w/ tool calling; override in .env
# gpt-6 models reject function tools unless reasoning_effort is set to "none" (Chat Completions).
# Set CHAT_REASONING_EFFORT= (empty) for models that don't accept the parameter.
REASONING_EFFORT = os.environ.get("CHAT_REASONING_EFFORT", "none")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

CHUNK_CHARS = 2000  # ~500 tokens; ponytail: chars not tokens, upgrade to tiktoken if needed
CHUNK_OVERLAP = 200
MIN_CHUNK_CHARS = 50
RELEVANCE_MAX_DISTANCE = 0.6  # cosine distance; tune with evals/run.py

MAX_MESSAGE_CHARS = 1000
MAX_AGENT_STEPS = 5
MAX_COMPLETION_TOKENS = 600
MAX_HISTORY_MESSAGES = 20  # messages sent to the model; older ones are dropped
MAX_USER_MESSAGES_PER_SESSION = 30
MAX_TICKETS_PER_SESSION = 3
