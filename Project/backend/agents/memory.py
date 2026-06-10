"""
backend/agents/memory.py
Manages conversation memory for multi-turn agent interactions.
Bridges SQLite persistence with Gemini chat history format.
"""
import logging
from backend.database.sqlite_manager import save_message, get_history, clear_history

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages in-session + persisted conversation history.
    Converts stored history to Gemini-compatible format for multi-turn chat.
    """

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self._in_memory: list[dict] = []

    def add(self, role: str, content: str):
        """
        Add a message to memory.
        role: 'user' | 'assistant'
        """
        self._in_memory.append({"role": role, "content": content})
        save_message(self.conversation_id, role, content)

    def get_recent(self, n: int = 10) -> list[dict]:
        """Get the last n messages from the current session."""
        return self._in_memory[-n:]

    def get_all_from_db(self, limit: int = 20) -> list[dict]:
        """Load history from SQLite."""
        return get_history(self.conversation_id, limit=limit)

    def to_gemini_history(self, limit: int = 10) -> list[dict]:
        """
        Convert recent history to Gemini multi-turn format.
        Returns: [{'role': 'user'|'model', 'parts': [str]}]
        """
        history = self.get_recent(limit)
        gemini_history = []
        for msg in history:
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_history.append({"role": role, "parts": [msg["content"]]})
        return gemini_history

    def get_context_window(self, limit: int = 6) -> str:
        """
        Build a plain-text context string from recent history.
        Used in the RAG prompt.
        """
        recent = self.get_recent(limit)
        if not recent:
            return ""
        lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def resolve_coreferences(self, question: str) -> str:
        """
        Simple heuristic: if question contains 'it', 'its', 'this column',
        'that column', try to replace with the last mentioned column name.
        """
        pronouns = ["it", "its", "this column", "that column", "the column",
                    "this feature", "that feature"]
        q_lower = question.lower()
        has_pronoun = any(p in q_lower for p in pronouns)
        if not has_pronoun:
            return question

        # Search recent history for a column mention
        recent = self.get_recent(4)
        last_column = None
        for msg in reversed(recent):
            content = msg.get("content", "")
            # Look for patterns like "Column: X", "'X' column", "column X"
            import re
            matches = re.findall(
                r"(?:column[:\s]+['\"]?(\w+)['\"]?|['\"](\w+)['\"]?\s+column)",
                content, re.IGNORECASE
            )
            if matches:
                last_column = next((m[0] or m[1] for m in matches if m[0] or m[1]), None)
                if last_column:
                    break
            # Also check for column names mentioned in backticks
            bt = re.findall(r"`(\w+)`", content)
            if bt:
                last_column = bt[-1]
                break

        if last_column:
            for p in pronouns:
                question = re.sub(
                    rf"\b{re.escape(p)}\b",
                    f"'{last_column}'",
                    question,
                    flags=re.IGNORECASE,
                )
            logger.info(f"Resolved coreference → '{last_column}': {question}")
        return question

    def clear(self):
        self._in_memory.clear()
        clear_history(self.conversation_id)