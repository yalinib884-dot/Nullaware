"""
backend/llm/round_robin.py
Round-robin API key rotation with quota-exceeded fallback.
"""
import logging
import threading

logger = logging.getLogger(__name__)


class RoundRobinKeyManager:
    """
    Thread-safe round-robin manager for multiple API keys.
    Automatically skips keys that have exceeded their quota.
    """

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("At least one API key must be provided.")
        self._keys = keys
        self._index = 0
        self._exhausted: set[int] = set()
        self._lock = threading.Lock()

    @property
    def current_key(self) -> str:
        with self._lock:
            return self._keys[self._index]

    def next_key(self) -> str:
        """Advance to the next available key."""
        with self._lock:
            start = self._index
            for _ in range(len(self._keys)):
                self._index = (self._index + 1) % len(self._keys)
                if self._index not in self._exhausted:
                    logger.info(f"Rotated to API key index {self._index}")
                    return self._keys[self._index]
            raise RuntimeError("All API keys have been exhausted.")

    def mark_exhausted(self, key: str):
        """Mark a key as quota-exceeded."""
        with self._lock:
            try:
                idx = self._keys.index(key)
                self._exhausted.add(idx)
                logger.warning(f"API key index {idx} marked as exhausted.")
            except ValueError:
                pass

    def has_available(self) -> bool:
        with self._lock:
            return len(self._exhausted) < len(self._keys)

    def reset(self):
        """Reset all exhausted keys (e.g., after quota refresh)."""
        with self._lock:
            self._exhausted.clear()
            logger.info("All API keys reset.")