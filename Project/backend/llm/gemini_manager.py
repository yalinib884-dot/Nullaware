"""
backend/llm/gemini_manager.py
Manages Gemini API calls with round-robin key rotation and retry logic.
"""
import logging
import time
import google.generativeai as genai
from config import GEMINI_API_KEYS, GEMINI_MODEL
from .round_robin import RoundRobinKeyManager

logger = logging.getLogger(__name__)

_key_manager = RoundRobinKeyManager(GEMINI_API_KEYS) if GEMINI_API_KEYS else None

QUOTA_ERRORS = ("429", "quota", "resource_exhausted", "rate_limit")


def _is_quota_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(kw in msg for kw in QUOTA_ERRORS)


def call_gemini(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.2,
    max_retries: int = 3,
) -> str:
    """
    Call Gemini with automatic key rotation on quota errors.

    Args:
        prompt: User message/prompt.
        system_prompt: Optional system-level instruction.
        temperature: Generation temperature.
        max_retries: Number of retries across available keys.

    Returns:
        Model response text.
    """
    if _key_manager is None:
        raise RuntimeError(
            "No Gemini API keys configured. Set GEMINI_API_KEY_1 in .env."
        )

    last_error = None
    for attempt in range(max_retries):
        current_key = _key_manager.current_key
        try:
            genai.configure(api_key=current_key)
            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=system_prompt if system_prompt else None,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=4096,
                ),
            )
            response = model.generate_content(prompt)
            return response.text

        except Exception as e:
            last_error = e
            logger.error(f"Gemini call failed (attempt {attempt + 1}): {e}")

            if _is_quota_error(e):
                _key_manager.mark_exhausted(current_key)
                if _key_manager.has_available():
                    _key_manager.next_key()
                    logger.info("Switched to next API key after quota error.")
                else:
                    logger.error("All API keys exhausted.")
                    break
            else:
                # Non-quota error: brief wait then retry same key
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Gemini API call failed after {max_retries} attempts: {last_error}")


def call_gemini_with_history(
    history: list[dict],
    new_message: str,
    system_prompt: str = "",
    temperature: float = 0.2,
) -> str:
    """
    Call Gemini with conversation history for multi-turn conversations.

    Args:
        history: List of {'role': 'user'|'model', 'parts': [str]} dicts.
        new_message: The new user message.
        system_prompt: System instruction.
        temperature: Generation temperature.

    Returns:
        Model response text.
    """
    if _key_manager is None:
        raise RuntimeError("No Gemini API keys configured.")

    genai.configure(api_key=_key_manager.current_key)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt if system_prompt else None,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=4096,
        ),
    )

    chat = model.start_chat(history=history)
    response = chat.send_message(new_message)
    return response.text