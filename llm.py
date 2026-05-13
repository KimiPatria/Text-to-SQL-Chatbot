import logging
from threading import Lock

from groq import Groq

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_MODEL_120B,
    GROQ_MODEL_20B,
    DEFAULT_PROVIDER,
    ROUTING_MODEL,
)

log = logging.getLogger(__name__)

VALID_PROVIDERS = {"groq", "gpt120b", "gpt20b"}

PROVIDER_LABELS = {
    "groq":    "Llama 3.3 70B" if "llama-3.3" in GROQ_MODEL else GROQ_MODEL,
    "gpt120b": "GPT OSS 120B",
    "gpt20b":  "GPT OSS 20B",
}

_PROVIDER_MODELS = {
    "groq":    GROQ_MODEL,
    "gpt120b": GROQ_MODEL_120B,
    "gpt20b":  GROQ_MODEL_20B,
}

_groq_client: Groq | None = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

_state_lock = Lock()


def _initial_provider() -> str:
    p = DEFAULT_PROVIDER if DEFAULT_PROVIDER in VALID_PROVIDERS else "groq"
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is required but not set.")
    return p


_active_provider: str = _initial_provider()


def get_provider() -> str:
    return _active_provider


def list_providers() -> dict[str, dict]:
    available = bool(GROQ_API_KEY)
    return {
        key: {
            "label": PROVIDER_LABELS[key],
            "model": _PROVIDER_MODELS[key],
            "available": available,
        }
        for key in ("groq", "gpt120b", "gpt20b")
    }


def set_provider(name: str) -> tuple[bool, str | None]:
    name = (name or "").lower().strip()
    if name not in VALID_PROVIDERS:
        return False, f"Unknown provider '{name}'. Use one of: {', '.join(sorted(VALID_PROVIDERS))}."
    if not GROQ_API_KEY:
        return False, "Groq API key not configured"
    global _active_provider
    with _state_lock:
        _active_provider = name
    return True, None


def _call_groq(messages: list[dict], temperature: float, max_tokens: int, model: str) -> tuple[str, dict]:
    if not _groq_client:
        raise RuntimeError("Groq API key not configured")
    completion = _groq_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    u = completion.usage
    usage = {
        "input_tokens":  u.prompt_tokens if u else 0,
        "output_tokens": u.completion_tokens if u else 0,
        "total_tokens":  u.total_tokens if u else 0,
        "model": model,
    }
    return (completion.choices[0].message.content or "").strip(), usage


def call_llm(
    messages: list[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int = 800,
    provider: str | None = None,
) -> tuple[str, str, dict]:
    """Route a chat-style messages list to the active (or specified) provider.

    Returns (text, provider_used, usage_dict).
    """
    p = (provider or _active_provider).lower()
    if p not in VALID_PROVIDERS:
        raise ValueError(f"Unknown provider: {p}")
    model = _PROVIDER_MODELS[p]
    text, usage = _call_groq(messages, temperature, max_tokens, model)
    return text, p, usage


def call_routing_llm(messages: list[dict]) -> list[str]:
    """Call the dedicated routing model and parse a JSON {"tables": [...]} response.

    Always uses Groq + ROUTING_MODEL (independent of the user-facing toggle).
    Returns [] on any failure — caller is expected to fall back to a heuristic.
    """
    import json
    if not _groq_client:
        log.warning("Routing LLM unavailable: GROQ_API_KEY missing")
        return []
    try:
        completion = _groq_client.chat.completions.create(
            model=ROUTING_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        raw = (completion.choices[0].message.content or "").strip()
        u = completion.usage
        log.info(
            "token_usage phase=routing provider=groq model=%s input=%d output=%d total=%d",
            ROUTING_MODEL,
            u.prompt_tokens if u else 0,
            u.completion_tokens if u else 0,
            u.total_tokens if u else 0,
        )
    except Exception as e:
        log.warning("Routing LLM call failed: %s", e)
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Routing LLM returned non-JSON: %r", raw[:200])
        return []

    tables = data.get("tables") if isinstance(data, dict) else None
    if not isinstance(tables, list):
        log.warning("Routing LLM JSON missing 'tables' list: %r", data)
        return []
    return [str(t).strip() for t in tables if isinstance(t, (str,)) and t.strip()]
