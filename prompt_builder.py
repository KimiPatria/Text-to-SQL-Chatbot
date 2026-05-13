import json
import logging
import re
from pathlib import Path

from metadata_loader import TableCard, compact_ddl_for

log = logging.getLogger(__name__)

_RULES_PATH = Path("./business_rules.txt")

_SQL_SYSTEM_PREAMBLE = (
    "You are a PostgreSQL expert for the EPMS palm-oil plantation system. "
    "Your job: convert the user's question into ONE PostgreSQL SELECT statement. "
    "Output ONLY the SQL — no commentary, no explanation, no markdown fences. "
    "Use ONLY the tables and columns present in the provided DDL. "
    "If a needed column does not exist in the DDL, choose the closest one or return "
    "a SELECT that the user can refine."
)

INTERPRET_SYSTEM_PROMPT = (
    "You summarize SQL query results in plain English for a non-technical plantation manager. "
    "Maximum 3 sentences. Be specific with numbers and units. "
    "If the result set is empty, say so plainly. Do not restate the SQL."
)

# Loaded once at import; stable across requests for prefix-cache hits.
_BUSINESS_RULES: str = ""


def _load_business_rules() -> str:
    if not _RULES_PATH.exists():
        log.warning("business_rules.txt not found at %s — using empty rules", _RULES_PATH)
        return ""
    try:
        return _RULES_PATH.read_text(encoding="utf-8").strip()
    except Exception as e:
        log.exception("Failed to load business_rules.txt: %s", e)
        return ""


def reload_business_rules() -> str:
    """Re-read business_rules.txt from disk. Returns the new content."""
    global _BUSINESS_RULES
    _BUSINESS_RULES = _load_business_rules()
    return _BUSINESS_RULES


# Load at import time
_BUSINESS_RULES = _load_business_rules()


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def build_sql_messages(
    question: str,
    table_cards: list[TableCard],
    examples: list[tuple[str, str]],
    prior_error: str | None = None,
    prior_sql: str | None = None,
) -> list[dict]:
    """
    Build chat messages for the SQL writer with cache-stable ordering:
      [SYSTEM: preamble + business rules]   — fixed every request
      [USER: schema + examples + question]  — schema is alphabetized for stability
    """
    # System message — completely fixed; maximises prefix-cache reuse
    system_content = _SQL_SYSTEM_PREAMBLE
    if _BUSINESS_RULES:
        system_content = f"{system_content}\n\n{_BUSINESS_RULES}"

    # User message parts — ordered for cache stability (stable → variable)
    table_names = [card.name for card in table_cards]
    schema_block = compact_ddl_for(table_names)

    # Token budget logging
    tokens_biz    = _estimate_tokens(_BUSINESS_RULES)
    tokens_schema = _estimate_tokens(schema_block)
    tokens_q      = _estimate_tokens(question)

    parts: list[str] = [f"DATABASE DDL (only these tables/columns exist):\n{schema_block}"]

    if examples:
        ex_lines = []
        for past_q, past_sql in examples:
            ex_lines.append(f"Q: {past_q}\nSQL: {past_sql}")
        ex_block = "\n\n".join(ex_lines)
        parts.append(f"EXAMPLES (past successful queries):\n{ex_block}")
        tokens_ex = _estimate_tokens(ex_block)
    else:
        tokens_ex = 0

    if prior_error and prior_sql:
        parts.append(
            f"The previous attempt failed with: {prior_error}\n"
            f"Previous SQL: {prior_sql}\n"
            "Fix the SQL and respond with the corrected SQL only."
        )

    parts.append(f"Q: {question}\nSQL:")

    user_content = "\n\n".join(parts)

    tokens_total = _estimate_tokens(system_content) + _estimate_tokens(user_content)
    log.info(
        "[prompt] tokens ≈ %d (business=%d, schema=%d, examples=%d, question=%d)",
        tokens_total, tokens_biz, tokens_schema, tokens_ex, tokens_q,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]


def build_interpretation_messages(
    question: str,
    sql: str,
    rows: list[dict],
    columns: list[str],
) -> list[dict]:
    preview = rows[:20]
    preview_json = json.dumps(preview, default=str, ensure_ascii=False)
    return [
        {"role": "system", "content": INTERPRET_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"SQL executed: {sql}\n\n"
                f"Columns: {columns}\n"
                f"Rows (first 20): {preview_json}\n"
                f"Total rows returned: {len(rows)}\n\n"
                "Answer in <= 3 sentences."
            ),
        },
    ]


_FENCE_RE  = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_SELECT_RE = re.compile(r"\b(WITH|SELECT)\b.*",    re.IGNORECASE | re.DOTALL)


def extract_sql(text: str) -> str:
    if not text:
        return ""
    fences    = _FENCE_RE.findall(text)
    candidate = max(fences, key=len).strip() if fences else text.strip()
    m = _SELECT_RE.search(candidate)
    if m:
        candidate = m.group(0)
    return candidate.strip().rstrip(";").strip()


# Keep old name as an alias so any future callers still work
reload_glossary = reload_business_rules
