import json
import logging
import re
from pathlib import Path

from metadata_loader import TableCard, compact_ddl_for

log = logging.getLogger(__name__)

_RULES_PATH = Path("./business_rules.txt")

_SQL_SYSTEM_PREAMBLE = (
    "PostgreSQL expert for EPMS palm-oil plantation. "
    "Convert the question to ONE SELECT statement. "
    "Output ONLY SQL — no commentary, no markdown fences. "
    "Use ONLY tables/columns in the DDL below. "
    "If a needed column is missing, pick the closest one. "
    "IMPORTANT: All text/string values stored in the database are in ALL CAPS. "
    "Always use UPPERCASE string literals in WHERE, CASE WHEN, and IN clauses "
    "(e.g. 'MALE' not 'Male', 'ACTIVE' not 'Active')."
)

INTERPRET_SYSTEM_PROMPT = (
    "Rangkum hasil SQL dalam Bahasa Indonesia yang mudah dipahami untuk manajer perkebunan non-teknis. "
    "≤3 kalimat. Sertakan angka dan satuan yang spesifik. "
    "Jika tidak ada data, sampaikan dengan jelas. Jangan ulangi pernyataan SQL."
)

REASONING_SYSTEM_PROMPT = (
    "Anda adalah analis data senior untuk perkebunan kelapa sawit EPMS. "
    "Berdasarkan pertanyaan pengguna, SQL yang dijalankan, dan data yang dikembalikan, buat laporan analitis terstruktur dalam Bahasa Indonesia:\n\n"
    "RINGKASAN: 1 kalimat yang menyatakan apa yang dikueri.\n"
    "TEMUAN UTAMA: 3–5 poin dengan angka, satuan, dan pola yang relevan.\n"
    "WAWASAN: 2–3 poin yang menginterpretasikan apa arti data bagi operasional perkebunan.\n"
    "REKOMENDASI: 1–3 langkah tindak lanjut yang dapat diambil. Hilangkan bagian ini jika data tidak mendukung.\n\n"
    "Jadilah spesifik. Sebutkan angka dan satuan. Hindari pernyataan yang ambigu. Jika data tidak mencukupi, sampaikan dengan jelas."
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

    parts: list[str] = [f"Schema:\n{schema_block}"]

    if examples:
        ex_lines = []
        for past_q, past_sql in examples:
            ex_lines.append(f"Q: {past_q}\nSQL: {past_sql}")
        ex_block = "\n\n".join(ex_lines)
        parts.append(f"Examples:\n{ex_block}")
        tokens_ex = _estimate_tokens(ex_block)
    else:
        tokens_ex = 0

    if prior_error and prior_sql:
        parts.append(
            f"Prior SQL failed: {prior_error}\n"
            f"SQL: {prior_sql}\n"
            "Return fixed SQL only."
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
    rows: list[dict],
    columns: list[str],
    stats: dict | None = None,
) -> list[dict]:
    if stats:
        stats_json  = json.dumps(stats,      default=str, ensure_ascii=False)
        sample_json = json.dumps(rows[:5],   default=str, ensure_ascii=False)
        data_section = (
            f"Summary stats (all {len(rows)} rows, computed deterministically):\n{stats_json}\n\n"
            f"Sample rows (first 5 of {len(rows)}):\n{sample_json}"
        )
    else:
        preview_json = json.dumps(rows[:20], default=str, ensure_ascii=False)
        data_section = f"Rows (first 20): {preview_json}"

    return [
        {"role": "system", "content": INTERPRET_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Q: {question}\n"
                f"Columns: {columns}\n"
                f"{data_section}\n"
                f"Total rows: {len(rows)}"
            ),
        },
    ]


def build_reasoning_messages(
    question: str,
    sql: str,
    rows: list[dict],
    columns: list[str],
    stats: dict,
) -> list[dict]:
    preview = rows[:50]
    preview_json = json.dumps(preview, default=str, ensure_ascii=False)
    stats_json   = json.dumps(stats,   default=str, ensure_ascii=False)
    return [
        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"SQL executed:\n{sql}\n\n"
                f"Columns: {columns}\n"
                f"Total rows returned: {len(rows)}\n"
                f"Summary stats (computed deterministically): {stats_json}\n\n"
                f"Rows (first 50): {preview_json}\n\n"
                "Produce the structured report as instructed."
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
