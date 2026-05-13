#!/usr/bin/env python3
"""
One-time script to draft schema_metadata.json using Gemini.

Usage:
    python bootstrap_metadata.py

Re-running is safe (idempotent):
  - Skips tables already in schema_metadata.json.
  - Never overwrites entries with manually_edited: true.
  - Saves progress after each table so you can Ctrl-C and resume.
  - If Gemini quota is exhausted mid-run, progress is preserved.
"""
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai
from sqlalchemy import text

from config import GEMINI_API_KEY, DATABASE_URL, DB_SCHEMA, engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger("bootstrap")

METADATA_PATH = Path("./schema_metadata.json")

_SYSTEM_PROMPT = (
    "This is an EPMS palm oil plantation management database. "
    "Tables prefixed m_ are master/reference tables; "
    "t_ are transactional; "
    "tr_ are transactional headers/details; "
    "ZEPMS_ are SAP integration tables."
)

_MIN_DELAY_SEC = 4.5   # 15 req/min free-tier → ~4s gap between calls


def _fetch_all_tables(schema: str) -> list[str]:
    sql = text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = :schema
          AND table_type IN ('BASE TABLE', 'VIEW')
        ORDER BY table_name
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"schema": schema}).fetchall()
    return [r.table_name for r in rows]


def _fetch_columns(table_name: str, schema: str) -> list[dict]:
    sql = text("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        ORDER BY ordinal_position
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"schema": schema, "table": table_name}).fetchall()
    return [{"name": r.column_name, "type": r.data_type, "nullable": r.is_nullable} for r in rows]


def _call_gemini(model: Any, table_name: str, columns: list[dict]) -> dict:
    col_lines = "\n".join(f"  - {c['name']} ({c['type']})" for c in columns)
    prompt = (
        f"Table: {table_name}\n"
        f"Columns:\n{col_lines}\n\n"
        'Return STRICT JSON (no extra text, no markdown fences) with exactly these keys:\n'
        '{\n'
        '  "description": "one-sentence business purpose of this table",\n'
        '  "synonyms": ["business term", "abbreviation"],\n'
        '  "example_questions": ["natural language question 1", "natural language question 2"]\n'
        '}'
    )

    backoff = 4
    for attempt in range(6):
        try:
            resp = model.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 500},
            )
            raw = (resp.text or "").strip()
            # Strip markdown fences if Gemini wraps output anyway
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
                raw = raw.rsplit("```", 1)[0].strip()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            log.warning("JSON parse error for %s (attempt %d): %s", table_name, attempt + 1, e)
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                wait = backoff * (2 ** attempt)
                log.warning("Rate limit hit — waiting %ds before retry %d", wait, attempt + 2)
                time.sleep(wait)
                continue
            log.warning("Gemini error for %s: %s", table_name, e)
            break

    log.warning("Returning empty metadata for %s after all retries", table_name)
    return {"description": f"Table {table_name}", "synonyms": [], "example_questions": []}


def main() -> None:
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set in .env — cannot run bootstrap")
        sys.exit(1)

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=_SYSTEM_PROMPT,
    )

    tables = _fetch_all_tables(DB_SCHEMA)
    total  = len(tables)
    log.info("Found %d tables in schema '%s'", total, DB_SCHEMA)

    # Load existing metadata (if any) so we can resume
    existing: dict = {}
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
            log.info("Loaded %d existing entries from schema_metadata.json", len(existing))
        except Exception as e:
            log.warning("Could not read existing schema_metadata.json: %s — starting fresh", e)

    result         = dict(existing)
    generated_count = 0
    skipped_count   = 0

    for idx, table_name in enumerate(tables, 1):
        entry = existing.get(table_name)

        if entry is not None:
            if entry.get("manually_edited"):
                log.info("[%d/%d] Skipping manually-edited entry: %s", idx, total, table_name)
            else:
                log.info("[%d/%d] Already exists, skipping: %s", idx, total, table_name)
            skipped_count += 1
            continue

        columns = _fetch_columns(table_name, DB_SCHEMA)
        log.info("[%d/%d] Generating metadata for %s (%d cols)...", idx, total, table_name, len(columns))

        metadata = _call_gemini(model, table_name, columns)
        result[table_name] = metadata
        generated_count += 1

        # Persist after every table so Ctrl-C or quota exhaustion doesn't lose progress
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if idx < total:
            time.sleep(_MIN_DELAY_SEC)

    log.info(
        "Done. Generated: %d, Skipped: %d, Total entries: %d",
        generated_count, skipped_count, len(result),
    )

    db_set   = set(tables)
    meta_set = set(result.keys())

    uncovered = db_set - meta_set
    if uncovered:
        log.warning("Tables NOT yet in schema_metadata.json: %s", sorted(uncovered))

    orphaned = meta_set - db_set
    if orphaned:
        log.warning("Orphaned entries (no matching table in EPMS): %s", sorted(orphaned))

    print(f"\nschema_metadata.json written with {len(result)} entries.")


if __name__ == "__main__":
    main()
