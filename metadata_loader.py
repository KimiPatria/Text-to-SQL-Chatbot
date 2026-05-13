import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from sqlalchemy import text

from config import engine, DB_SCHEMA

log = logging.getLogger(__name__)

METADATA_PATH = Path("./schema_metadata.json")

_AUDIT_RE = re.compile(
    r"^(created|updated|modified|deleted)_(by|at|date|time)$",
    re.IGNORECASE,
)

_DATE_COL_RE = re.compile(r"(date|time)", re.IGNORECASE)

_TYPE_MAP: dict[str, str] = {
    "character varying": "varchar",
    "character": "varchar",
    "text": "varchar",
    "integer": "int",
    "smallint": "int",
    "bigint": "bigint",
    "numeric": "dec",
    "decimal": "dec",
    "double precision": "dec",
    "real": "dec",
    "boolean": "bool",
    "date": "date",
    "timestamp without time zone": "ts",
    "timestamp with time zone": "ts",
}


@dataclass
class ColumnSpec:
    name: str
    data_type: str
    is_nullable: bool
    is_pk: bool
    fk_ref: tuple[str, str] | None = None


@dataclass
class TableCard:
    name: str
    description: str
    synonyms: list[str]
    example_questions: list[str]
    columns: list[ColumnSpec] = field(default_factory=list)


def embedding_text(card: TableCard) -> str:
    col_names = ", ".join(c.name for c in card.columns[:15])
    syns = ", ".join(card.synonyms) if card.synonyms else ""
    examples = " | ".join(card.example_questions) if card.example_questions else ""
    parts = [f"{card.name} — {card.description}"]
    if col_names:
        parts.append(f"Columns: {col_names}")
    if syns:
        parts.append(f"Synonyms: {syns}")
    if examples:
        parts.append(f"Examples: {examples}")
    return "\n".join(parts)


def _type_alias(data_type: str) -> str:
    return _TYPE_MAP.get(data_type.lower(), data_type[:10])


def _is_varchar_date(col_name: str, data_type: str) -> bool:
    if data_type.lower() in ("character varying", "text", "character"):
        return bool(_DATE_COL_RE.search(col_name))
    return False


def compact_ddl_for(table_names: list[str]) -> str:
    """Return compact one-line-per-table DDL, tables sorted alphabetically."""
    lines = []
    for name in sorted(table_names):
        card = _CARDS_CACHE.get(name)
        if card is None:
            continue
        col_parts = []
        for col in card.columns:
            is_audit = bool(_AUDIT_RE.match(col.name))
            if is_audit and not col.is_pk and col.fk_ref is None:
                continue
            if col.is_pk:
                col_parts.append(f"{col.name} PK")
            elif col.fk_ref:
                ref_table, ref_col = col.fk_ref
                col_parts.append(f"{col.name}→{ref_table}.{ref_col}")
            else:
                alias = _type_alias(col.data_type)
                date_hint = "[date]" if _is_varchar_date(col.name, col.data_type) else ""
                col_parts.append(f"{col.name} {alias}{date_hint}")
        lines.append(f"{name}({', '.join(col_parts)})")
    return "\n".join(lines)


# Module-level cache — populated once at server startup by load_all_cards()
_CARDS_CACHE: dict[str, TableCard] = {}
_CACHE_LOCK = Lock()


def _load_metadata_json() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        print(
            "ERROR: schema_metadata.json not found. "
            "Run 'python bootstrap_metadata.py' first to generate schema_metadata.json.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _fetch_db_schema(schema: str) -> tuple[
    dict[str, list[dict]],
    dict[str, set[str]],
    dict[str, dict[str, tuple[str, str]]],
]:
    """Fetch columns, PKs, and FKs from information_schema. Returns three dicts."""
    col_sql = text("""
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = :schema
        ORDER BY table_name, ordinal_position
    """)
    pk_sql = text("""
        SELECT kcu.table_name, kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        WHERE tc.constraint_type = 'PRIMARY KEY'
          AND tc.table_schema = :schema
    """)
    fk_sql = text("""
        SELECT
            kcu.table_name,
            kcu.column_name,
            ccu.table_name  AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
          ON ccu.constraint_name = tc.constraint_name
         AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = :schema
    """)

    with engine.connect() as conn:
        col_rows = conn.execute(col_sql, {"schema": schema}).fetchall()
        pk_rows  = conn.execute(pk_sql,  {"schema": schema}).fetchall()
        fk_rows  = conn.execute(fk_sql,  {"schema": schema}).fetchall()

    cols_by_table: dict[str, list[dict]] = {}
    for r in col_rows:
        cols_by_table.setdefault(r.table_name, []).append({
            "name": r.column_name,
            "data_type": r.data_type,
            "is_nullable": r.is_nullable == "YES",
        })

    pks_by_table: dict[str, set[str]] = {}
    for r in pk_rows:
        pks_by_table.setdefault(r.table_name, set()).add(r.column_name)

    fks_by_table: dict[str, dict[str, tuple[str, str]]] = {}
    for r in fk_rows:
        fks_by_table.setdefault(r.table_name, {})[r.column_name] = (
            r.foreign_table_name,
            r.foreign_column_name,
        )

    return cols_by_table, pks_by_table, fks_by_table


def load_all_cards(schema: str = DB_SCHEMA) -> dict[str, TableCard]:
    """Merge schema_metadata.json with live information_schema data. Exits if JSON missing."""
    global _CARDS_CACHE

    metadata = _load_metadata_json()
    cols_by_table, pks_by_table, fks_by_table = _fetch_db_schema(schema)

    db_tables   = set(cols_by_table.keys())
    meta_tables = set(metadata.keys())

    new_tables = db_tables - meta_tables
    if new_tables:
        log.warning(
            "Tables in EPMS not in schema_metadata.json: %s — "
            "Run bootstrap_metadata.py to add them.",
            sorted(new_tables),
        )

    orphaned = meta_tables - db_tables
    if orphaned:
        log.warning(
            "Orphaned entries in schema_metadata.json (no matching table): %s",
            sorted(orphaned),
        )

    cards: dict[str, TableCard] = {}
    for table_name, db_cols in cols_by_table.items():
        meta = metadata.get(table_name, {})
        pk_set = pks_by_table.get(table_name, set())
        fk_map = fks_by_table.get(table_name, {})

        columns = [
            ColumnSpec(
                name=c["name"],
                data_type=c["data_type"],
                is_nullable=c["is_nullable"],
                is_pk=c["name"] in pk_set,
                fk_ref=fk_map.get(c["name"]),
            )
            for c in db_cols
        ]

        cards[table_name] = TableCard(
            name=table_name,
            description=meta.get("description", f"Table {table_name}"),
            synonyms=meta.get("synonyms", []),
            example_questions=meta.get("example_questions", []),
            columns=columns,
        )

    with _CACHE_LOCK:
        _CARDS_CACHE = cards

    log.info("Loaded %d table cards (schema_metadata.json + information_schema)", len(cards))
    return cards


def get_card(name: str) -> TableCard | None:
    return _CARDS_CACHE.get(name)


def reload_cards(schema: str = DB_SCHEMA) -> dict[str, TableCard]:
    """Force-reload from DB and JSON. Called by /admin/reload-schema."""
    log.info("/admin/reload-schema triggered — reloading table cards")
    return load_all_cards(schema)
