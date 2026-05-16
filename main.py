import logging
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from config import MAX_RESULT_ROWS, LARGE_RESULT_THRESHOLD, engine
import metadata_loader
from metadata_loader import load_all_cards, reload_cards
from retrieval import initialize as init_retrieval, retrieve_tables, retrieve_examples, refresh_query_history_index
from query_history import init_db, record_query
from sql_validator import validate, ensure_limit
from prompt_builder import (
    build_sql_messages,
    build_interpretation_messages,
    build_reasoning_messages,
    extract_sql,
    reload_business_rules,
)
from llm import call_llm, get_provider, set_provider, list_providers, PROVIDER_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("epms-chatbot")

app = FastAPI(title="EPMS Text-to-SQL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── startup ────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup() -> None:
    init_db()
    cards = load_all_cards()   # exits with clear message if schema_metadata.json missing
    init_retrieval(cards)


# ── request / response models ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    mode: str = "fast"   # "fast" | "reasoning"


class ChatResponse(BaseModel):
    answer: str | None = None
    sql: str | None = None
    rows: list[dict] | None = None
    columns: list[str] | None = None
    tables_considered: list[str] | None = None  # kept for frontend compat
    tables_used: list[str] | None = None
    provider: str | None = None
    provider_label: str | None = None
    mode: str | None = None
    error: str | None = None


class SetProviderRequest(BaseModel):
    provider: str


class SetRowLimitRequest(BaseModel):
    limit: int


_current_row_limit: int = MAX_RESULT_ROWS


# ── helpers ────────────────────────────────────────────────────────────────

def _execute(sql: str) -> tuple[list[str], list[dict]]:
    limited = ensure_limit(sql, _current_row_limit)
    with engine.connect() as conn:
        result  = conn.execute(text(limited))
        columns = list(result.keys())
        rows    = [dict(r._mapping) for r in result.fetchall()]
    return columns, rows


def _is_trivial_result(rows: list[dict], columns: list[str]) -> bool:
    return (
        len(rows) == 0
        or (len(rows) == 1 and len(columns) == 1)
        or len(rows) <= 3
    )


def _format_deterministic(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "No matching records found."
    if len(rows) == 1 and len(columns) == 1:
        val = rows[0].get(columns[0], "")
        return f"{columns[0]}: {val}"
    # ≤3 rows — simple markdown table
    header    = "| " + " | ".join(str(c) for c in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    data_rows = [
        "| " + " | ".join(str(row.get(c, "")) for c in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator] + data_rows)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _compute_summary_stats(rows: list[dict], columns: list[str]) -> dict:
    """Pure-Python summary stats so the LLM gets ground-truth aggregates without spending tokens on math.

    For numeric columns: {min, max, mean, n_distinct}
    For text/other columns: {n_distinct, top_5}
    """
    out: dict[str, dict] = {}
    if not rows:
        return out

    for col in columns:
        values     = [r.get(col) for r in rows]
        non_null   = [v for v in values if v is not None]
        if not non_null:
            out[col] = {"n_null": len(values), "n_distinct": 0}
            continue

        if all(_is_number(v) for v in non_null):
            mn = min(non_null)
            mx = max(non_null)
            mean = sum(non_null) / len(non_null)
            out[col] = {
                "type":       "numeric",
                "min":        mn,
                "max":        mx,
                "mean":       round(mean, 4) if isinstance(mean, float) else mean,
                "n_distinct": len(set(non_null)),
                "n_null":     len(values) - len(non_null),
            }
        else:
            counts: dict = {}
            for v in non_null:
                key = str(v)
                counts[key] = counts.get(key, 0) + 1
            top5 = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
            out[col] = {
                "type":       "categorical",
                "n_distinct": len(counts),
                "top_5":      [{"value": k, "count": c} for k, c in top5],
                "n_null":     len(values) - len(non_null),
            }
    return out


# ── routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "provider": get_provider(), "providers": list_providers()}


@app.get("/get-provider")
def get_provider_endpoint():
    return {
        "provider": get_provider(),
        "label": PROVIDER_LABELS[get_provider()],
        "providers": list_providers(),
    }


@app.post("/set-provider")
def set_provider_endpoint(req: SetProviderRequest):
    ok, err = set_provider(req.provider)
    if not ok:
        return JSONResponse(status_code=400, content={"ok": False, "error": err})
    return {
        "ok": True,
        "provider": get_provider(),
        "label": PROVIDER_LABELS[get_provider()],
    }


@app.get("/get-row-limit")
def get_row_limit_endpoint():
    return {"limit": _current_row_limit}


@app.post("/set-row-limit")
def set_row_limit_endpoint(req: SetRowLimitRequest):
    global _current_row_limit
    if req.limit not in (100, 5000):
        return JSONResponse(status_code=400, content={"ok": False, "error": "limit must be 100 or 5000"})
    _current_row_limit = req.limit
    return {"ok": True, "limit": _current_row_limit}


@app.get("/admin/catalog")
def admin_catalog():
    cards = metadata_loader._CARDS_CACHE
    sample = sorted(cards.keys())[:25]
    return {
        "table_count": len(cards),
        "sample": sample,
    }


@app.post("/admin/reload-schema")
def admin_reload_schema():
    log.info("/admin/reload-schema called")
    try:
        cards = reload_cards()
        init_retrieval(cards)
        return {"ok": True, "table_count": len(cards)}
    except Exception as e:
        log.exception("reload-schema failed")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.post("/admin/reload-glossary")
def admin_reload_glossary():
    rules = reload_business_rules()
    return {"ok": True, "chars": len(rules)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    question = (req.question or "").strip()
    mode     = (req.mode or "fast").lower()
    if mode not in ("fast", "reasoning"):
        mode = "fast"
    active   = get_provider()
    label    = PROVIDER_LABELS.get(active, active)

    if not question:
        return ChatResponse(error="Please enter a question.", provider=active, provider_label=label, mode=mode)

    t_start = time.monotonic()
    log.info('[chat] mode=%s Q: "%s"', mode, question)

    # ── Step 1: Retrieval (no LLM) ────────────────────────────────────────
    tables   = retrieve_tables(question, k=5)
    examples = retrieve_examples(question, k=3)

    if not tables:
        log.warning("[retrieval] returned 0 tables — this should not happen")
        return ChatResponse(
            error="Retrieval returned no tables. Check that indexes are built.",
            provider=active, provider_label=label,
        )

    table_names = [t.name for t in tables]

    # ── Step 2: SQL generation ────────────────────────────────────────────
    try:
        messages = build_sql_messages(question, tables, examples)
        raw, used, sql_usage = call_llm(messages, temperature=0, max_tokens=800)
        log.info(
            "[llm] provider=%s latency=%.0fms input=%d output=%d cached_tokens=%s",
            used,
            (time.monotonic() - t_start) * 1000,
            sql_usage["input_tokens"],
            sql_usage["output_tokens"],
            sql_usage.get("cached_tokens", 0),
        )
    except Exception as e:
        log.exception("LLM SQL generation failed")
        return ChatResponse(
            error=f"LLM call failed: {e.__class__.__name__}: {e}",
            tables_considered=table_names, tables_used=table_names,
            provider=active, provider_label=label,
        )

    sql = extract_sql(raw)
    if not sql:
        return ChatResponse(
            error="The model did not return a SQL query.",
            tables_considered=table_names, tables_used=table_names,
            provider=used, provider_label=PROVIDER_LABELS.get(used, used),
        )

    ok, reason = validate(sql)
    if not ok:
        record_query(question, sql, used, table_names, success=False, error=f"unsafe SQL: {reason}")
        return ChatResponse(
            error=f"Generated SQL failed safety validation: {reason}",
            sql=sql,
            tables_considered=table_names, tables_used=table_names,
            provider=used, provider_label=PROVIDER_LABELS.get(used, used),
        )

    # ── Step 3: Execute (with one self-correction retry) ──────────────────
    try:
        columns, rows = _execute(sql)
    except SQLAlchemyError as exec_err:
        err_text = str(exec_err.orig) if hasattr(exec_err, "orig") and exec_err.orig else str(exec_err)
        log.warning("First SQL execution failed: %s", err_text)

        # Self-correction retry
        try:
            retry_msgs = build_sql_messages(question, tables, examples, prior_error=err_text, prior_sql=sql)
            raw_retry, used, retry_usage = call_llm(retry_msgs, temperature=0, max_tokens=800)
            log.info(
                "[llm] retry provider=%s input=%d output=%d",
                used, retry_usage["input_tokens"], retry_usage["output_tokens"],
            )
        except Exception as e:
            log.exception("LLM retry call failed")
            record_query(question, sql, used, table_names, success=False, error=err_text)
            return ChatResponse(
                error=f"Query failed and retry LLM call failed: {e.__class__.__name__}.",
                sql=sql,
                tables_considered=table_names, tables_used=table_names,
                provider=used, provider_label=PROVIDER_LABELS.get(used, used),
            )

        sql_retry = extract_sql(raw_retry)
        if not sql_retry:
            record_query(question, sql, used, table_names, success=False, error=err_text)
            return ChatResponse(
                error="Query failed and the model did not return a corrected SQL.",
                sql=sql,
                tables_considered=table_names, tables_used=table_names,
                provider=used, provider_label=PROVIDER_LABELS.get(used, used),
            )

        ok2, reason2 = validate(sql_retry)
        if not ok2:
            record_query(question, sql_retry, used, table_names, success=False, error=reason2)
            return ChatResponse(
                error=f"Retry SQL failed safety validation: {reason2}",
                sql=sql_retry,
                tables_considered=table_names, tables_used=table_names,
                provider=used, provider_label=PROVIDER_LABELS.get(used, used),
            )

        try:
            columns, rows = _execute(sql_retry)
            sql = sql_retry
        except SQLAlchemyError as exec_err2:
            err2 = str(exec_err2.orig) if hasattr(exec_err2, "orig") and exec_err2.orig else str(exec_err2)
            log.warning("Retry SQL execution also failed: %s", err2)
            record_query(question, sql_retry, used, table_names, success=False, error=err2)
            return ChatResponse(
                error=f"Query failed twice. Last error: {err2}",
                sql=sql_retry,
                tables_considered=table_names, tables_used=table_names,
                provider=used, provider_label=PROVIDER_LABELS.get(used, used),
            )

    log.info("[exec] rows=%d trivial=%s", len(rows), _is_trivial_result(rows, columns))

    # ── Step 4: Interpretation ────────────────────────────────────────────
    # Reasoning mode bypasses the trivial-result short-circuit (user asked for analysis).
    if mode == "reasoning":
        try:
            stats = _compute_summary_stats(rows, columns)
            reason_msgs = build_reasoning_messages(question, sql, rows, columns, stats)
            answer, used, reason_usage = call_llm(reason_msgs, temperature=0.3, max_tokens=1500)
            log.info(
                "[llm] reasoning provider=%s input=%d output=%d",
                used, reason_usage["input_tokens"], reason_usage["output_tokens"],
            )
        except Exception as e:
            log.exception("Reasoning LLM call failed")
            answer = f"Returned {len(rows)} row(s). (Reasoning failed: {e.__class__.__name__}.)"
    elif _is_trivial_result(rows, columns):
        answer = _format_deterministic(rows, columns)
    else:
        try:
            stats = _compute_summary_stats(rows, columns) if len(rows) > LARGE_RESULT_THRESHOLD else None
            interp_msgs = build_interpretation_messages(question, rows, columns, stats)
            answer, used, interp_usage = call_llm(interp_msgs, temperature=0.2, max_tokens=300)
            log.info(
                "[llm] interpret provider=%s input=%d output=%d",
                used, interp_usage["input_tokens"], interp_usage["output_tokens"],
            )
        except Exception as e:
            log.exception("Interpretation LLM call failed")
            answer = f"Returned {len(rows)} row(s). (Interpretation failed: {e.__class__.__name__}.)"

    # ── Record success and update few-shot index ───────────────────────────
    record_query(question, sql, used, table_names, success=True)
    refresh_query_history_index()

    elapsed = time.monotonic() - t_start
    log.info("[chat] done in %.2fs", elapsed)

    return ChatResponse(
        answer=answer,
        sql=sql,
        rows=rows,
        columns=columns,
        tables_considered=table_names,
        tables_used=table_names,
        provider=used,
        provider_label=PROVIDER_LABELS.get(used, used),
        mode=mode,
    )
