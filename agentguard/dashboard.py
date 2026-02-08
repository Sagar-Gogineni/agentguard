"""
AgentGuard Human Review Dashboard

Streamlit app for EU AI Act Article 14 human oversight compliance.
Provides a review interface for escalated AI interactions, audit log
browsing, CSV export, and live compliance report viewing.

Usage:
    agentguard-dashboard --audit-path ./agentguard_audit

    # Or directly:
    streamlit run agentguard/dashboard.py -- --audit-path ./agentguard_audit
"""

from __future__ import annotations

import argparse
import csv
import io
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import streamlit as st


# ------------------------------------------------------------------ #
#  Database helpers
# ------------------------------------------------------------------ #


def _get_db(audit_path: str) -> sqlite3.Connection:
    db_path = Path(audit_path) / "audit.db"
    if not db_path.exists():
        st.error(f"Audit database not found at {db_path}")
        st.stop()
    db = sqlite3.connect(str(db_path), check_same_thread=False)
    db.row_factory = sqlite3.Row
    # Ensure review_decisions table exists
    db.execute("""
        CREATE TABLE IF NOT EXISTS review_decisions (
            interaction_id TEXT PRIMARY KEY,
            decision TEXT NOT NULL,
            reason TEXT,
            decided_at TEXT NOT NULL
        )
    """)
    db.commit()
    return db


def _get_stats(db: sqlite3.Connection) -> dict:
    stats = {}
    stats["total_interactions"] = db.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    stats["total_escalations"] = db.execute(
        "SELECT COUNT(*) FROM audit_log WHERE human_escalated = 1"
    ).fetchone()[0]
    stats["avg_confidence"] = db.execute(
        "SELECT AVG(confidence_score) FROM audit_log WHERE confidence_score IS NOT NULL"
    ).fetchone()[0]
    stats["avg_latency_ms"] = db.execute("SELECT AVG(latency_ms) FROM audit_log").fetchone()[0]
    stats["inputs_blocked"] = db.execute(
        "SELECT COUNT(*) FROM audit_log WHERE input_policy_blocked = 1"
    ).fetchone()[0]
    stats["outputs_blocked"] = db.execute(
        "SELECT COUNT(*) FROM audit_log WHERE output_policy_blocked = 1"
    ).fetchone()[0]
    stats["disclaimers_added"] = db.execute(
        "SELECT COUNT(*) FROM audit_log WHERE output_disclaimer_added = 1"
    ).fetchone()[0]
    return stats


def _get_category_breakdown(db: sqlite3.Connection) -> dict[str, int]:
    """Count how often each flagged category appears across all interactions."""
    rows = db.execute(
        "SELECT input_policy_flagged FROM audit_log "
        "WHERE input_policy_flagged IS NOT NULL AND input_policy_flagged != '[]'"
    ).fetchall()
    counts: dict[str, int] = {}
    for r in rows:
        try:
            import json as _json

            cats = _json.loads(r[0])
            for cat in cats:
                counts[cat] = counts.get(cat, 0) + 1
        except (ValueError, TypeError):
            pass
    # Also count output-flagged categories
    rows = db.execute(
        "SELECT output_policy_flagged FROM audit_log "
        "WHERE output_policy_flagged IS NOT NULL AND output_policy_flagged != '[]'"
    ).fetchall()
    for r in rows:
        try:
            import json as _json

            cats = _json.loads(r[0])
            for cat in cats:
                counts[cat] = counts.get(cat, 0) + 1
        except (ValueError, TypeError):
            pass
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def _get_distinct_categories(db: sqlite3.Connection) -> list[str]:
    """Get all distinct content categories that have been flagged."""
    rows = db.execute(
        "SELECT input_policy_flagged FROM audit_log "
        "WHERE input_policy_flagged IS NOT NULL AND input_policy_flagged != '[]'"
    ).fetchall()
    cats: set[str] = set()
    for r in rows:
        try:
            import json as _json

            for cat in _json.loads(r[0]):
                cats.add(cat)
        except (ValueError, TypeError):
            pass
    return sorted(cats)


def _query_escalated(
    db: sqlite3.Connection,
    start_date: str | None = None,
    end_date: str | None = None,
    user_id: str | None = None,
    reason_filter: str | None = None,
    category_filter: str | None = None,
) -> list[dict]:
    conditions = ["a.human_escalated = 1"]
    params: list = []

    if start_date:
        conditions.append("a.timestamp >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("a.timestamp <= ?")
        params.append(end_date + "T23:59:59")
    if user_id:
        conditions.append("a.user_id = ?")
        params.append(user_id)
    if reason_filter and reason_filter != "All":
        conditions.append("a.escalation_reason = ?")
        params.append(reason_filter)
    if category_filter and category_filter != "All":
        conditions.append("a.input_policy_flagged LIKE ?")
        params.append(f'%"{category_filter}"%')

    where = "WHERE " + " AND ".join(conditions)
    query = f"""
        SELECT a.*, d.decision, d.reason AS review_reason, d.decided_at
        FROM audit_log a
        LEFT JOIN review_decisions d ON a.interaction_id = d.interaction_id
        {where}
        ORDER BY a.timestamp DESC
        LIMIT 500
    """
    rows = db.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def _query_audit_log(
    db: sqlite3.Connection,
    start_date: str | None = None,
    end_date: str | None = None,
    user_id: str | None = None,
    escalated_only: bool = False,
    category_filter: str | None = None,
) -> list[dict]:
    conditions: list[str] = []
    params: list = []

    if start_date:
        conditions.append("timestamp >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("timestamp <= ?")
        params.append(end_date + "T23:59:59")
    if user_id:
        conditions.append("user_id = ?")
        params.append(user_id)
    if escalated_only:
        conditions.append("human_escalated = 1")
    if category_filter and category_filter != "All":
        conditions.append("(input_policy_flagged LIKE ? OR output_policy_flagged LIKE ?)")
        params.append(f'%"{category_filter}"%')
        params.append(f'%"{category_filter}"%')

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    query = f"SELECT * FROM audit_log {where} ORDER BY timestamp DESC LIMIT 1000"
    rows = db.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def _get_distinct_reasons(db: sqlite3.Connection) -> list[str]:
    rows = db.execute(
        "SELECT DISTINCT escalation_reason FROM audit_log "
        "WHERE escalation_reason IS NOT NULL ORDER BY escalation_reason"
    ).fetchall()
    return [r[0] for r in rows]


def _record_decision(
    db: sqlite3.Connection, interaction_id: str, decision: str, reason: str = ""
) -> None:
    db.execute(
        "INSERT OR REPLACE INTO review_decisions (interaction_id, decision, reason, decided_at) "
        "VALUES (?, ?, ?, ?)",
        (interaction_id, decision, reason, datetime.now(timezone.utc).isoformat()),
    )
    db.commit()


def _parse_categories(raw: str | None) -> list[str]:
    """Parse a JSON-encoded category list from the DB into a Python list."""
    if not raw or raw == "[]":
        return []
    try:
        import json as _json

        return _json.loads(raw)
    except (ValueError, TypeError):
        return []


def _truncate(text: str | None, length: int = 100) -> str:
    if not text:
        return ""
    return text[:length] + "..." if len(text) > length else text


def _rows_to_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


# ------------------------------------------------------------------ #
#  Compliance report helper
# ------------------------------------------------------------------ #


def _generate_report(db: sqlite3.Connection, audit_path: str) -> str:
    from agentguard.config import AgentGuardConfig, AuditBackend
    from agentguard.audit import AuditLogger
    from agentguard.report import ComplianceReporter

    # Read system_name and provider_name from the most recent log entry
    row = db.execute(
        "SELECT system_name, provider_name FROM audit_log ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()
    if not row:
        return "*No audit data available to generate a report.*"

    config = AgentGuardConfig(
        system_name=row["system_name"],
        provider_name=row["provider_name"],
        audit_backend=AuditBackend.SQLITE,
        audit_path=audit_path,
    )
    logger = AuditLogger(backend=AuditBackend.SQLITE, path=Path(audit_path))
    reporter = ComplianceReporter(config, logger)
    md = reporter.generate_markdown()
    logger.close()
    return md


# ------------------------------------------------------------------ #
#  Streamlit app
# ------------------------------------------------------------------ #


def _run_app() -> None:
    st.set_page_config(page_title="AgentGuard Dashboard", page_icon="🛡️", layout="wide")
    st.title("🛡️ AgentGuard Human Review Dashboard")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")
        audit_path = st.text_input("Audit path", value="./agentguard_audit")

        st.header("Filters")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From", value=None)
        with col2:
            end_date = st.date_input("To", value=None)
        user_id = st.text_input("User ID", placeholder="Filter by user...")

        page = st.radio(
            "Navigate",
            ["Review Queue", "Audit Log", "Compliance Report"],
            index=0,
        )

    # --- Connect to DB (early, so sidebar can query categories) ---
    db = _get_db(audit_path)

    with st.sidebar:
        all_categories = _get_distinct_categories(db)
        category_filter = st.selectbox(
            "Content Category",
            ["All"] + all_categories,
            index=0,
        ) if all_categories else "All"

    start_str = start_date.isoformat() if isinstance(start_date, date) else None
    end_str = end_date.isoformat() if isinstance(end_date, date) else None
    user_filter = user_id.strip() if user_id else None

    # --- Page: Review Queue ---
    if page == "Review Queue":
        # Stats bar
        stats = _get_stats(db)
        total = stats["total_interactions"]
        escalations = stats["total_escalations"]
        rate = (escalations / total * 100) if total > 0 else 0
        avg_conf = stats["avg_confidence"]
        avg_lat = stats["avg_latency_ms"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Interactions", f"{total:,}")
        c2.metric("Escalation Rate", f"{rate:.1f}%")
        c3.metric("Avg Confidence", f"{avg_conf:.2f}" if avg_conf else "N/A")
        c4.metric("Avg Latency", f"{avg_lat:.0f} ms" if avg_lat else "N/A")

        # Policy enforcement stats
        p1, p2, p3 = st.columns(3)
        p1.metric("Inputs Blocked", f"{stats.get('inputs_blocked', 0):,}")
        p2.metric("Outputs Blocked", f"{stats.get('outputs_blocked', 0):,}")
        p3.metric("Disclaimers Added", f"{stats.get('disclaimers_added', 0):,}")

        # Category breakdown
        cat_counts = _get_category_breakdown(db)
        if cat_counts:
            with st.expander("Category Breakdown"):
                for cat, count in cat_counts.items():
                    st.write(f"**{cat}**: {count}")

        st.divider()

        # Reason filter (specific to review queue)
        reasons = _get_distinct_reasons(db)
        reason_filter = st.selectbox(
            "Filter by escalation reason",
            ["All"] + reasons,
            index=0,
        )

        # Query escalated interactions
        rows = _query_escalated(
            db,
            start_date=start_str,
            end_date=end_str,
            user_id=user_filter,
            reason_filter=reason_filter,
            category_filter=category_filter,
        )

        pending = [r for r in rows if r.get("decision") is None]
        decided = [r for r in rows if r.get("decision") is not None]

        st.subheader(f"Pending Review ({len(pending)})")

        if not pending:
            st.info("No pending escalations.")
        else:
            for row in pending:
                with st.container(border=True):
                    cols = st.columns([2, 1, 2, 2, 2, 1, 1, 2])
                    cols[0].caption("Timestamp")
                    cols[0].write(row["timestamp"][:19])
                    cols[1].caption("User")
                    cols[1].write(row["user_id"] or "—")
                    cols[2].caption("Input")
                    cols[2].write(_truncate(row["input_text"], 80))
                    cols[3].caption("Output")
                    cols[3].write(_truncate(row["output_text"], 80))
                    cols[4].caption("Reason")
                    cols[4].write(row["escalation_reason"] or "—")
                    cols[5].caption("Conf.")
                    conf = row["confidence_score"]
                    cols[5].write(f"{conf:.2f}" if conf is not None else "—")
                    cols[6].caption("Category")
                    cats = _parse_categories(row.get("input_policy_flagged"))
                    cols[6].write(", ".join(cats) if cats else "—")

                    btn_cols = cols[7]
                    btn_cols.caption("Action")
                    iid = row["interaction_id"]
                    if btn_cols.button("Approve", key=f"approve_{iid}"):
                        _record_decision(db, iid, "approved")
                        st.rerun()
                    if btn_cols.button("Reject", key=f"reject_{iid}"):
                        _record_decision(db, iid, "rejected")
                        st.rerun()

        if decided:
            with st.expander(f"Previously Reviewed ({len(decided)})"):
                for row in decided:
                    status = "✅" if row["decision"] == "approved" else "❌"
                    st.write(
                        f"{status} **{row['interaction_id'][:8]}...** — "
                        f"{row['decision']} at {row['decided_at'][:19]} — "
                        f"Reason: {row['escalation_reason'] or '—'}"
                    )

    # --- Page: Audit Log ---
    elif page == "Audit Log":
        st.subheader("Audit Log")
        col_a, col_b = st.columns(2)
        escalated_only = col_a.checkbox("Escalated only")

        rows = _query_audit_log(
            db,
            start_date=start_str,
            end_date=end_str,
            user_id=user_filter,
            escalated_only=escalated_only,
            category_filter=category_filter,
        )

        if not rows:
            st.info("No audit entries found for the selected filters.")
        else:
            # Build display table
            display = []
            for r in rows:
                in_cats = _parse_categories(r.get("input_policy_flagged"))
                out_cats = _parse_categories(r.get("output_policy_flagged"))
                display.append(
                    {
                        "Timestamp": r["timestamp"][:19],
                        "User": r["user_id"] or "—",
                        "Model": r["model_used"] or "—",
                        "Input": _truncate(r["input_text"], 80),
                        "Output": _truncate(r["output_text"], 80),
                        "Blocked": "Yes" if r.get("input_policy_blocked") else "",
                        "Categories": ", ".join(in_cats) if in_cats else "",
                        "Disclaimer": "Yes" if r.get("output_disclaimer_added") else "",
                        "Escalated": "Yes" if r["human_escalated"] else "",
                        "Reason": r["escalation_reason"] or "",
                        "Conf.": f"{r['confidence_score']:.2f}"
                        if r["confidence_score"]
                        else "",
                        "Latency": f"{r['latency_ms']:.0f}" if r["latency_ms"] else "",
                    }
                )
            st.dataframe(display, use_container_width=True)

            csv_data = _rows_to_csv(rows)
            st.download_button(
                "📥 Export as CSV",
                data=csv_data,
                file_name="agentguard_audit_log.csv",
                mime="text/csv",
            )

    # --- Page: Compliance Report ---
    elif page == "Compliance Report":
        st.subheader("EU AI Act Compliance Report")
        report_md = _generate_report(db, audit_path)
        st.markdown(report_md)

    db.close()


# ------------------------------------------------------------------ #
#  CLI entry point
# ------------------------------------------------------------------ #


def main() -> None:
    """Entry point for the ``agentguard-dashboard`` CLI command."""
    parser = argparse.ArgumentParser(description="AgentGuard Human Review Dashboard")
    parser.add_argument(
        "--audit-path",
        default="./agentguard_audit",
        help="Path to the AgentGuard audit directory (default: ./agentguard_audit)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run the dashboard on (default: 8501)",
    )
    args = parser.parse_args()

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        "--",
        "--audit-path",
        args.audit_path,
    ]
    stcli.main()


if __name__ == "__main__":
    # When run via `streamlit run`, parse the audit-path from args after "--"
    _run_app()
