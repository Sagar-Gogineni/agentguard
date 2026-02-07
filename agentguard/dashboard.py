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
    stats["total_interactions"] = db.execute(
        "SELECT COUNT(*) FROM audit_log"
    ).fetchone()[0]
    stats["total_escalations"] = db.execute(
        "SELECT COUNT(*) FROM audit_log WHERE human_escalated = 1"
    ).fetchone()[0]
    stats["avg_confidence"] = db.execute(
        "SELECT AVG(confidence_score) FROM audit_log WHERE confidence_score IS NOT NULL"
    ).fetchone()[0]
    stats["avg_latency_ms"] = db.execute(
        "SELECT AVG(latency_ms) FROM audit_log"
    ).fetchone()[0]
    return stats


def _query_escalated(
    db: sqlite3.Connection,
    start_date: str | None = None,
    end_date: str | None = None,
    user_id: str | None = None,
    reason_filter: str | None = None,
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

    # --- Connect to DB ---
    db = _get_db(audit_path)

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
        )

        pending = [r for r in rows if r.get("decision") is None]
        decided = [r for r in rows if r.get("decision") is not None]

        st.subheader(f"Pending Review ({len(pending)})")

        if not pending:
            st.info("No pending escalations.")
        else:
            for row in pending:
                with st.container(border=True):
                    cols = st.columns([2, 1, 3, 3, 2, 1, 2])
                    cols[0].caption("Timestamp")
                    cols[0].write(row["timestamp"][:19])
                    cols[1].caption("User")
                    cols[1].write(row["user_id"] or "—")
                    cols[2].caption("Input")
                    cols[2].write(_truncate(row["input_text"]))
                    cols[3].caption("Output")
                    cols[3].write(_truncate(row["output_text"]))
                    cols[4].caption("Reason")
                    cols[4].write(row["escalation_reason"] or "—")
                    cols[5].caption("Conf.")
                    conf = row["confidence_score"]
                    cols[5].write(f"{conf:.2f}" if conf is not None else "—")

                    btn_cols = cols[6]
                    btn_cols.caption("Action")
                    iid = row["interaction_id"]
                    if btn_cols.button("✅ Approve", key=f"approve_{iid}"):
                        _record_decision(db, iid, "approved")
                        st.rerun()
                    if btn_cols.button("❌ Reject", key=f"reject_{iid}"):
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
        escalated_only = st.checkbox("Escalated only")

        rows = _query_audit_log(
            db,
            start_date=start_str,
            end_date=end_str,
            user_id=user_filter,
            escalated_only=escalated_only,
        )

        if not rows:
            st.info("No audit entries found for the selected filters.")
        else:
            # Build display table
            display = []
            for r in rows:
                display.append({
                    "Timestamp": r["timestamp"][:19],
                    "User": r["user_id"] or "—",
                    "Model": r["model_used"] or "—",
                    "Input": _truncate(r["input_text"], 80),
                    "Output": _truncate(r["output_text"], 80),
                    "Escalated": "Yes" if r["human_escalated"] else "No",
                    "Reason": r["escalation_reason"] or "—",
                    "Confidence": f"{r['confidence_score']:.2f}" if r["confidence_score"] else "—",
                    "Latency (ms)": f"{r['latency_ms']:.0f}" if r["latency_ms"] else "—",
                    "Error": r["error"] or "",
                })
            st.dataframe(display, width="stretch")

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
