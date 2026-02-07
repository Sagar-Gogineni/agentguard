"""
AgentGuard Audit Logger

Implements record-keeping requirements per EU AI Act Article 12.
Logs all AI interactions with timestamps, inputs, outputs, and metadata
for compliance auditing and incident investigation.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from .config import AuditBackend


class AuditEntry(BaseModel):
    """A single auditable AI interaction record."""

    interaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    system_name: str
    provider_name: str

    # Interaction data
    user_id: str | None = None
    session_id: str | None = None
    input_text: str | None = None
    input_hash: str | None = None  # SHA-256 hash if input not logged
    output_text: str | None = None
    model_used: str | None = None

    # Compliance metadata
    disclosure_shown: bool = False
    content_labeled: bool = False
    human_escalated: bool = False
    escalation_reason: str | None = None
    confidence_score: float | None = None

    # Performance & safety
    latency_ms: float | None = None
    token_count_input: int | None = None
    token_count_output: int | None = None
    error: str | None = None

    # Custom metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditLogger:
    """
    Compliance audit logger for AI interactions.

    Supports multiple backends:
    - FILE: JSON Lines files (one per day), easy to parse and ship
    - SQLITE: Local SQLite database, good for querying
    - CUSTOM: User-provided callback for external systems
    """

    def __init__(
        self,
        backend: AuditBackend = AuditBackend.FILE,
        path: Path = Path("./agentguard_audit"),
        custom_callback: Callable[[AuditEntry], None] | None = None,
    ):
        self.backend = backend
        self.path = Path(path)
        self._custom_callback = custom_callback
        self._db: sqlite3.Connection | None = None

        if backend == AuditBackend.FILE:
            self.path.mkdir(parents=True, exist_ok=True)
        elif backend == AuditBackend.SQLITE:
            self.path.mkdir(parents=True, exist_ok=True)
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        db_path = self.path / "audit.db"
        self._db = sqlite3.connect(str(db_path))
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                interaction_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                system_name TEXT NOT NULL,
                provider_name TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                input_text TEXT,
                input_hash TEXT,
                output_text TEXT,
                model_used TEXT,
                disclosure_shown INTEGER DEFAULT 0,
                content_labeled INTEGER DEFAULT 0,
                human_escalated INTEGER DEFAULT 0,
                escalation_reason TEXT,
                confidence_score REAL,
                latency_ms REAL,
                token_count_input INTEGER,
                token_count_output INTEGER,
                error TEXT,
                metadata TEXT
            )
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp)
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_user ON audit_log(user_id)
        """)
        self._db.commit()

    def log(self, entry: AuditEntry) -> str:
        """Log an audit entry. Returns the interaction_id."""
        if self.backend == AuditBackend.FILE:
            self._log_file(entry)
        elif self.backend == AuditBackend.SQLITE:
            self._log_sqlite(entry)
        elif self.backend == AuditBackend.CUSTOM and self._custom_callback:
            self._custom_callback(entry)

        return entry.interaction_id

    def _log_file(self, entry: AuditEntry) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = self.path / f"audit_{today}.jsonl"
        with open(log_file, "a") as f:
            f.write(entry.model_dump_json() + "\n")

    def _log_sqlite(self, entry: AuditEntry) -> None:
        if not self._db:
            return
        data = entry.model_dump()
        data["metadata"] = json.dumps(data["metadata"])
        data["disclosure_shown"] = int(data["disclosure_shown"])
        data["content_labeled"] = int(data["content_labeled"])
        data["human_escalated"] = int(data["human_escalated"])

        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        self._db.execute(
            f"INSERT INTO audit_log ({columns}) VALUES ({placeholders})",
            list(data.values()),
        )
        self._db.commit()

    def query(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        user_id: str | None = None,
        escalated_only: bool = False,
        limit: int = 1000,
    ) -> list[AuditEntry]:
        """Query audit logs. Only works with SQLITE backend."""
        if self.backend != AuditBackend.SQLITE or not self._db:
            raise NotImplementedError("Query only supported with SQLITE backend")

        conditions = []
        params: list[Any] = []

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if escalated_only:
            conditions.append("human_escalated = 1")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM audit_log {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self._db.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        entries = []
        for row in rows:
            data = dict(zip(columns, row))
            data["disclosure_shown"] = bool(data["disclosure_shown"])
            data["content_labeled"] = bool(data["content_labeled"])
            data["human_escalated"] = bool(data["human_escalated"])
            data["metadata"] = json.loads(data["metadata"]) if data["metadata"] else {}
            entries.append(AuditEntry(**data))

        return entries

    def get_stats(self) -> dict[str, Any]:
        """Get summary statistics from audit logs."""
        if self.backend != AuditBackend.SQLITE or not self._db:
            raise NotImplementedError("Stats only supported with SQLITE backend")

        stats = {}
        cursor = self._db.execute("SELECT COUNT(*) FROM audit_log")
        stats["total_interactions"] = cursor.fetchone()[0]

        cursor = self._db.execute(
            "SELECT COUNT(*) FROM audit_log WHERE human_escalated = 1"
        )
        stats["total_escalations"] = cursor.fetchone()[0]

        cursor = self._db.execute(
            "SELECT COUNT(*) FROM audit_log WHERE error IS NOT NULL"
        )
        stats["total_errors"] = cursor.fetchone()[0]

        cursor = self._db.execute(
            "SELECT COUNT(*) FROM audit_log WHERE disclosure_shown = 1"
        )
        stats["disclosures_shown"] = cursor.fetchone()[0]

        cursor = self._db.execute("SELECT AVG(latency_ms) FROM audit_log")
        stats["avg_latency_ms"] = cursor.fetchone()[0]

        cursor = self._db.execute(
            "SELECT COUNT(DISTINCT user_id) FROM audit_log WHERE user_id IS NOT NULL"
        )
        stats["unique_users"] = cursor.fetchone()[0]

        return stats

    def close(self) -> None:
        """Close database connection if applicable."""
        if self._db:
            self._db.close()
