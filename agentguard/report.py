"""
AgentGuard Compliance Reporter

Generates compliance documentation per EU AI Act Articles 11 and 18.
Produces audit summaries and technical documentation from logged data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audit import AuditLogger
from .config import AgentGuardConfig, RiskLevel


class ComplianceReporter:
    """
    Generates EU AI Act compliance reports from audit data.

    Covers:
    - Article 11: Technical documentation
    - Article 12: Record-keeping summary
    - Article 13: Transparency reporting
    - Article 18: Deployer documentation
    """

    def __init__(self, config: AgentGuardConfig, logger: AuditLogger):
        self.config = config
        self.logger = logger

    def generate_summary(self) -> dict[str, Any]:
        """
        Generate a compliance summary report.

        Returns a structured dict that can be serialized to JSON
        or rendered into any format.
        """
        stats = {}
        try:
            stats = self.logger.get_stats()
        except NotImplementedError:
            stats = {"note": "Stats require SQLITE audit backend"}

        report = {
            "report_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generator": "AgentGuard v0.1.0",
                "report_type": "EU AI Act Compliance Summary",
            },
            "system_identification": {
                "system_name": self.config.system_name,
                "provider_name": self.config.provider_name,
                "version": self.config.version,
                "risk_level": self.config.risk_level.value,
                "intended_purpose": self.config.intended_purpose,
            },
            "transparency_compliance": {
                "article_50_disclosure": {
                    "method": self.config.disclosure_method.value,
                    "disclosure_text": self.config.render_disclosure(),
                    "content_labeling_enabled": self.config.label_content,
                },
            },
            "human_oversight": {
                "article_14_compliance": {
                    "escalation_mode": self.config.human_escalation.value,
                    "confidence_threshold": self.config.confidence_threshold,
                    "sensitive_keywords_count": len(
                        self.config.sensitive_keywords
                    ),
                },
            },
            "record_keeping": {
                "article_12_compliance": {
                    "audit_backend": self.config.audit_backend.value,
                    "inputs_logged": self.config.log_inputs,
                    "outputs_logged": self.config.log_outputs,
                    "retention_days": self.config.retention_days,
                    "statistics": stats,
                },
            },
        }

        if self.config.risk_level == RiskLevel.HIGH:
            report["high_risk_documentation"] = {
                "article_9_risk_management": self.config.risk_management_ref
                or "NOT PROVIDED",
                "article_10_data_governance": self.config.data_governance_ref
                or "NOT PROVIDED",
                "article_11_technical_doc": self.config.technical_doc_ref
                or "NOT PROVIDED",
            }

        return report

    def save_report(self, path: str | Path | None = None) -> Path:
        """Generate and save compliance report as JSON."""
        report = self.generate_summary()
        output_path = Path(
            path or self.config.audit_path / "compliance_report.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return output_path

    def generate_markdown(self) -> str:
        """Generate a human-readable Markdown compliance report."""
        report = self.generate_summary()
        sys_info = report["system_identification"]
        transparency = report["transparency_compliance"]["article_50_disclosure"]
        oversight = report["human_oversight"]["article_14_compliance"]
        records = report["record_keeping"]["article_12_compliance"]
        stats = records.get("statistics", {})

        md = f"""# EU AI Act Compliance Report

**Generated:** {report["report_metadata"]["generated_at"]}
**Generator:** {report["report_metadata"]["generator"]}

---

## 1. System Identification (Article 16)

| Field | Value |
|-------|-------|
| System Name | {sys_info["system_name"]} |
| Provider | {sys_info["provider_name"]} |
| Version | {sys_info["version"]} |
| Risk Level | {sys_info["risk_level"].upper()} |
| Intended Purpose | {sys_info["intended_purpose"] or "Not specified"} |

## 2. Transparency Compliance (Article 50)

| Requirement | Status |
|-------------|--------|
| User Disclosure | ✅ Enabled ({transparency["method"]}) |
| Content Labeling | {"✅ Enabled" if transparency["content_labeling_enabled"] else "❌ Disabled"} |
| Disclosure Text | "{transparency["disclosure_text"]}" |

## 3. Human Oversight (Article 14)

| Setting | Value |
|---------|-------|
| Escalation Mode | {oversight["escalation_mode"]} |
| Confidence Threshold | {oversight["confidence_threshold"]} |
| Sensitive Keywords Monitored | {oversight["sensitive_keywords_count"]} |

## 4. Record-Keeping (Article 12)

| Setting | Value |
|---------|-------|
| Audit Backend | {records["audit_backend"]} |
| Inputs Logged | {"Yes" if records["inputs_logged"] else "No"} |
| Outputs Logged | {"Yes" if records["outputs_logged"] else "No"} |
| Retention Period | {records["retention_days"]} days |

### Interaction Statistics

| Metric | Value |
|--------|-------|
| Total Interactions | {stats.get("total_interactions", "N/A")} |
| Human Escalations | {stats.get("total_escalations", "N/A")} |
| Errors | {stats.get("total_errors", "N/A")} |
| Disclosures Shown | {stats.get("disclosures_shown", "N/A")} |
| Unique Users | {stats.get("unique_users", "N/A")} |
| Avg Latency (ms) | {stats.get("avg_latency_ms", "N/A")} |
"""

        if "high_risk_documentation" in report:
            hr = report["high_risk_documentation"]
            md += f"""
## 5. High-Risk System Documentation

| Reference | Status |
|-----------|--------|
| Risk Management (Art. 9) | {hr["article_9_risk_management"]} |
| Data Governance (Art. 10) | {hr["article_10_data_governance"]} |
| Technical Documentation (Art. 11) | {hr["article_11_technical_doc"]} |
"""

        md += """
---

*This report was automatically generated by AgentGuard.
It provides a technical compliance snapshot and does not constitute legal advice.
Consult qualified legal professionals for full EU AI Act compliance assessment.*
"""
        return md
