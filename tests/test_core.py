"""Tests for AgentGuard core functionality."""

import json
from pathlib import Path

import pytest

from agentguard import AgentGuard, EscalationTriggered


@pytest.fixture
def tmp_audit_dir(tmp_path):
    return str(tmp_path / "audit")


@pytest.fixture
def guard(tmp_audit_dir):
    return AgentGuard(
        system_name="test-bot",
        provider_name="Test Corp",
        risk_level="limited",
        audit_backend="sqlite",
        audit_path=tmp_audit_dir,
        human_escalation="low_confidence",
        confidence_threshold=0.7,
    )


def dummy_llm(query: str) -> str:
    return f"Answer to: {query}"


class TestBasicInvoke:
    def test_invoke_returns_response(self, guard):
        result = guard.invoke(func=dummy_llm, input_text="Hello")
        assert "response" in result
        assert "Answer to: Hello" in result["raw_response"]

    def test_invoke_returns_interaction_id(self, guard):
        result = guard.invoke(func=dummy_llm, input_text="Hello")
        assert "interaction_id" in result
        assert len(result["interaction_id"]) == 36  # UUID format

    def test_invoke_includes_disclosure_headers(self, guard):
        result = guard.invoke(func=dummy_llm, input_text="Hello")
        headers = result["disclosure"]
        assert headers["X-AI-Generated"] == "true"
        assert headers["X-AI-System"] == "test-bot"
        assert headers["X-AI-Provider"] == "Test Corp"
        assert headers["X-AI-Act-Compliant"] == "true"

    def test_invoke_includes_content_label(self, guard):
        result = guard.invoke(func=dummy_llm, input_text="Hello", model="gpt-4")
        label = result["content_label"]
        assert label["ai_generated"] is True
        assert label["generator"] == "test-bot"
        assert label["provider"] == "Test Corp"
        assert label["model"] == "gpt-4"

    def test_invoke_measures_latency(self, guard):
        result = guard.invoke(func=dummy_llm, input_text="Hello")
        assert result["latency_ms"] > 0


class TestHumanEscalation:
    def test_low_confidence_triggers_escalation(self, guard):
        result = guard.invoke(
            func=dummy_llm,
            input_text="Hello",
            confidence=0.3,
        )
        assert result["escalated"] is True

    def test_high_confidence_no_escalation(self, guard):
        result = guard.invoke(
            func=dummy_llm,
            input_text="Hello",
            confidence=0.95,
        )
        assert result["escalated"] is False

    def test_sensitive_keyword_triggers_escalation(self, guard):
        result = guard.invoke(
            func=dummy_llm,
            input_text="I need legal advice about my contract",
            confidence=0.95,
        )
        assert result["escalated"] is True
        assert "legal" in result["escalation_reason"].lower()

    def test_block_on_escalation_raises(self, tmp_audit_dir):
        guard = AgentGuard(
            system_name="test-bot",
            provider_name="Test Corp",
            audit_backend="sqlite",
            audit_path=tmp_audit_dir,
            human_escalation="low_confidence",
            confidence_threshold=0.7,
            block_on_escalation=True,
        )
        with pytest.raises(EscalationTriggered):
            guard.invoke(
                func=dummy_llm,
                input_text="Hello",
                confidence=0.1,
            )

    def test_pending_reviews_populated(self, guard):
        guard.invoke(
            func=dummy_llm,
            input_text="I need medical diagnosis",
            confidence=0.95,
        )
        assert len(guard.pending_reviews) > 0


class TestDecorator:
    def test_compliant_decorator(self, guard):
        @guard.compliant
        def my_bot(query: str) -> str:
            return f"Bot says: {query}"

        result = my_bot("Hello")
        assert "Bot says: Hello" in result["raw_response"]
        assert result["interaction_id"] is not None

    def test_compliant_decorator_with_model(self, guard):
        @guard.compliant(model="claude-3")
        def my_bot(query: str) -> str:
            return f"Bot says: {query}"

        result = my_bot("Hello")
        assert result["content_label"]["model"] == "claude-3"


class TestContextManager:
    def test_context_manager_records(self, guard):
        with guard.interaction(user_id="u-1") as ctx:
            info = ctx.record(
                input_text="Hello",
                output_text="Hi there",
            )
        assert info["interaction_id"] is not None

    def test_context_manager_detects_escalation(self, guard):
        with guard.interaction() as ctx:
            info = ctx.record(
                input_text="I need legal help",
                output_text="Please consult a lawyer",
                confidence=0.3,
            )
        assert info["escalated"] is True


class TestAuditLogging:
    def test_sqlite_logs_interaction(self, guard):
        guard.invoke(func=dummy_llm, input_text="Test query", user_id="u-1")
        stats = guard.audit.get_stats()
        assert stats["total_interactions"] == 1
        assert stats["unique_users"] == 1

    def test_multiple_interactions_logged(self, guard):
        for i in range(5):
            guard.invoke(func=dummy_llm, input_text=f"Query {i}")
        stats = guard.audit.get_stats()
        assert stats["total_interactions"] == 5

    def test_file_backend_creates_log(self, tmp_audit_dir):
        guard = AgentGuard(
            system_name="test-bot",
            provider_name="Test Corp",
            audit_backend="file",
            audit_path=tmp_audit_dir,
        )
        guard.invoke(func=dummy_llm, input_text="Hello")
        log_files = list(Path(tmp_audit_dir).glob("audit_*.jsonl"))
        assert len(log_files) == 1
        with open(log_files[0]) as f:
            entry = json.loads(f.readline())
        assert entry["system_name"] == "test-bot"


class TestComplianceReport:
    def test_generate_json_report(self, guard):
        guard.invoke(func=dummy_llm, input_text="Hello")
        path = guard.generate_report()
        assert Path(path).exists()
        with open(path) as f:
            report = json.load(f)
        assert report["system_identification"]["system_name"] == "test-bot"

    def test_generate_markdown_report(self, guard):
        guard.invoke(func=dummy_llm, input_text="Hello")
        md = guard.generate_report_markdown()
        assert "# EU AI Act Compliance Report" in md
        assert "test-bot" in md
        assert "Test Corp" in md


class TestErrorHandling:
    def test_llm_error_logged(self, guard):
        def failing_llm(query: str) -> str:
            raise ValueError("LLM failed")

        with pytest.raises(ValueError):
            guard.invoke(func=failing_llm, input_text="Hello")

        stats = guard.audit.get_stats()
        assert stats["total_errors"] == 1
