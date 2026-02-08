"""Tests for the Azure OpenAI wrapper using mock objects.

Azure OpenAI uses the same SDK interface as OpenAI, so the wrapper
delegates to wrap_openai.  These tests verify that wrap_azure_openai
works end-to-end with the same mock objects used in test_openai_wrapper.
"""

import pytest

from agentguard import AgentGuard
from agentguard.wrappers.azure_openai import wrap_azure_openai


# --------------------------------------------------------------------------- #
#  Mock Azure OpenAI objects (identical interface to OpenAI)
# --------------------------------------------------------------------------- #


class MockMessage:
    def __init__(self, content: str, role: str = "assistant"):
        self.content = content
        self.role = role


class MockChoice:
    def __init__(self, content: str, index: int = 0):
        self.index = index
        self.message = MockMessage(content)
        self.finish_reason = "stop"


class MockChatCompletion:
    def __init__(self, content: str, model: str = "gpt-4"):
        self.id = "chatcmpl-azure-mock-123"
        self.object = "chat.completion"
        self.created = 1700000000
        self.model = model
        self.choices = [MockChoice(content)]
        self.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


class MockDelta:
    def __init__(self, content: str | None = None):
        self.content = content
        self.role = None


class MockChunkChoice:
    def __init__(self, content: str | None = None, index: int = 0):
        self.index = index
        self.delta = MockDelta(content)
        self.finish_reason = None


class MockChunk:
    def __init__(self, content: str | None = None):
        self.id = "chatcmpl-azure-mock-chunk"
        self.object = "chat.completion.chunk"
        self.created = 1700000000
        self.model = "gpt-4"
        self.choices = [MockChunkChoice(content)]


class MockStream:
    def __init__(self, chunks: list[str]):
        self._chunks = [MockChunk(c) for c in chunks]

    def __iter__(self):
        return iter(self._chunks)


class MockCompletions:
    def __init__(self):
        self._response_text = "Hello from Azure OpenAI."
        self._stream_chunks = ["Hello", " from", " Azure", " OpenAI."]

    def create(self, **kwargs):
        if kwargs.get("stream"):
            return MockStream(self._stream_chunks)
        return MockChatCompletion(self._response_text, model=kwargs.get("model", "gpt-4"))


class MockChat:
    def __init__(self):
        self.completions = MockCompletions()


class MockAzureClient:
    """Mimics openai.AzureOpenAI — same chat.completions.create interface."""

    def __init__(self):
        self.chat = MockChat()


class FailingCompletions:
    def create(self, **kwargs):
        raise ConnectionError("Azure OpenAI endpoint unreachable")


class FailingChat:
    def __init__(self):
        self.completions = FailingCompletions()


class FailingAzureClient:
    def __init__(self):
        self.chat = FailingChat()


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def guard(tmp_path):
    return AgentGuard(
        system_name="azure-bot",
        provider_name="Test Corp",
        risk_level="limited",
        audit_backend="sqlite",
        audit_path=str(tmp_path / "audit"),
        human_escalation="low_confidence",
        confidence_threshold=0.7,
    )


@pytest.fixture
def wrapped_client(guard):
    client = MockAzureClient()
    return wrap_azure_openai(client, guard)


# --------------------------------------------------------------------------- #
#  Tests: Non-streaming
# --------------------------------------------------------------------------- #


class TestNonStreaming:
    def test_returns_response_with_agentguard_metadata(self, wrapped_client):
        response = wrapped_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        assert response.choices[0].message.content == "Hello from Azure OpenAI."
        assert hasattr(response, "compliance")
        assert "interaction_id" in response.compliance
        assert len(response.compliance["interaction_id"]) == 36

    def test_preserves_azure_response_fields(self, wrapped_client):
        response = wrapped_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        assert response.id == "chatcmpl-azure-mock-123"
        assert response.model == "gpt-4"

    def test_disclosure_headers_use_guard_config(self, wrapped_client):
        response = wrapped_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        headers = response.compliance["http_headers"]
        assert headers["X-AI-Generated"] == "true"
        assert headers["X-AI-System"] == "azure-bot"
        assert headers["X-AI-Provider"] == "Test Corp"

    def test_content_label_includes_deployment_model(self, wrapped_client):
        response = wrapped_client.chat.completions.create(
            model="my-gpt4-deployment",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        label = response.compliance["content_label"]
        assert label["model"] == "my-gpt4-deployment"
        assert label["ai_generated"] is True

    def test_compliance_headers_always_present(self, wrapped_client):
        response = wrapped_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        assert hasattr(response, "compliance_headers")
        assert isinstance(response.compliance_headers, dict)
        assert response.compliance_headers["X-AI-Generated"] == "true"


# --------------------------------------------------------------------------- #
#  Tests: Streaming
# --------------------------------------------------------------------------- #


class TestStreaming:
    def test_yields_all_chunks(self, wrapped_client):
        stream = wrapped_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )
        chunks = list(stream)
        assert len(chunks) == 4
        contents = [c.choices[0].delta.content for c in chunks]
        assert contents == ["Hello", " from", " Azure", " OpenAI."]

    def test_agentguard_metadata_after_iteration(self, wrapped_client):
        stream = wrapped_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )
        for _ in stream:
            pass
        assert stream._agentguard is not None
        assert "interaction_id" in stream.compliance


# --------------------------------------------------------------------------- #
#  Tests: Escalation
# --------------------------------------------------------------------------- #


class TestEscalation:
    def test_sensitive_keyword_triggers_escalation(self, guard):
        client = MockAzureClient()
        client.chat.completions._response_text = "Consult a lawyer for legal advice."
        wrapped = wrap_azure_openai(client, guard)

        response = wrapped.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "I need legal help"}],
        )
        assert response.compliance["escalation"]["escalated"] is True

    def test_no_escalation_for_normal_content(self, wrapped_client):
        response = wrapped_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )
        assert response.compliance["escalation"]["escalated"] is False


# --------------------------------------------------------------------------- #
#  Tests: Audit logging
# --------------------------------------------------------------------------- #


class TestAuditLogging:
    def test_interaction_logged(self, guard):
        client = MockAzureClient()
        wrapped = wrap_azure_openai(client, guard)

        wrapped.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        stats = guard.audit.get_stats()
        assert stats["total_interactions"] == 1

    def test_streaming_logged(self, guard):
        client = MockAzureClient()
        wrapped = wrap_azure_openai(client, guard)

        stream = wrapped.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )
        for _ in stream:
            pass
        stats = guard.audit.get_stats()
        assert stats["total_interactions"] == 1


# --------------------------------------------------------------------------- #
#  Tests: Error handling
# --------------------------------------------------------------------------- #


class TestErrorHandling:
    def test_error_logged_and_reraised(self, guard):
        client = FailingAzureClient()
        wrapped = wrap_azure_openai(client, guard)

        with pytest.raises(ConnectionError, match="Azure OpenAI endpoint unreachable"):
            wrapped.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello!"}],
            )
        stats = guard.audit.get_stats()
        assert stats["total_errors"] == 1


# --------------------------------------------------------------------------- #
#  Tests: Default kwargs
# --------------------------------------------------------------------------- #


class TestDefaults:
    def test_default_user_id(self, guard):
        client = MockAzureClient()
        wrapped = wrap_azure_openai(client, guard, user_id="azure-user")

        wrapped.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        entries = guard.audit.query(user_id="azure-user")
        assert len(entries) == 1
