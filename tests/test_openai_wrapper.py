"""Tests for the OpenAI wrapper using mock objects (no openai dependency)."""

import pytest

from agentguard import AgentGuard
from agentguard.wrappers.openai import wrap_openai, _extract_input


# --------------------------------------------------------------------------- #
#  Mock OpenAI objects
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
        self.id = "chatcmpl-mock-123"
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
        self.id = "chatcmpl-mock-chunk"
        self.object = "chat.completion.chunk"
        self.created = 1700000000
        self.model = "gpt-4"
        self.choices = [MockChunkChoice(content)]


class MockStream:
    """Mimics openai.Stream[ChatCompletionChunk]."""

    def __init__(self, chunks: list[str]):
        self._chunks = [MockChunk(c) for c in chunks]

    def __iter__(self):
        return iter(self._chunks)


class MockCompletions:
    def __init__(self):
        self._response_text = "Hello, I can help you with that."
        self._stream_chunks = ["Hello", ", I can", " help you", " with that."]

    def create(self, **kwargs):
        if kwargs.get("stream"):
            return MockStream(self._stream_chunks)
        return MockChatCompletion(self._response_text, model=kwargs.get("model", "gpt-4"))


class MockChat:
    def __init__(self):
        self.completions = MockCompletions()


class MockClient:
    def __init__(self):
        self.chat = MockChat()


class FailingCompletions:
    def create(self, **kwargs):
        raise ConnectionError("OpenAI API unreachable")


class FailingChat:
    def __init__(self):
        self.completions = FailingCompletions()


class FailingClient:
    def __init__(self):
        self.chat = FailingChat()


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def guard(tmp_path):
    return AgentGuard(
        system_name="test-bot",
        provider_name="Test Corp",
        risk_level="limited",
        audit_backend="sqlite",
        audit_path=str(tmp_path / "audit"),
        human_escalation="low_confidence",
        confidence_threshold=0.7,
    )


@pytest.fixture
def wrapped_client(guard):
    client = MockClient()
    return wrap_openai(client, guard)


# --------------------------------------------------------------------------- #
#  Tests: _extract_input helper
# --------------------------------------------------------------------------- #


class TestExtractInput:
    def test_extracts_last_user_message(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Follow up question"},
        ]
        assert _extract_input(messages) == "Follow up question"

    def test_falls_back_to_all_contents(self):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": "Hello"},
        ]
        assert _extract_input(messages) == "System prompt Hello"

    def test_handles_empty_messages(self):
        assert _extract_input([]) == ""

    def test_handles_content_array(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            }
        ]
        assert _extract_input(messages) == "Describe this image"


# --------------------------------------------------------------------------- #
#  Tests: Non-streaming
# --------------------------------------------------------------------------- #


class TestNonStreaming:
    def test_returns_response_with_agentguard_metadata(self, wrapped_client):
        response = wrapped_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        assert response.choices[0].message.content == "Hello, I can help you with that."
        assert hasattr(response, "_agentguard")
        assert "interaction_id" in response._agentguard
        assert len(response._agentguard["interaction_id"]) == 36

    def test_preserves_original_response_fields(self, wrapped_client):
        response = wrapped_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        assert response.id == "chatcmpl-mock-123"
        assert response.model == "gpt-4"
        assert response.object == "chat.completion"
        assert response.choices[0].finish_reason == "stop"

    def test_disclosure_headers_present(self, wrapped_client):
        response = wrapped_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        headers = response._agentguard["disclosure"]
        assert headers["X-AI-Generated"] == "true"
        assert headers["X-AI-System"] == "test-bot"

    def test_content_label_includes_model(self, wrapped_client):
        response = wrapped_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        label = response._agentguard["content_label"]
        assert label["model"] == "gpt-4o"
        assert label["ai_generated"] is True


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
        assert contents == ["Hello", ", I can", " help you", " with that."]

    def test_agentguard_metadata_available_after_iteration(self, wrapped_client):
        stream = wrapped_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )
        # Consume the stream
        for _ in stream:
            pass
        assert stream._agentguard is not None
        assert "interaction_id" in stream._agentguard
        assert len(stream._agentguard["interaction_id"]) == 36


# --------------------------------------------------------------------------- #
#  Tests: Escalation
# --------------------------------------------------------------------------- #


class TestEscalation:
    def test_sensitive_keyword_triggers_escalation(self, guard, tmp_path):
        client = MockClient()
        client.chat.completions._response_text = "You should consult a lawyer."
        wrapped = wrap_openai(client, guard)

        response = wrapped.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "I need legal advice"}],
        )
        assert response._agentguard["escalated"] is True
        assert "legal" in response._agentguard["escalation_reason"].lower()

    def test_no_escalation_for_normal_content(self, wrapped_client):
        response = wrapped_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "What is the weather?"}],
        )
        assert response._agentguard["escalated"] is False


# --------------------------------------------------------------------------- #
#  Tests: Audit logging
# --------------------------------------------------------------------------- #


class TestAuditLogging:
    def test_interaction_logged_to_audit(self, guard):
        client = MockClient()
        wrapped = wrap_openai(client, guard)

        wrapped.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        stats = guard.audit.get_stats()
        assert stats["total_interactions"] == 1

    def test_streaming_interaction_logged(self, guard):
        client = MockClient()
        wrapped = wrap_openai(client, guard)

        stream = wrapped.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )
        for _ in stream:
            pass

        stats = guard.audit.get_stats()
        assert stats["total_interactions"] == 1

    def test_multiple_calls_logged(self, guard):
        client = MockClient()
        wrapped = wrap_openai(client, guard)

        for i in range(3):
            wrapped.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"Query {i}"}],
            )
        stats = guard.audit.get_stats()
        assert stats["total_interactions"] == 3


# --------------------------------------------------------------------------- #
#  Tests: Error handling
# --------------------------------------------------------------------------- #


class TestErrorHandling:
    def test_error_logged_and_reraised(self, guard):
        client = FailingClient()
        wrapped = wrap_openai(client, guard)

        with pytest.raises(ConnectionError, match="OpenAI API unreachable"):
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
    def test_default_user_id_forwarded(self, guard):
        client = MockClient()
        wrapped = wrap_openai(client, guard, user_id="default-user")

        wrapped.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        entries = guard.audit.query(user_id="default-user")
        assert len(entries) == 1

    def test_per_call_user_id_overrides_default(self, guard):
        client = MockClient()
        wrapped = wrap_openai(client, guard, user_id="default-user")

        wrapped.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            user_id="override-user",
        )
        entries = guard.audit.query(user_id="override-user")
        assert len(entries) == 1
