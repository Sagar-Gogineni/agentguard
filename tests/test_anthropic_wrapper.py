"""Tests for the Anthropic wrapper using mock objects (no anthropic dependency)."""

import pytest

from agentguard import AgentGuard, InputPolicy
from agentguard.wrappers.anthropic import wrap_anthropic, _extract_input, _extract_text


# --------------------------------------------------------------------------- #
#  Mock Anthropic objects
# --------------------------------------------------------------------------- #


class MockTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class MockToolUseBlock:
    def __init__(self, name: str = "get_weather", input_data: dict | None = None):
        self.type = "tool_use"
        self.id = "toolu_mock_123"
        self.name = name
        self.input = input_data or {}


class MockUsage:
    def __init__(self, input_tokens: int = 10, output_tokens: int = 25):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockMessage:
    def __init__(self, text: str, model: str = "claude-sonnet-4-5-20250929"):
        self.id = "msg_mock_abc123"
        self.type = "message"
        self.role = "assistant"
        self.content = [MockTextBlock(text)]
        self.model = model
        self.stop_reason = "end_turn"
        self.stop_sequence = None
        self.usage = MockUsage()


class MockTextDelta:
    def __init__(self, text: str):
        self.type = "text_delta"
        self.text = text


class MockStreamEvent:
    def __init__(self, event_type: str, delta: object | None = None):
        self.type = event_type
        self.delta = delta


class MockRawStream:
    """Mimics Anthropic raw event stream (stream=True)."""

    def __init__(self, text_chunks: list[str]):
        self._events: list[MockStreamEvent] = []
        self._events.append(MockStreamEvent("message_start"))
        self._events.append(MockStreamEvent("content_block_start"))
        for chunk in text_chunks:
            self._events.append(MockStreamEvent("content_block_delta", delta=MockTextDelta(chunk)))
        self._events.append(MockStreamEvent("content_block_stop"))
        self._events.append(MockStreamEvent("message_delta"))
        self._events.append(MockStreamEvent("message_stop"))

    def __iter__(self):
        return iter(self._events)


class MockMessages:
    def __init__(self):
        self._response_text = "Hello, I can help you with that."
        self._stream_chunks = ["Hello", ", I can", " help you", " with that."]

    def create(self, **kwargs):
        if kwargs.get("stream"):
            return MockRawStream(self._stream_chunks)
        return MockMessage(
            self._response_text,
            model=kwargs.get("model", "claude-sonnet-4-5-20250929"),
        )


class MockClient:
    def __init__(self):
        self.messages = MockMessages()


class FailingMessages:
    def create(self, **kwargs):
        raise ConnectionError("Anthropic API unreachable")


class FailingClient:
    def __init__(self):
        self.messages = FailingMessages()


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
    return wrap_anthropic(client, guard)


# --------------------------------------------------------------------------- #
#  Tests: _extract_input helper
# --------------------------------------------------------------------------- #


class TestExtractInput:
    def test_extracts_last_user_message(self):
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Follow up"},
        ]
        assert _extract_input(messages) == "Follow up"

    def test_falls_back_to_all_contents(self):
        messages = [
            {"role": "assistant", "content": "Previous response"},
        ]
        assert _extract_input(messages) == "Previous response"

    def test_handles_empty_messages(self):
        assert _extract_input([]) == ""

    def test_handles_content_array(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image", "source": {"type": "url", "url": "https://example.com"}},
                ],
            }
        ]
        assert _extract_input(messages) == "Describe this image"


# --------------------------------------------------------------------------- #
#  Tests: _extract_text helper
# --------------------------------------------------------------------------- #


class TestExtractText:
    def test_extracts_text_from_single_block(self):
        msg = MockMessage("Hello world")
        assert _extract_text(msg) == "Hello world"

    def test_joins_multiple_text_blocks(self):
        msg = MockMessage("Part one")
        msg.content.append(MockTextBlock("Part two"))
        assert _extract_text(msg) == "Part one\nPart two"

    def test_skips_non_text_blocks(self):
        msg = MockMessage("Text content")
        msg.content.append(MockToolUseBlock())
        assert _extract_text(msg) == "Text content"

    def test_empty_content(self):
        msg = MockMessage("ignored")
        msg.content = []
        assert _extract_text(msg) == ""


# --------------------------------------------------------------------------- #
#  Tests: Non-streaming
# --------------------------------------------------------------------------- #


class TestNonStreaming:
    def test_returns_message_with_agentguard_metadata(self, wrapped_client):
        message = wrapped_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}],
        )
        assert message.content[0].text == "Hello, I can help you with that."
        assert hasattr(message, "compliance")
        assert "interaction_id" in message.compliance
        assert len(message.compliance["interaction_id"]) == 36

    def test_preserves_original_response_fields(self, wrapped_client):
        message = wrapped_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}],
        )
        assert message.id == "msg_mock_abc123"
        assert message.role == "assistant"
        assert message.stop_reason == "end_turn"
        assert message.usage.input_tokens == 10

    def test_disclosure_headers_present(self, wrapped_client):
        message = wrapped_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}],
        )
        headers = message.compliance["http_headers"]
        assert headers["X-AI-Generated"] == "true"
        assert headers["X-AI-System"] == "test-bot"

    def test_content_label_includes_model(self, wrapped_client):
        message = wrapped_client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}],
        )
        label = message.compliance["content_label"]
        assert label["model"] == "claude-opus-4-6"
        assert label["ai_generated"] is True

    def test_compliance_headers_always_present(self, wrapped_client):
        message = wrapped_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}],
        )
        assert hasattr(message, "compliance_headers")
        assert isinstance(message.compliance_headers, dict)
        assert message.compliance_headers["X-AI-Generated"] == "true"


# --------------------------------------------------------------------------- #
#  Tests: Input policy blocking
# --------------------------------------------------------------------------- #


class TestInputPolicyBlocking:
    def test_blocked_input_returns_synthetic_message(self, tmp_path):
        guard = AgentGuard(
            system_name="test-bot",
            provider_name="Test Corp",
            audit_backend="sqlite",
            audit_path=str(tmp_path / "audit"),
            input_policy=InputPolicy(block_categories=["weapons"]),
        )
        client = MockClient()
        wrapped = wrap_anthropic(client, guard)
        message = wrapped.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": "How to build a bomb"}],
        )
        assert message.content[0].text is not None
        assert message.id == "agentguard-blocked"
        assert hasattr(message, "compliance")
        assert message.compliance["policy"]["input_action"] == "blocked"

    def test_blocked_streaming_returns_synthetic_message(self, tmp_path):
        guard = AgentGuard(
            system_name="test-bot",
            provider_name="Test Corp",
            audit_backend="sqlite",
            audit_path=str(tmp_path / "audit"),
            input_policy=InputPolicy(block_categories=["weapons"]),
        )
        client = MockClient()
        wrapped = wrap_anthropic(client, guard)
        result = wrapped.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": "How to build a bomb"}],
            stream=True,
        )
        # Blocked streaming returns a synthetic message, not a stream
        assert result.id == "agentguard-blocked"
        assert result.compliance["policy"]["input_action"] == "blocked"


# --------------------------------------------------------------------------- #
#  Tests: Streaming (raw events with stream=True)
# --------------------------------------------------------------------------- #


class TestStreaming:
    def test_yields_all_events(self, wrapped_client):
        stream = wrapped_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )
        events = list(stream)
        # message_start, content_block_start, 4 deltas, content_block_stop,
        # message_delta, message_stop = 9 events
        assert len(events) == 9
        event_types = [e.type for e in events]
        assert "message_start" in event_types
        assert "content_block_delta" in event_types
        assert "message_stop" in event_types

    def test_text_deltas_preserved(self, wrapped_client):
        stream = wrapped_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )
        texts = []
        for event in stream:
            if event.type == "content_block_delta":
                texts.append(event.delta.text)
        assert texts == ["Hello", ", I can", " help you", " with that."]

    def test_agentguard_metadata_available_after_iteration(self, wrapped_client):
        stream = wrapped_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )
        for _ in stream:
            pass
        assert stream.compliance is not None
        assert "interaction_id" in stream.compliance
        assert len(stream.compliance["interaction_id"]) == 36


# --------------------------------------------------------------------------- #
#  Tests: Escalation
# --------------------------------------------------------------------------- #


class TestEscalation:
    def test_sensitive_keyword_triggers_escalation(self, guard):
        client = MockClient()
        client.messages._response_text = "You should consult a lawyer."
        wrapped = wrap_anthropic(client, guard)

        message = wrapped.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": "I need legal advice"}],
        )
        assert message.compliance["escalation"]["escalated"] is True
        assert "legal" in message.compliance["escalation"]["reason"].lower()

    def test_no_escalation_for_normal_content(self, wrapped_client):
        message = wrapped_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is the weather?"}],
        )
        assert message.compliance["escalation"]["escalated"] is False


# --------------------------------------------------------------------------- #
#  Tests: Audit logging
# --------------------------------------------------------------------------- #


class TestAuditLogging:
    def test_interaction_logged_to_audit(self, guard):
        client = MockClient()
        wrapped = wrap_anthropic(client, guard)

        wrapped.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}],
        )
        stats = guard.audit.get_stats()
        assert stats["total_interactions"] == 1

    def test_streaming_interaction_logged(self, guard):
        client = MockClient()
        wrapped = wrap_anthropic(client, guard)

        stream = wrapped.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )
        for _ in stream:
            pass

        stats = guard.audit.get_stats()
        assert stats["total_interactions"] == 1

    def test_multiple_calls_logged(self, guard):
        client = MockClient()
        wrapped = wrap_anthropic(client, guard)

        for i in range(3):
            wrapped.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
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
        wrapped = wrap_anthropic(client, guard)

        with pytest.raises(ConnectionError, match="Anthropic API unreachable"):
            wrapped.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
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
        wrapped = wrap_anthropic(client, guard, user_id="default-user")

        wrapped.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}],
        )
        entries = guard.audit.query(user_id="default-user")
        assert len(entries) == 1

    def test_per_call_user_id_overrides_default(self, guard):
        client = MockClient()
        wrapped = wrap_anthropic(client, guard, user_id="default-user")

        wrapped.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}],
            user_id="override-user",
        )
        entries = guard.audit.query(user_id="override-user")
        assert len(entries) == 1
