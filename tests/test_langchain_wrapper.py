"""Tests for the LangChain callback handler using mock objects."""

import uuid

import pytest

from agentguard import AgentGuard
from agentguard.wrappers.langchain import AgentGuardCallback


# --------------------------------------------------------------------------- #
#  Mock LangChain objects
# --------------------------------------------------------------------------- #


class MockGeneration:
    """Mimics langchain_core.outputs.Generation."""

    def __init__(self, text: str):
        self.text = text
        self.generation_info = None
        self.type = "Generation"


class MockChatGeneration:
    """Mimics langchain_core.outputs.ChatGeneration."""

    def __init__(self, text: str):
        self.text = text
        self.message = MockAIMessage(text)
        self.type = "ChatGeneration"


class MockAIMessage:
    """Mimics langchain_core.messages.AIMessage."""

    def __init__(self, content: str):
        self.content = content
        self.type = "ai"


class MockHumanMessage:
    """Mimics langchain_core.messages.HumanMessage."""

    def __init__(self, content: str):
        self.content = content
        self.type = "human"


class MockSystemMessage:
    """Mimics langchain_core.messages.SystemMessage."""

    def __init__(self, content: str):
        self.content = content
        self.type = "system"


class MockLLMResult:
    """Mimics langchain_core.outputs.LLMResult."""

    def __init__(self, text: str):
        self.generations = [[MockGeneration(text)]]
        self.llm_output = {"token_usage": {"total_tokens": 20}}
        self.run = None


class MockChatLLMResult:
    """Mimics LLMResult from a chat model."""

    def __init__(self, text: str):
        self.generations = [[MockChatGeneration(text)]]
        self.llm_output = {"token_usage": {"total_tokens": 20}}
        self.run = None


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
def callback(guard):
    return AgentGuardCallback(guard)


def make_run_id():
    return uuid.uuid4()


SERIALIZED_LLM = {"name": "OpenAI", "id": ["langchain_openai", "llms", "OpenAI"]}
SERIALIZED_CHAT = {"name": "ChatOpenAI", "id": ["langchain_openai", "chat_models", "ChatOpenAI"]}


# --------------------------------------------------------------------------- #
#  Tests: on_llm_start + on_llm_end (text completion models)
# --------------------------------------------------------------------------- #


class TestTextLLM:
    def test_basic_llm_call(self, callback):
        run_id = make_run_id()

        callback.on_llm_start(
            serialized=SERIALIZED_LLM,
            prompts=["What is the capital of France?"],
            run_id=run_id,
        )
        callback.on_llm_end(
            response=MockLLMResult("Paris is the capital of France."),
            run_id=run_id,
        )

        assert callback.last_result is not None
        assert "interaction_id" in callback.last_result
        assert len(callback.last_result["interaction_id"]) == 36

    def test_result_stored_by_run_id(self, callback):
        run_id = make_run_id()

        callback.on_llm_start(
            serialized=SERIALIZED_LLM,
            prompts=["Hello"],
            run_id=run_id,
        )
        callback.on_llm_end(
            response=MockLLMResult("Hi there!"),
            run_id=run_id,
        )

        assert str(run_id) in callback.results
        assert (
            callback.results[str(run_id)]["interaction_id"]
            == callback.last_result["interaction_id"]
        )

    def test_disclosure_headers_present(self, callback):
        run_id = make_run_id()

        callback.on_llm_start(
            serialized=SERIALIZED_LLM,
            prompts=["Hello"],
            run_id=run_id,
        )
        callback.on_llm_end(
            response=MockLLMResult("Hi!"),
            run_id=run_id,
        )

        headers = callback.last_result["disclosure"]
        assert headers["X-AI-Generated"] == "true"
        assert headers["X-AI-System"] == "test-bot"
        assert headers["X-AI-Provider"] == "Test Corp"

    def test_content_label_includes_model(self, callback):
        run_id = make_run_id()

        callback.on_llm_start(
            serialized=SERIALIZED_LLM,
            prompts=["Hello"],
            run_id=run_id,
        )
        callback.on_llm_end(
            response=MockLLMResult("Hi!"),
            run_id=run_id,
        )

        label = callback.last_result["content_label"]
        assert label["ai_generated"] is True
        assert label["model"] == "OpenAI"


# --------------------------------------------------------------------------- #
#  Tests: on_chat_model_start + on_llm_end (chat models)
# --------------------------------------------------------------------------- #


class TestChatModel:
    def test_chat_model_call(self, callback):
        run_id = make_run_id()

        callback.on_chat_model_start(
            serialized=SERIALIZED_CHAT,
            messages=[
                [
                    MockSystemMessage("You are helpful."),
                    MockHumanMessage("What is 2+2?"),
                ]
            ],
            run_id=run_id,
        )
        callback.on_llm_end(
            response=MockChatLLMResult("2+2 is 4."),
            run_id=run_id,
        )

        assert callback.last_result is not None
        assert "interaction_id" in callback.last_result

    def test_extracts_human_message_as_input(self, guard):
        """Verify the human message is used as input_text for escalation checks."""
        callback = AgentGuardCallback(guard)
        run_id = make_run_id()

        callback.on_chat_model_start(
            serialized=SERIALIZED_CHAT,
            messages=[
                [
                    MockSystemMessage("You are a helpful assistant."),
                    MockHumanMessage("I need legal advice"),
                ]
            ],
            run_id=run_id,
        )
        callback.on_llm_end(
            response=MockChatLLMResult("Please consult a lawyer."),
            run_id=run_id,
        )

        assert callback.last_result["escalated"] is True
        assert "legal" in callback.last_result["escalation_reason"].lower()

    def test_fallback_to_last_message(self, callback):
        """When no human message, fall back to last message content."""
        run_id = make_run_id()

        callback.on_chat_model_start(
            serialized=SERIALIZED_CHAT,
            messages=[[MockSystemMessage("Summarize the report")]],
            run_id=run_id,
        )
        callback.on_llm_end(
            response=MockChatLLMResult("Summary here."),
            run_id=run_id,
        )

        assert callback.last_result is not None

    def test_content_label_uses_chat_model_name(self, callback):
        run_id = make_run_id()

        callback.on_chat_model_start(
            serialized=SERIALIZED_CHAT,
            messages=[[MockHumanMessage("Hello")]],
            run_id=run_id,
        )
        callback.on_llm_end(
            response=MockChatLLMResult("Hi!"),
            run_id=run_id,
        )

        label = callback.last_result["content_label"]
        assert label["model"] == "ChatOpenAI"


# --------------------------------------------------------------------------- #
#  Tests: Escalation
# --------------------------------------------------------------------------- #


class TestEscalation:
    def test_sensitive_keyword_in_prompt(self, callback):
        run_id = make_run_id()

        callback.on_llm_start(
            serialized=SERIALIZED_LLM,
            prompts=["I need legal advice about my contract"],
            run_id=run_id,
        )
        callback.on_llm_end(
            response=MockLLMResult("Please consult a lawyer."),
            run_id=run_id,
        )

        assert callback.last_result["escalated"] is True

    def test_no_escalation_normal_content(self, callback):
        run_id = make_run_id()

        callback.on_llm_start(
            serialized=SERIALIZED_LLM,
            prompts=["What is the weather today?"],
            run_id=run_id,
        )
        callback.on_llm_end(
            response=MockLLMResult("It's sunny."),
            run_id=run_id,
        )

        assert callback.last_result["escalated"] is False


# --------------------------------------------------------------------------- #
#  Tests: Audit logging
# --------------------------------------------------------------------------- #


class TestAuditLogging:
    def test_interaction_logged(self, guard):
        callback = AgentGuardCallback(guard)
        run_id = make_run_id()

        callback.on_llm_start(
            serialized=SERIALIZED_LLM,
            prompts=["Hello"],
            run_id=run_id,
        )
        callback.on_llm_end(
            response=MockLLMResult("Hi!"),
            run_id=run_id,
        )

        stats = guard.audit.get_stats()
        assert stats["total_interactions"] == 1

    def test_multiple_calls_logged(self, guard):
        callback = AgentGuardCallback(guard)

        for _ in range(3):
            run_id = make_run_id()
            callback.on_llm_start(
                serialized=SERIALIZED_LLM,
                prompts=["Hello"],
                run_id=run_id,
            )
            callback.on_llm_end(
                response=MockLLMResult("Hi!"),
                run_id=run_id,
            )

        stats = guard.audit.get_stats()
        assert stats["total_interactions"] == 3

    def test_user_id_forwarded(self, guard):
        callback = AgentGuardCallback(guard, user_id="langchain-user")
        run_id = make_run_id()

        callback.on_llm_start(
            serialized=SERIALIZED_LLM,
            prompts=["Hello"],
            run_id=run_id,
        )
        callback.on_llm_end(
            response=MockLLMResult("Hi!"),
            run_id=run_id,
        )

        entries = guard.audit.query(user_id="langchain-user")
        assert len(entries) == 1


# --------------------------------------------------------------------------- #
#  Tests: Error handling
# --------------------------------------------------------------------------- #


class TestErrorHandling:
    def test_error_logged(self, guard):
        callback = AgentGuardCallback(guard)
        run_id = make_run_id()

        callback.on_llm_start(
            serialized=SERIALIZED_LLM,
            prompts=["Hello"],
            run_id=run_id,
        )
        callback.on_llm_error(
            error=ConnectionError("API unreachable"),
            run_id=run_id,
        )

        stats = guard.audit.get_stats()
        assert stats["total_errors"] == 1

    def test_error_does_not_raise(self, guard):
        """on_llm_error should not propagate exceptions from guard."""
        callback = AgentGuardCallback(guard)
        run_id = make_run_id()

        callback.on_llm_start(
            serialized=SERIALIZED_LLM,
            prompts=["Hello"],
            run_id=run_id,
        )
        # Should not raise
        callback.on_llm_error(
            error=ValueError("Something went wrong"),
            run_id=run_id,
        )

    def test_on_llm_end_without_start_is_noop(self, callback):
        """on_llm_end for unknown run_id should not crash."""
        callback.on_llm_end(
            response=MockLLMResult("Hi!"),
            run_id=make_run_id(),
        )
        assert callback.last_result is None


# --------------------------------------------------------------------------- #
#  Tests: Multiple concurrent runs
# --------------------------------------------------------------------------- #


class TestConcurrentRuns:
    def test_interleaved_runs(self, callback):
        run_a = make_run_id()
        run_b = make_run_id()

        callback.on_llm_start(
            serialized=SERIALIZED_LLM,
            prompts=["Question A"],
            run_id=run_a,
        )
        callback.on_llm_start(
            serialized=SERIALIZED_LLM,
            prompts=["Question B"],
            run_id=run_b,
        )

        callback.on_llm_end(
            response=MockLLMResult("Answer B"),
            run_id=run_b,
        )
        callback.on_llm_end(
            response=MockLLMResult("Answer A"),
            run_id=run_a,
        )

        assert str(run_a) in callback.results
        assert str(run_b) in callback.results
        # last_result should be the most recent (run_a finished last)
        assert callback.last_result == callback.results[str(run_a)]
