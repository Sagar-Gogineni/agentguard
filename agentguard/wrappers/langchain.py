"""
AgentGuard LangChain Callback Handler

Provides a LangChain callback handler that makes every LLM call
EU AI Act compliant via AgentGuard.

Usage:
    from agentguard import AgentGuard
    from agentguard.wrappers.langchain import AgentGuardCallback
    from langchain_openai import ChatOpenAI

    guard = AgentGuard(system_name="my-bot", provider_name="my-provider")
    callback = AgentGuardCallback(guard)
    llm = ChatOpenAI(callbacks=[callback])

    # All LLM calls now automatically compliant
    response = llm.invoke("Hello!")
    print(callback.last_result)  # compliance metadata for most recent call
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from ..core import AgentGuard

try:
    from langchain_core.callbacks.base import BaseCallbackHandler as _Base
except ImportError:
    _Base = object  # type: ignore[assignment,misc]


class AgentGuardCallback(_Base):  # type: ignore[misc]
    """LangChain callback handler that runs every LLM call through
    AgentGuard's compliance pipeline.

    Implements ``on_llm_start`` / ``on_chat_model_start``,
    ``on_llm_end``, and ``on_llm_error`` to capture inputs and outputs,
    then logs them via ``guard.invoke()``.

    After each call completes, the compliance metadata is available at
    ``callback.last_result`` and ``callback.results[run_id]``.
    """

    # Properties expected by LangChain's callback system
    raise_error: bool = False
    run_inline: bool = False

    @property
    def ignore_llm(self) -> bool:
        return False

    @property
    def ignore_chat_model(self) -> bool:
        return False

    @property
    def ignore_chain(self) -> bool:
        return True

    @property
    def ignore_agent(self) -> bool:
        return True

    @property
    def ignore_retriever(self) -> bool:
        return True

    @property
    def ignore_retry(self) -> bool:
        return True

    def __init__(
        self,
        guard: AgentGuard,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.guard = guard
        self.user_id = user_id
        self.metadata = metadata

        # Per-run state keyed by run_id
        self._runs: dict[str, dict[str, Any]] = {}

        # Results keyed by run_id string
        self.results: dict[str, dict[str, Any]] = {}
        self.last_result: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    #  on_llm_start — for text-completion LLMs
    # ------------------------------------------------------------------ #

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        model = serialized.get("name") or serialized.get("id", ["unknown"])[-1]
        input_text = prompts[0] if prompts else ""

        self._runs[str(run_id)] = {
            "model": model,
            "input_text": input_text,
            "start_time": time.time(),
            "metadata": metadata,
        }

    # ------------------------------------------------------------------ #
    #  on_chat_model_start — for chat models (ChatOpenAI, etc.)
    # ------------------------------------------------------------------ #

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        model = serialized.get("name") or serialized.get("id", ["unknown"])[-1]

        # Extract the last human message as input_text
        input_text = ""
        if messages:
            for msg in reversed(messages[0]):
                msg_type = getattr(msg, "type", None)
                if msg_type == "human":
                    content = getattr(msg, "content", "")
                    input_text = content if isinstance(content, str) else str(content)
                    break
            if not input_text:
                # Fallback: use the last message's content
                last = messages[0][-1] if messages[0] else None
                if last:
                    content = getattr(last, "content", "")
                    input_text = content if isinstance(content, str) else str(content)

        self._runs[str(run_id)] = {
            "model": model,
            "input_text": input_text,
            "start_time": time.time(),
            "metadata": metadata,
        }

    # ------------------------------------------------------------------ #
    #  on_llm_end — called after successful LLM response
    # ------------------------------------------------------------------ #

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id)
        run_data = self._runs.pop(run_key, None)
        if run_data is None:
            return

        # Extract generated text from LLMResult
        output_text = ""
        if response.generations:
            first_gen = response.generations[0]
            if first_gen:
                output_text = first_gen[0].text

        # Run through guard compliance pipeline
        result = self.guard.invoke(
            func=lambda _: output_text,
            input_text=run_data["input_text"],
            model=run_data["model"],
            user_id=self.user_id,
            metadata=self.metadata or run_data.get("metadata"),
        )

        compliance = {
            "interaction_id": result["interaction_id"],
            "disclosure": result["disclosure"],
            "escalated": result["escalated"],
            "escalation_reason": result["escalation_reason"],
            "content_label": result["content_label"],
            "latency_ms": result["latency_ms"],
        }

        self.results[run_key] = compliance
        self.last_result = compliance

    # ------------------------------------------------------------------ #
    #  on_llm_error — called when the LLM raises an exception
    # ------------------------------------------------------------------ #

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id)
        run_data = self._runs.pop(run_key, None)
        if run_data is None:
            return

        # Log the failed interaction through guard so audit captures the error
        try:
            self.guard.invoke(
                func=lambda _: (_ for _ in ()).throw(error),
                input_text=run_data["input_text"],
                model=run_data["model"],
                user_id=self.user_id,
                metadata=self.metadata or run_data.get("metadata"),
            )
        except BaseException:
            pass  # Error is already logged by guard; don't re-raise from callback
