"""
AgentGuard OpenAI Wrapper

Drop-in wrapper for the OpenAI Python client that makes every
chat.completions.create() call EU AI Act compliant.

Usage:
    from agentguard import AgentGuard
    from agentguard.wrappers.openai import wrap_openai
    from openai import OpenAI

    guard = AgentGuard(system_name="my-bot", provider_name="my-provider")
    client = wrap_openai(OpenAI(), guard)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response.choices[0].message.content)  # untouched LLM output
    print(response.compliance)                  # structured compliance metadata
"""

from __future__ import annotations

from typing import Any, Iterator

from ..core import AgentGuard


class _SyntheticMessage:
    """Minimal stand-in for an OpenAI ChatCompletionMessage when the
    request was blocked by input policy (LLM never called)."""

    def __init__(self, content: str):
        self.content = content
        self.role = "assistant"
        self.function_call = None
        self.tool_calls = None


class _SyntheticChoice:
    def __init__(self, content: str):
        self.index = 0
        self.message = _SyntheticMessage(content)
        self.finish_reason = "stop"


class _SyntheticResponse:
    """Lightweight response returned when input policy blocks the call."""

    def __init__(self, content: str, model: str | None = None):
        self.id = "agentguard-blocked"
        self.object = "chat.completion"
        self.model = model or "agentguard-blocked"
        self.choices = [_SyntheticChoice(content)]
        self.usage = None
        self.compliance: dict[str, Any] = {}
        self._agentguard: dict[str, Any] = {}


def _extract_input(messages: list[dict[str, Any]]) -> str:
    """Extract user input text from the messages list.

    Finds the last user message. Falls back to joining all message
    contents if no user message is found.
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            # Handle content arrays (e.g. vision messages with text parts)
            if isinstance(content, list):
                parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                return " ".join(parts)
    # Fallback: join all message contents
    texts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str) and content:
            texts.append(content)
    return " ".join(texts) if texts else ""


class ComplianceStream:
    """Wrapper around an OpenAI Stream that yields chunks unchanged
    while accumulating the full response for compliance logging.

    After iteration completes, ``compliance`` contains the structured
    compliance metadata from ``guard.invoke()``.
    """

    def __init__(
        self,
        stream: Any,
        guard: AgentGuard,
        input_text: str,
        model: str | None,
        user_id: str | None,
        metadata: dict[str, Any] | None,
    ):
        self._stream = stream
        self._guard = guard
        self._input_text = input_text
        self._model = model
        self._user_id = user_id
        self._metadata = metadata
        self.compliance: dict[str, Any] | None = None
        self._agentguard: dict[str, Any] | None = None
        self.compliance_headers: dict[str, str] | None = None

    def __iter__(self) -> Iterator[Any]:
        accumulated: list[str] = []
        try:
            for chunk in self._stream:
                # Accumulate content deltas
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        accumulated.append(delta.content)
                yield chunk
        finally:
            # Run compliance pipeline with accumulated response
            full_response = "".join(accumulated)
            result = self._guard.invoke(
                func=lambda _: full_response,
                input_text=self._input_text,
                model=self._model,
                user_id=self._user_id,
                metadata=self._metadata,
            )
            self.compliance = result.get("compliance", {})
            self._agentguard = self.compliance
            self.compliance_headers = self.compliance.get("http_headers", {})


def wrap_openai(client: Any, guard: AgentGuard, **defaults: Any) -> Any:
    """Wrap an OpenAI client so every chat.completions.create() call
    is automatically EU AI Act compliant.

    Args:
        client: An ``openai.OpenAI`` instance.
        guard: An ``AgentGuard`` instance with your compliance config.
        **defaults: Default kwargs forwarded to ``guard.invoke()``
            (e.g. ``user_id``, ``metadata``).

    Returns:
        The same client instance, with ``chat.completions.create``
        monkey-patched. The original response object is preserved;
        compliance metadata is attached as ``response.compliance``.
    """
    original_create = client.chat.completions.create
    default_user_id = defaults.get("user_id")
    default_metadata = defaults.get("metadata")

    def patched_create(*args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        if args:
            messages = args[0]
        model = kwargs.get("model")
        stream = kwargs.get("stream", False)
        user_id = kwargs.pop("user_id", default_user_id)
        ag_metadata = kwargs.pop("ag_metadata", default_metadata)
        input_text = _extract_input(list(messages))

        if stream:
            # Pre-check input policy before starting the stream
            if guard._input_policy:
                ip_result = guard._input_policy.check(input_text)
                if ip_result.blocked:
                    result = guard.invoke(
                        func=lambda _: "",
                        input_text=input_text,
                        model=model,
                        user_id=user_id,
                        metadata=ag_metadata,
                    )
                    resp = _SyntheticResponse(content=result["response"], model=model)
                    resp.compliance = result.get("compliance", {})
                    resp._agentguard = resp.compliance
                    return resp

            raw_stream = original_create(*args, **kwargs)
            return ComplianceStream(
                stream=raw_stream,
                guard=guard,
                input_text=input_text,
                model=model,
                user_id=user_id,
                metadata=ag_metadata,
            )

        # Non-streaming: capture the response inside invoke's func
        captured_response = None

        def call_and_capture(_input_text: str) -> str:
            nonlocal captured_response
            captured_response = original_create(*args, **kwargs)
            content = captured_response.choices[0].message.content
            return content if content is not None else ""

        result = guard.invoke(
            func=call_and_capture,
            input_text=input_text,
            model=model,
            user_id=user_id,
            metadata=ag_metadata,
        )

        # If input policy blocked the request, func was never called
        if captured_response is None:
            captured_response = _SyntheticResponse(
                content=result["response"],
                model=model,
            )
        elif result["response"] != result["raw_response"]:
            # Only modify content when disclosure was actually prepended
            # (PREPEND/FIRST_ONLY modes). In METADATA/HEADER/NONE modes
            # response == raw_response so content stays untouched.
            captured_response.choices[0].message.content = result["response"]

        # Attach structured compliance metadata
        captured_response.compliance = result.get("compliance", {})
        captured_response._agentguard = captured_response.compliance
        captured_response.compliance_headers = captured_response.compliance.get("http_headers", {})

        return captured_response

    client.chat.completions.create = patched_create
    return client
