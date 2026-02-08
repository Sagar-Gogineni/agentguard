"""
AgentGuard Anthropic Wrapper

Drop-in wrapper for the Anthropic Python client that makes every
messages.create() call EU AI Act compliant.

Usage:
    from agentguard import AgentGuard
    from agentguard.wrappers.anthropic import wrap_anthropic
    from anthropic import Anthropic

    guard = AgentGuard(system_name="my-bot", provider_name="my-provider")
    client = wrap_anthropic(Anthropic(), guard)

    # Every call is now automatically compliant
    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(message.content[0].text)   # untouched LLM output
    print(message.compliance)        # structured compliance metadata
"""

from __future__ import annotations

from typing import Any, Iterator

from ..core import AgentGuard


# ------------------------------------------------------------------ #
#  Synthetic response objects (used when input policy blocks a call)
# ------------------------------------------------------------------ #


class _SyntheticTextBlock:
    """Minimal stand-in for an Anthropic TextBlock."""

    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _SyntheticUsage:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0


class _SyntheticMessage:
    """Lightweight response returned when input policy blocks the call."""

    def __init__(self, content: str, model: str | None = None):
        self.id = "agentguard-blocked"
        self.type = "message"
        self.role = "assistant"
        self.content = [_SyntheticTextBlock(content)]
        self.model = model or "agentguard-blocked"
        self.stop_reason = "end_turn"
        self.stop_sequence = None
        self.usage = _SyntheticUsage()
        self.compliance: dict[str, Any] = {}
        self._agentguard: dict[str, Any] = {}
        self.compliance_headers: dict[str, str] = {}


def _extract_input(messages: list[dict[str, Any]], system: str | None = None) -> str:
    """Extract user input text from the messages list.

    Finds the last user message. Falls back to joining all message
    contents. Prepends the system prompt if provided.
    """
    user_text = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_text = content
                break
            if isinstance(content, list):
                parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                user_text = " ".join(parts)
                break

    if not user_text:
        texts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                texts.append(content)
        user_text = " ".join(texts) if texts else ""

    return user_text


def _extract_text(message: Any) -> str:
    """Extract the text response from an Anthropic Message object.

    Iterates content blocks and joins all text blocks.
    """
    parts: list[str] = []
    for block in message.content:
        if hasattr(block, "type") and block.type == "text":
            parts.append(block.text)
    return "\n".join(parts) if parts else ""


class ComplianceStream:
    """Wrapper around an Anthropic raw event stream (``stream=True``)
    that yields events unchanged while accumulating the full text
    response for compliance logging.

    After iteration completes, ``_agentguard`` contains the compliance
    metadata from ``guard.invoke()``.
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
            for event in self._stream:
                # Accumulate text deltas from content_block_delta events
                if hasattr(event, "type") and event.type == "content_block_delta":
                    delta = event.delta
                    if hasattr(delta, "type") and delta.type == "text_delta":
                        accumulated.append(delta.text)
                yield event
        finally:
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


def wrap_anthropic(client: Any, guard: AgentGuard, **defaults: Any) -> Any:
    """Wrap an Anthropic client so every messages.create() call
    is automatically EU AI Act compliant.

    Args:
        client: An ``anthropic.Anthropic`` instance.
        guard: An ``AgentGuard`` instance with your compliance config.
        **defaults: Default kwargs forwarded to ``guard.invoke()``
            (e.g. ``user_id``, ``metadata``).

    Returns:
        The same client instance, with ``messages.create``
        monkey-patched. The original response object is preserved;
        compliance metadata is attached as ``message.compliance``.
    """
    original_create = client.messages.create
    default_user_id = defaults.get("user_id")
    default_metadata = defaults.get("metadata")

    def patched_create(*args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model")
        system = kwargs.get("system")
        stream = kwargs.get("stream", False)
        user_id = kwargs.pop("user_id", default_user_id)
        ag_metadata = kwargs.pop("ag_metadata", default_metadata)
        input_text = _extract_input(list(messages), system=system)

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
                    resp = _SyntheticMessage(content=result["response"], model=model)
                    resp.compliance = result.get("compliance", {})
                    resp._agentguard = resp.compliance
                    resp.compliance_headers = resp.compliance.get("http_headers", {})
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
            return _extract_text(captured_response)

        result = guard.invoke(
            func=call_and_capture,
            input_text=input_text,
            model=model,
            user_id=user_id,
            metadata=ag_metadata,
        )

        # If input policy blocked the request, func was never called
        if captured_response is None:
            captured_response = _SyntheticMessage(
                content=result["response"],
                model=model,
            )

        # Attach structured compliance metadata
        captured_response.compliance = result.get("compliance", {})
        captured_response._agentguard = captured_response.compliance
        captured_response.compliance_headers = captured_response.compliance.get("http_headers", {})

        return captured_response

    client.messages.create = patched_create
    return client
