"""
AgentGuard - EU AI Act Compliance Middleware for AI Agents

Make any LLM-powered agent legally deployable in Europe with 3 lines of code.

Usage:
    from agentguard import AgentGuard

    guard = AgentGuard(
        system_name="my-chatbot",
        provider_name="my-provider",
    )

    # Wrap any function
    result = guard.invoke(
        func=my_llm_call,
        input_text="Hello, how are you?",
        user_id="user-123",
    )

    # Or use as decorator
    @guard.compliant
    def my_agent(query: str) -> str:
        return openai_client.chat.completions.create(...)
"""

from __future__ import annotations

import functools
import hashlib
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator

from .audit import AuditEntry, AuditLogger
from .config import (
    AgentGuardConfig,
    AuditBackend,
    DisclosureMethod,
    HumanEscalation,
    RiskLevel,
)
from .disclosure import DisclosureManager
from .human_loop import HumanOversight
from .policy import InputPolicy, OutputPolicy
from .report import ComplianceReporter


class EscalationTriggered(Exception):
    """Raised when human escalation is triggered and no callback is set."""

    def __init__(self, reason: str, interaction_id: str):
        self.reason = reason
        self.interaction_id = interaction_id
        super().__init__(f"Human escalation triggered: {reason} (interaction: {interaction_id})")


class AgentGuard:
    """
    EU AI Act compliance middleware for AI agents.

    Wraps any LLM call or AI agent with:
    - Audit logging (Article 12)
    - User disclosure (Article 50)
    - Content labeling (Article 50(2))
    - Human oversight (Article 14)
    - Compliance reporting (Articles 11, 18)

    Example:
        guard = AgentGuard(
            system_name="customer-bot",
            provider_name="my-provider",
            risk_level="limited",
        )

        result = guard.invoke(
            func=lambda q: openai.chat(q),
            input_text="What is your return policy?",
        )
    """

    def __init__(
        self,
        system_name: str,
        provider_name: str,
        risk_level: str | RiskLevel = RiskLevel.LIMITED,
        intended_purpose: str = "",
        # Transparency
        disclosure_method: str | DisclosureMethod = DisclosureMethod.METADATA,
        disclosure_text: str | None = None,
        label_content: bool = True,
        # Audit
        audit_backend: str | AuditBackend = AuditBackend.FILE,
        audit_path: str = "./agentguard_audit",
        log_inputs: bool = True,
        log_outputs: bool = True,
        # Human oversight
        human_escalation: str | HumanEscalation = HumanEscalation.LOW_CONFIDENCE,
        confidence_threshold: float = 0.7,
        sensitive_keywords: list[str] | None = None,
        escalation_callback: Callable | None = None,
        # Behavior
        block_on_escalation: bool = False,
        # Content policies
        input_policy: InputPolicy | None = None,
        output_policy: OutputPolicy | None = None,
    ):
        # Coerce string enums
        if isinstance(risk_level, str):
            risk_level = RiskLevel(risk_level)
        if isinstance(disclosure_method, str):
            disclosure_method = DisclosureMethod(disclosure_method)
        if isinstance(audit_backend, str):
            audit_backend = AuditBackend(audit_backend)
        if isinstance(human_escalation, str):
            human_escalation = HumanEscalation(human_escalation)

        self._config = AgentGuardConfig(
            system_name=system_name,
            provider_name=provider_name,
            risk_level=risk_level,
            intended_purpose=intended_purpose,
            disclosure_method=disclosure_method,
            disclosure_text=disclosure_text
            or AgentGuardConfig.model_fields["disclosure_text"].default,
            label_content=label_content,
            audit_backend=audit_backend,
            audit_path=audit_path,
            log_inputs=log_inputs,
            log_outputs=log_outputs,
            human_escalation=human_escalation,
            confidence_threshold=confidence_threshold,
            sensitive_keywords=sensitive_keywords
            or AgentGuardConfig.model_fields["sensitive_keywords"].default_factory(),
            escalation_callback=escalation_callback,
        )

        self._logger = AuditLogger(
            backend=audit_backend,
            path=audit_path,
        )
        self._disclosure = DisclosureManager(
            system_name=system_name,
            provider_name=provider_name,
            method=disclosure_method,
            disclosure_text=disclosure_text,
        )
        self._oversight = HumanOversight(
            mode=human_escalation,
            confidence_threshold=confidence_threshold,
            sensitive_keywords=self._config.sensitive_keywords,
            escalation_callback=escalation_callback,
        )
        self._reporter = ComplianceReporter(self._config, self._logger)
        self._block_on_escalation = block_on_escalation
        self._input_policy = input_policy
        self._output_policy = output_policy

    # ------------------------------------------------------------------ #
    #  Primary API: invoke()
    # ------------------------------------------------------------------ #

    def invoke(
        self,
        func: Callable[..., str],
        input_text: str,
        *args: Any,
        user_id: str | None = None,
        session_id: str | None = None,
        model: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Invoke an AI function with full EU AI Act compliance.

        Args:
            func: The AI/LLM function to call. Should accept input_text
                  as first arg and return a string.
            input_text: The user's input message.
            user_id: Optional user identifier for audit trail.
            session_id: Optional session identifier.
            model: Name of the underlying model (for labeling).
            confidence: Model confidence score (0.0-1.0).
            metadata: Additional metadata to log.
            *args, **kwargs: Passed through to func.

        Returns:
            Dict with keys:
            - response: The (possibly modified) AI response
            - interaction_id: Unique ID for this interaction
            - disclosure: Disclosure information shown to user
            - escalated: Whether human escalation was triggered
            - audit_entry: The full audit record
            - content_label: Machine-readable content label
        """
        start_time = time.time()
        entry = AuditEntry(
            system_name=self._config.system_name,
            provider_name=self._config.provider_name,
            user_id=user_id,
            session_id=session_id,
            model_used=model,
            input_text=input_text if self._config.log_inputs else None,
            input_hash=hashlib.sha256(input_text.encode()).hexdigest()
            if not self._config.log_inputs
            else None,
        )

        # --- Step 1: Input policy check (BEFORE LLM call) ---
        input_policy_result = None
        if self._input_policy:
            input_policy_result = self._input_policy.check(input_text)
            entry.input_policy_blocked = input_policy_result.blocked
            entry.input_policy_flagged_categories = input_policy_result.matched_categories

            if input_policy_result.blocked:
                entry.latency_ms = (time.time() - start_time) * 1000
                entry.output_text = input_policy_result.safe_response
                self._logger.log(entry)
                return {
                    "response": input_policy_result.safe_response,
                    "raw_response": input_policy_result.safe_response,
                    "interaction_id": entry.interaction_id,
                    "disclosure": self._disclosure.get_http_headers(entry.interaction_id),
                    "escalated": False,
                    "escalation_reason": None,
                    "content_label": None,
                    "latency_ms": entry.latency_ms,
                    "input_policy": {
                        "blocked": True,
                        "reason": input_policy_result.block_reason,
                        "flagged_categories": [],
                        "categories": input_policy_result.matched_categories,
                    },
                    "output_policy": None,
                }

            if input_policy_result.escalate:
                entry.human_escalated = True
                entry.escalation_reason = input_policy_result.escalation_reason
                self._oversight.escalate(
                    interaction_id=entry.interaction_id,
                    input_text=input_text,
                    reason=input_policy_result.escalation_reason,
                )
                if self._block_on_escalation:
                    entry.latency_ms = (time.time() - start_time) * 1000
                    self._logger.log(entry)
                    raise EscalationTriggered(
                        input_policy_result.escalation_reason, entry.interaction_id
                    )

        # --- Step 2: Check for human escalation BEFORE calling LLM ---
        pre_check = self._oversight.check(input_text=input_text, confidence=confidence)
        if pre_check.should_escalate:
            entry.human_escalated = True
            entry.escalation_reason = pre_check.reason
            self._oversight.escalate(
                interaction_id=entry.interaction_id,
                input_text=input_text,
                reason=pre_check.reason,
            )
            if self._block_on_escalation:
                entry.latency_ms = (time.time() - start_time) * 1000
                self._logger.log(entry)
                raise EscalationTriggered(pre_check.reason, entry.interaction_id)

        # --- Step 3: Call the actual AI function ---
        raw_response = ""
        try:
            raw_response = func(input_text, *args, **kwargs)
            if not isinstance(raw_response, str):
                raw_response = str(raw_response)
        except Exception as e:
            entry.error = str(e)
            entry.latency_ms = (time.time() - start_time) * 1000
            self._logger.log(entry)
            raise

        # --- Step 4: Output policy check (AFTER LLM call) ---
        output_policy_result = None
        if self._output_policy:
            output_policy_result = self._output_policy.check(raw_response)
            entry.output_policy_blocked = output_policy_result.blocked
            entry.output_policy_flagged_categories = output_policy_result.matched_categories

            if output_policy_result.blocked:
                raw_response = output_policy_result.safe_response
            elif output_policy_result.flagged_categories:
                raw_response = self._output_policy.apply_disclaimers(
                    raw_response, output_policy_result.flagged_categories
                )
                entry.output_disclaimer_added = True

        # --- Step 5: Check output for escalation ---
        post_check = self._oversight.check(
            input_text=input_text,
            output_text=raw_response,
            confidence=confidence,
        )
        if post_check.should_escalate and not pre_check.should_escalate:
            entry.human_escalated = True
            entry.escalation_reason = post_check.reason
            self._oversight.escalate(
                interaction_id=entry.interaction_id,
                input_text=input_text,
                output_text=raw_response,
                reason=post_check.reason,
            )

        # --- Step 6: Apply disclosure ---
        disclosed_response = self._disclosure.apply_disclosure(
            raw_response, interaction_id=entry.interaction_id
        )
        entry.disclosure_shown = True

        # --- Step 7: Create content label ---
        content_label = None
        if self._config.label_content:
            content_label = self._disclosure.create_content_label(
                model=model, interaction_id=entry.interaction_id
            )
            entry.content_labeled = True

        # --- Step 8: Finalize audit entry ---
        entry.output_text = raw_response if self._config.log_outputs else None
        entry.confidence_score = confidence
        entry.latency_ms = (time.time() - start_time) * 1000
        entry.metadata = metadata or {}

        self._logger.log(entry)

        result = {
            "response": disclosed_response,
            "raw_response": raw_response,
            "interaction_id": entry.interaction_id,
            "disclosure": self._disclosure.get_http_headers(entry.interaction_id),
            "escalated": entry.human_escalated,
            "escalation_reason": entry.escalation_reason,
            "content_label": content_label.model_dump() if content_label else None,
            "latency_ms": entry.latency_ms,
        }

        if self._input_policy and input_policy_result:
            result["input_policy"] = {
                "blocked": input_policy_result.blocked,
                "flagged_categories": input_policy_result.flagged_categories,
                "categories": input_policy_result.matched_categories,
            }
        if self._output_policy and output_policy_result:
            result["output_policy"] = {
                "blocked": output_policy_result.blocked,
                "flagged_categories": output_policy_result.flagged_categories,
                "categories": output_policy_result.matched_categories,
                "disclaimer_added": entry.output_disclaimer_added,
            }

        return result

    # ------------------------------------------------------------------ #
    #  Decorator API: @guard.compliant
    # ------------------------------------------------------------------ #

    def compliant(
        self,
        func: Callable | None = None,
        *,
        model: str | None = None,
    ) -> Callable:
        """
        Decorator to make any AI function EU AI Act compliant.

        The decorated function should accept a string input as its
        first argument and return a string response.

        Example:
            @guard.compliant
            def ask_ai(query: str) -> str:
                return openai_client.chat(query)

            result = ask_ai("What is your return policy?")
            # result is a dict with response, interaction_id, etc.

            @guard.compliant(model="gpt-4")
            def ask_gpt(query: str) -> str:
                return openai_client.chat(query)
        """

        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(input_text: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
                # Extract agentguard-specific kwargs
                ag_kwargs = {}
                for key in ["user_id", "session_id", "confidence", "metadata"]:
                    if key in kwargs:
                        ag_kwargs[key] = kwargs.pop(key)

                return self.invoke(
                    func=lambda q: fn(q, *args, **kwargs),
                    input_text=input_text,
                    model=model,
                    **ag_kwargs,
                )

            return wrapper

        if func is not None:
            return decorator(func)
        return decorator

    # ------------------------------------------------------------------ #
    #  Context Manager API
    # ------------------------------------------------------------------ #

    @contextmanager
    def interaction(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> Generator[InteractionContext, None, None]:
        """
        Context manager for manual compliance tracking.

        Example:
            with guard.interaction(user_id="u-123") as ctx:
                response = my_llm.invoke("Hello")
                ctx.record(input_text="Hello", output_text=response)
        """
        ctx = InteractionContext(
            guard=self,
            user_id=user_id,
            session_id=session_id,
        )
        try:
            yield ctx
        finally:
            ctx._finalize()

    # ------------------------------------------------------------------ #
    #  Reporting
    # ------------------------------------------------------------------ #

    def generate_report(self, path: str | None = None) -> str:
        """Generate and save a compliance report. Returns the file path."""
        return str(self._reporter.save_report(path))

    def generate_report_markdown(self) -> str:
        """Generate a Markdown compliance report."""
        return self._reporter.generate_markdown()

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def config(self) -> AgentGuardConfig:
        return self._config

    @property
    def audit(self) -> AuditLogger:
        return self._logger

    @property
    def oversight(self) -> HumanOversight:
        return self._oversight

    @property
    def pending_reviews(self) -> list[dict[str, Any]]:
        return self._oversight.get_pending_reviews()

    def close(self) -> None:
        """Clean up resources."""
        self._logger.close()


class InteractionContext:
    """Context for manual interaction tracking within a `with` block."""

    def __init__(
        self,
        guard: AgentGuard,
        user_id: str | None = None,
        session_id: str | None = None,
    ):
        self._guard = guard
        self._user_id = user_id
        self._session_id = session_id
        self._entry = AuditEntry(
            system_name=guard._config.system_name,
            provider_name=guard._config.provider_name,
            user_id=user_id,
            session_id=session_id,
        )
        self._start_time = time.time()
        self._recorded = False

    def record(
        self,
        input_text: str,
        output_text: str,
        model: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record an interaction within this context."""
        self._recorded = True
        self._entry.input_text = input_text if self._guard._config.log_inputs else None
        self._entry.output_text = output_text if self._guard._config.log_outputs else None
        self._entry.model_used = model
        self._entry.confidence_score = confidence
        self._entry.metadata = metadata or {}

        # Policy checks (informational — can't block retroactively in context manager)
        if self._guard._input_policy:
            ip_result = self._guard._input_policy.check(input_text)
            self._entry.input_policy_blocked = ip_result.blocked
            self._entry.input_policy_flagged_categories = ip_result.matched_categories
        if self._guard._output_policy:
            op_result = self._guard._output_policy.check(output_text)
            self._entry.output_policy_blocked = op_result.blocked
            self._entry.output_policy_flagged_categories = op_result.matched_categories

        # Check escalation
        check = self._guard._oversight.check(
            input_text=input_text,
            output_text=output_text,
            confidence=confidence,
        )
        if check.should_escalate:
            self._entry.human_escalated = True
            self._entry.escalation_reason = check.reason

        # Apply disclosure
        self._entry.disclosure_shown = True
        self._entry.content_labeled = self._guard._config.label_content

        return {
            "interaction_id": self._entry.interaction_id,
            "escalated": self._entry.human_escalated,
            "disclosure_headers": self._guard._disclosure.get_http_headers(
                self._entry.interaction_id
            ),
        }

    def _finalize(self) -> None:
        """Write audit entry when context exits."""
        self._entry.latency_ms = (time.time() - self._start_time) * 1000
        if self._recorded:
            self._guard._logger.log(self._entry)
