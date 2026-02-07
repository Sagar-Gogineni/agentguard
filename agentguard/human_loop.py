"""
AgentGuard Human Oversight

Implements human oversight requirements per EU AI Act Article 14.
Provides mechanisms for human-in-the-loop escalation and review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import HumanEscalation


@dataclass
class EscalationResult:
    """Result of an escalation check."""

    should_escalate: bool
    reason: str | None = None
    matched_keywords: list[str] = field(default_factory=list)
    confidence_score: float | None = None


class HumanOversight:
    """
    Human oversight enforcement for AI systems.

    Article 14 requires that high-risk AI systems are designed to be
    effectively overseen by natural persons. This module provides:
    - Automatic escalation based on confidence scores
    - Keyword-based sensitive topic detection
    - Configurable escalation callbacks
    - Review queue management
    """

    def __init__(
        self,
        mode: HumanEscalation = HumanEscalation.LOW_CONFIDENCE,
        confidence_threshold: float = 0.7,
        sensitive_keywords: list[str] | None = None,
        escalation_callback: Callable[[dict[str, Any]], Any] | None = None,
    ):
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.sensitive_keywords = sensitive_keywords or [
            "legal",
            "medical",
            "financial advice",
            "diagnosis",
            "terminate",
            "fire",
            "lawsuit",
            "suicide",
            "emergency",
        ]
        self._escalation_callback = escalation_callback
        self._review_queue: list[dict[str, Any]] = []
        # Pre-compile regex pattern for keyword matching
        if self.sensitive_keywords:
            escaped = [re.escape(kw) for kw in self.sensitive_keywords]
            self._keyword_pattern = re.compile(
                r"\b(" + "|".join(escaped) + r")\b",
                re.IGNORECASE,
            )
        else:
            self._keyword_pattern = None

    def check(
        self,
        input_text: str | None = None,
        output_text: str | None = None,
        confidence: float | None = None,
    ) -> EscalationResult:
        """
        Check whether this interaction should be escalated to a human.

        Args:
            input_text: User's input message
            output_text: AI's generated response
            confidence: Model's confidence score (0.0-1.0)

        Returns:
            EscalationResult with escalation decision and reason
        """
        if self.mode == HumanEscalation.NEVER:
            return EscalationResult(should_escalate=False)

        if self.mode == HumanEscalation.ALWAYS_REVIEW:
            return EscalationResult(
                should_escalate=True,
                reason="All interactions queued for human review (ALWAYS_REVIEW mode)",
            )

        if self.mode == HumanEscalation.LOW_CONFIDENCE:
            if confidence is not None and confidence < self.confidence_threshold:
                return EscalationResult(
                    should_escalate=True,
                    reason=f"Confidence {confidence:.2f} below threshold {self.confidence_threshold}",
                    confidence_score=confidence,
                )

        if self.mode in (
            HumanEscalation.SENSITIVE_TOPIC,
            HumanEscalation.LOW_CONFIDENCE,
        ):
            # Check both input and output for sensitive keywords
            text_to_check = " ".join(filter(None, [input_text, output_text]))
            matched = self._find_sensitive_keywords(text_to_check)
            if matched:
                return EscalationResult(
                    should_escalate=True,
                    reason=f"Sensitive topic detected: {', '.join(matched)}",
                    matched_keywords=matched,
                )

        return EscalationResult(should_escalate=False)

    def _find_sensitive_keywords(self, text: str) -> list[str]:
        """Find sensitive keywords in text."""
        if not text or not self._keyword_pattern:
            return []
        matches = self._keyword_pattern.findall(text)
        return list(set(m.lower() for m in matches))

    def escalate(
        self,
        interaction_id: str,
        input_text: str | None = None,
        output_text: str | None = None,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Escalate an interaction to human review.

        If a callback is configured, it will be called immediately.
        Otherwise, the interaction is added to the internal review queue.
        """
        escalation_record = {
            "interaction_id": interaction_id,
            "input_text": input_text,
            "output_text": output_text,
            "reason": reason,
            "metadata": metadata or {},
            "status": "pending_review",
        }

        if self._escalation_callback:
            self._escalation_callback(escalation_record)
            escalation_record["status"] = "callback_invoked"
        else:
            self._review_queue.append(escalation_record)

        return escalation_record

    def get_pending_reviews(self) -> list[dict[str, Any]]:
        """Get all pending human review items."""
        return [r for r in self._review_queue if r["status"] == "pending_review"]

    def approve(self, interaction_id: str) -> bool:
        """Mark an escalated interaction as approved by human reviewer."""
        for record in self._review_queue:
            if record["interaction_id"] == interaction_id:
                record["status"] = "approved"
                return True
        return False

    def reject(self, interaction_id: str, reason: str = "") -> bool:
        """Mark an escalated interaction as rejected by human reviewer."""
        for record in self._review_queue:
            if record["interaction_id"] == interaction_id:
                record["status"] = "rejected"
                record["rejection_reason"] = reason
                return True
        return False
