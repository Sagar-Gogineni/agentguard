"""
AgentGuard Content Policy

Pre-call and post-call content enforcement for AI interactions.
Uses lightweight keyword/pattern matching for fast classification.
Supports pluggable custom classifiers (Azure Content Safety, OpenAI Moderation, etc.).
"""

from __future__ import annotations

import logging
import re
import signal
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .taxonomy import DEFAULT_CATEGORIES, DEFAULT_DISCLAIMERS, CategoryDefinition

logger = logging.getLogger(__name__)


class PolicyAction(str, Enum):
    """Action to take when a policy rule matches."""

    BLOCK = "block"
    FLAG = "flag"
    ESCALATE = "escalate"


@dataclass
class CustomRule:
    """A user-defined regex rule with an associated action."""

    name: str
    pattern: str
    action: PolicyAction
    message: str = ""


@dataclass
class PolicyResult:
    """Result of a policy check against content."""

    allowed: bool = True
    blocked: bool = False
    block_reason: str | None = None
    flagged_categories: list[str] = field(default_factory=list)
    matched_categories: list[str] = field(default_factory=list)
    escalate: bool = False
    escalation_reason: str | None = None
    custom_rule_matches: list[str] = field(default_factory=list)
    safe_response: str | None = None


def _run_custom_classifier(
    classifier: Callable[[str], list[str]] | None,
    text: str,
    timeout: float,
) -> list[str]:
    """Run a custom classifier with timeout and error handling.

    Returns the list of category strings from the classifier, or an empty
    list if the classifier is ``None``, crashes, or exceeds the timeout.
    """
    if classifier is None:
        return []

    result: list[str] = []
    error: BaseException | None = None

    def _target() -> None:
        nonlocal result, error
        try:
            result = classifier(text)
        except Exception as exc:  # noqa: BLE001
            error = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        logger.warning(
            "Custom classifier timed out after %.1fs — skipping.",
            timeout,
        )
        return []

    if error is not None:
        logger.warning(
            "Custom classifier raised %s: %s — skipping.",
            type(error).__name__,
            error,
        )
        return []

    if not isinstance(result, list):
        logger.warning(
            "Custom classifier returned %s instead of list — skipping.",
            type(result).__name__,
        )
        return []

    return result


def _compile_category(cat_def: CategoryDefinition) -> re.Pattern[str] | None:
    """Compile a CategoryDefinition into a single regex pattern."""
    parts: list[str] = []
    for kw in cat_def.keywords:
        parts.append(r"\b" + re.escape(kw) + r"\b")
    parts.extend(cat_def.patterns)
    if parts:
        return re.compile("|".join(parts), re.IGNORECASE)
    return None


class InputPolicy:
    """Content policy for user inputs — runs BEFORE the LLM call.

    Args:
        block_categories: Categories that reject input (LLM never called).
        flag_categories: Categories that tag input but allow it through.
        custom_rules: Additional regex rules with block/flag/escalate actions.
        max_input_length: Maximum allowed input length (0 = unlimited).
        categories: Custom category definitions (merged with/overriding defaults).
        safe_refusal: Default refusal message when input is blocked.
        custom_classifier: Optional callable that takes a string and returns a list
            of category name strings.  Works with Azure Content Safety, OpenAI
            Moderation, Llama Guard, or any custom rules function.
        classifier_timeout: Maximum seconds to wait for the custom classifier
            before skipping it (default 5.0).
    """

    def __init__(
        self,
        block_categories: list[str] | None = None,
        flag_categories: list[str] | None = None,
        custom_rules: list[CustomRule] | None = None,
        max_input_length: int = 0,
        categories: dict[str, CategoryDefinition] | None = None,
        safe_refusal: str = (
            "I'm unable to process this request as it violates our content policy."
        ),
        custom_classifier: Callable[[str], list[str]] | None = None,
        classifier_timeout: float = 5.0,
    ):
        self.block_categories = block_categories or []
        self.flag_categories = flag_categories or []
        self.custom_rules = custom_rules or []
        self.max_input_length = max_input_length
        self.safe_refusal = safe_refusal
        self.custom_classifier = custom_classifier
        self.classifier_timeout = classifier_timeout

        # Merge user categories on top of defaults
        self._categories: dict[str, CategoryDefinition] = dict(DEFAULT_CATEGORIES)
        if categories:
            self._categories.update(categories)

        # Pre-compile patterns per category
        self._compiled: dict[str, re.Pattern[str] | None] = {}
        for name, cat_def in self._categories.items():
            self._compiled[name] = _compile_category(cat_def)

        # Pre-compile custom rules
        self._compiled_rules: list[tuple[CustomRule, re.Pattern[str]]] = [
            (rule, re.compile(rule.pattern, re.IGNORECASE)) for rule in self.custom_rules
        ]

    def check(self, text: str) -> PolicyResult:
        """Check input text against the policy."""
        result = PolicyResult()

        # Check max length
        if self.max_input_length > 0 and len(text) > self.max_input_length:
            result.allowed = False
            result.blocked = True
            result.block_reason = (
                f"Input exceeds maximum length ({len(text)} > {self.max_input_length})"
            )
            result.safe_response = self.safe_refusal
            return result

        # --- Step A: Built-in keyword/regex matching ---
        keyword_categories: list[str] = []
        all_relevant = set(self.block_categories) | set(self.flag_categories)
        for cat_name in all_relevant:
            pattern = self._compiled.get(cat_name)
            if pattern and pattern.search(text):
                keyword_categories.append(cat_name)

        # --- Step B: Custom classifier ---
        custom_categories = _run_custom_classifier(
            self.custom_classifier, text, self.classifier_timeout
        )

        # --- Step C: Merge categories ---
        all_categories = list(dict.fromkeys(keyword_categories + custom_categories))

        # --- Step D: Apply block/flag rules using merged categories ---
        for cat_name in self.block_categories:
            if cat_name in all_categories:
                result.allowed = False
                result.blocked = True
                result.block_reason = f"Input matched blocked category: {cat_name}"
                result.matched_categories.append(cat_name)
                result.safe_response = self.safe_refusal
                return result

        for cat_name in self.flag_categories:
            if cat_name in all_categories:
                result.flagged_categories.append(cat_name)
                result.matched_categories.append(cat_name)

        # Also record any custom classifier categories that aren't in block/flag lists
        for cat_name in all_categories:
            if cat_name not in result.matched_categories:
                result.matched_categories.append(cat_name)

        # Check custom rules
        for rule, pattern in self._compiled_rules:
            if pattern.search(text):
                result.custom_rule_matches.append(rule.name)
                if rule.action == PolicyAction.BLOCK:
                    result.allowed = False
                    result.blocked = True
                    result.block_reason = f"Input matched custom rule: {rule.name}"
                    result.safe_response = rule.message or self.safe_refusal
                    return result
                elif rule.action == PolicyAction.FLAG:
                    result.flagged_categories.append(f"custom:{rule.name}")
                    result.matched_categories.append(f"custom:{rule.name}")
                elif rule.action == PolicyAction.ESCALATE:
                    result.escalate = True
                    result.escalation_reason = f"Custom rule escalation: {rule.name}"

        return result


class OutputPolicy:
    """Content policy for AI outputs — runs AFTER the LLM call.

    Args:
        scan_categories: Categories to check in output.
        block_on_detect: If True, replace output with safe message on match.
        add_disclaimer: If True, append category-specific disclaimer on match.
        disclaimers: Custom disclaimer text per category (merged with defaults).
        categories: Custom category definitions (merged with/overriding defaults).
        safe_message: Replacement message when output is blocked.
        custom_classifier: Optional callable that takes a string and returns a list
            of category name strings for post-call scanning.
        classifier_timeout: Maximum seconds to wait for the custom classifier
            before skipping it (default 5.0).
    """

    def __init__(
        self,
        scan_categories: list[str] | None = None,
        block_on_detect: bool = False,
        add_disclaimer: bool = True,
        disclaimers: dict[str, str] | None = None,
        categories: dict[str, CategoryDefinition] | None = None,
        safe_message: str = (
            "I'm unable to provide this response as it may violate our content policy."
        ),
        custom_classifier: Callable[[str], list[str]] | None = None,
        classifier_timeout: float = 5.0,
    ):
        self.scan_categories = scan_categories or []
        self.block_on_detect = block_on_detect
        self.add_disclaimer = add_disclaimer
        self.safe_message = safe_message
        self.custom_classifier = custom_classifier
        self.classifier_timeout = classifier_timeout

        self._disclaimers: dict[str, str] = dict(DEFAULT_DISCLAIMERS)
        if disclaimers:
            self._disclaimers.update(disclaimers)

        self._categories: dict[str, CategoryDefinition] = dict(DEFAULT_CATEGORIES)
        if categories:
            self._categories.update(categories)

        self._compiled: dict[str, re.Pattern[str] | None] = {}
        for name, cat_def in self._categories.items():
            self._compiled[name] = _compile_category(cat_def)

    def check(self, text: str) -> PolicyResult:
        """Check output text against the policy."""
        result = PolicyResult()

        # --- Step A: Built-in keyword/regex scan ---
        keyword_detected: list[str] = []
        for cat_name in self.scan_categories:
            pattern = self._compiled.get(cat_name)
            if pattern and pattern.search(text):
                keyword_detected.append(cat_name)

        # --- Step B: Custom classifier ---
        custom_categories = _run_custom_classifier(
            self.custom_classifier, text, self.classifier_timeout
        )

        # --- Step C: Merge — only keep categories in scan_categories ---
        all_detected = list(dict.fromkeys(
            keyword_detected + [c for c in custom_categories if c in self.scan_categories]
        ))

        result.matched_categories = list(all_detected)

        if not all_detected:
            return result

        if self.block_on_detect:
            result.allowed = False
            result.blocked = True
            result.block_reason = f"Output matched blocked categories: {', '.join(all_detected)}"
            result.safe_response = self.safe_message
            return result

        result.flagged_categories = all_detected
        return result

    def get_disclaimer_text(self, categories: list[str]) -> str | None:
        """Return combined disclaimer text for the given categories.

        Unlike :meth:`apply_disclaimers`, this does **not** modify any
        response string — it just returns the raw disclaimer text (or
        ``None`` when nothing applies).
        """
        if not self.add_disclaimer or not categories:
            return None
        parts: list[str] = []
        added: set[str] = set()
        for cat in categories:
            disclaimer = self._disclaimers.get(cat)
            if disclaimer and cat not in added:
                parts.append(disclaimer)
                added.add(cat)
        return "".join(parts) if parts else None

    def apply_disclaimers(self, text: str, categories: list[str]) -> str:
        """Append disclaimers for the given categories to the text."""
        if not self.add_disclaimer or not categories:
            return text
        added: set[str] = set()
        for cat in categories:
            disclaimer = self._disclaimers.get(cat)
            if disclaimer and cat not in added:
                text += disclaimer
                added.add(cat)
        return text
