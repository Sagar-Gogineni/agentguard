"""
AgentGuard Content Policy

Pre-call and post-call content enforcement for AI interactions.
Uses lightweight keyword/pattern matching for fast classification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .taxonomy import DEFAULT_CATEGORIES, DEFAULT_DISCLAIMERS, CategoryDefinition


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
    ):
        self.block_categories = block_categories or []
        self.flag_categories = flag_categories or []
        self.custom_rules = custom_rules or []
        self.max_input_length = max_input_length
        self.safe_refusal = safe_refusal

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

        # Check block categories (short-circuit on first match)
        for cat_name in self.block_categories:
            pattern = self._compiled.get(cat_name)
            if pattern and pattern.search(text):
                result.allowed = False
                result.blocked = True
                result.block_reason = f"Input matched blocked category: {cat_name}"
                result.matched_categories.append(cat_name)
                result.safe_response = self.safe_refusal
                return result

        # Check flag categories
        for cat_name in self.flag_categories:
            pattern = self._compiled.get(cat_name)
            if pattern and pattern.search(text):
                result.flagged_categories.append(cat_name)
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
    ):
        self.scan_categories = scan_categories or []
        self.block_on_detect = block_on_detect
        self.add_disclaimer = add_disclaimer
        self.safe_message = safe_message

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

        detected: list[str] = []
        for cat_name in self.scan_categories:
            pattern = self._compiled.get(cat_name)
            if pattern and pattern.search(text):
                detected.append(cat_name)
                result.matched_categories.append(cat_name)

        if not detected:
            return result

        if self.block_on_detect:
            result.allowed = False
            result.blocked = True
            result.block_reason = f"Output matched blocked categories: {', '.join(detected)}"
            result.safe_response = self.safe_message
            return result

        result.flagged_categories = detected
        return result

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
