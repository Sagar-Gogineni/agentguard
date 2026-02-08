"""
AgentGuard - EU AI Act Compliance Middleware for AI Agents

Make any LLM-powered agent legally deployable in Europe.

Quick Start:
    from agentguard import AgentGuard

    guard = AgentGuard(
        system_name="my-chatbot",
        provider_name="my-provider",
    )

    result = guard.invoke(
        func=my_llm_function,
        input_text="Hello!",
    )
    print(result["response"])  # AI response with compliance applied
"""

from .config import (
    AgentGuardConfig,
    AuditBackend,
    ContentLabel,
    DisclosureMethod,
    HumanEscalation,
    RiskLevel,
)
from .core import AgentGuard, EscalationTriggered
from .disclosure import DISCLOSURE_TEMPLATES, DisclosureManager
from .policy import CustomRule, InputPolicy, OutputPolicy, PolicyAction, PolicyResult
from .taxonomy import CategoryDefinition, DEFAULT_CATEGORIES, DEFAULT_DISCLAIMERS
from .wrappers.anthropic import wrap_anthropic
from .wrappers.azure_openai import wrap_azure_openai
from .wrappers.langchain import AgentGuardCallback
from .wrappers.openai import wrap_openai

__version__ = "0.2.1"

__all__ = [
    "AgentGuard",
    "AgentGuardCallback",
    "AgentGuardConfig",
    "AuditBackend",
    "CategoryDefinition",
    "ContentLabel",
    "CustomRule",
    "DEFAULT_CATEGORIES",
    "DEFAULT_DISCLAIMERS",
    "DISCLOSURE_TEMPLATES",
    "DisclosureManager",
    "DisclosureMethod",
    "EscalationTriggered",
    "HumanEscalation",
    "InputPolicy",
    "OutputPolicy",
    "PolicyAction",
    "PolicyResult",
    "RiskLevel",
    "wrap_anthropic",
    "wrap_azure_openai",
    "wrap_openai",
]
