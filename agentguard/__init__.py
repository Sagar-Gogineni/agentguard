"""
AgentGuard - EU AI Act Compliance Middleware for AI Agents

Make any LLM-powered agent legally deployable in Europe.

Quick Start:
    from agentguard import AgentGuard

    guard = AgentGuard(
        system_name="my-chatbot",
        provider_name="Acme Corp",
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

__version__ = "0.1.0"

__all__ = [
    "AgentGuard",
    "AgentGuardConfig",
    "AuditBackend",
    "ContentLabel",
    "DisclosureMethod",
    "EscalationTriggered",
    "HumanEscalation",
    "RiskLevel",
]
