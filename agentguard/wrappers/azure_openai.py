"""
AgentGuard Azure OpenAI Wrapper

Drop-in wrapper for the Azure OpenAI Python client that makes every
chat.completions.create() call EU AI Act compliant.

Azure OpenAI uses the same ``openai`` SDK with ``AzureOpenAI`` instead
of ``OpenAI``.  The ``chat.completions.create()`` interface is identical,
so this module delegates to the OpenAI wrapper.

Usage:
    from agentguard import AgentGuard
    from agentguard.wrappers.azure_openai import wrap_azure_openai
    from openai import AzureOpenAI

    guard = AgentGuard(system_name="my-bot", provider_name="my-provider")
    client = wrap_azure_openai(
        AzureOpenAI(
            azure_endpoint="https://my-resource.openai.azure.com",
            api_version="2024-02-01",
            api_key="...",
        ),
        guard,
    )

    # Every call is now automatically compliant
    response = client.chat.completions.create(
        model="gpt-4",  # Azure deployment name
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response.choices[0].message.content)  # unchanged
    print(response._agentguard)                 # compliance metadata
"""

from __future__ import annotations

from typing import Any

from ..core import AgentGuard
from .openai import wrap_openai


def wrap_azure_openai(client: Any, guard: AgentGuard, **defaults: Any) -> Any:
    """Wrap an Azure OpenAI client so every chat.completions.create()
    call is automatically EU AI Act compliant.

    This is functionally identical to ``wrap_openai`` because the
    ``AzureOpenAI`` client exposes the same ``chat.completions.create``
    interface as the standard ``OpenAI`` client.

    Args:
        client: An ``openai.AzureOpenAI`` instance.
        guard: An ``AgentGuard`` instance with your compliance config.
        **defaults: Default kwargs forwarded to ``guard.invoke()``
            (e.g. ``user_id``, ``metadata``).

    Returns:
        The same client instance, with ``chat.completions.create``
        monkey-patched.  Compliance metadata is attached as
        ``response._agentguard``.
    """
    return wrap_openai(client, guard, **defaults)
