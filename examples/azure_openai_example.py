"""
Simple test script for the Azure OpenAI wrapper.

Setup:
    1. Fill in your credentials in .env
    2. pip install openai python-dotenv
    3. python examples/azure_openai_example.py
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

from agentguard import AgentGuard
from agentguard.wrappers.azure_openai import wrap_azure_openai

load_dotenv()

# --- Setup ---
guard = AgentGuard(
    system_name="azure-test-bot",
    provider_name="My Company",
    risk_level="limited",
    audit_backend="file",
    audit_path="./agentguard_audit",
    human_escalation="low_confidence",
    confidence_threshold=0.7,
    sensitive_keywords=["legal", "medical", "financial advice"],
)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)
client = wrap_azure_openai(client, guard, user_id="test-user")

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")

# --- Test 1: Basic non-streaming call ---
print("=" * 50)
print("Test 1: Non-streaming call")
print("=" * 50)

response = client.chat.completions.create(
    model=deployment,
    messages=[{"role": "user", "content": "What is the capital of Germany?"}],
)

print(f"Response: {response.choices[0].message.content}")
print(f"Interaction ID: {response._agentguard['interaction_id']}")
print(f"Escalated: {response._agentguard['escalated']}")
print(f"Latency: {response._agentguard['latency_ms']:.1f}ms")
print(f"Headers: {response._agentguard['disclosure']}")
print()

# --- Test 2: Streaming call ---
print("=" * 50)
print("Test 2: Streaming call")
print("=" * 50)

stream = client.chat.completions.create(
    model=deployment,
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    stream=True,
)

print("Streamed response: ", end="")
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
print(f"Interaction ID: {stream._agentguard['interaction_id']}")
print(f"Escalated: {stream._agentguard['escalated']}")
print()

# --- Test 3: Escalation trigger ---
print("=" * 50)
print("Test 3: Escalation trigger (sensitive keyword)")
print("=" * 50)

response = client.chat.completions.create(
    model=deployment,
    messages=[{"role": "user", "content": "I need legal advice about my contract."}],
)

print(f"Response: {response.choices[0].message.content[:100]}...")
print(f"Escalated: {response._agentguard['escalated']}")
print(f"Reason: {response._agentguard['escalation_reason']}")
print()

# --- Test 4: Compliance report ---
print("=" * 50)
print("Test 4: Compliance report")
print("=" * 50)

print(guard.generate_report_markdown())
