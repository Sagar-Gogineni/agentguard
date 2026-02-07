"""
Simple test script for the Anthropic wrapper.

Setup:
    1. Fill in ANTHROPIC_API_KEY in .env
    2. pip install -e ".[all]"
    3. python examples/anthropic_example.py
"""

import os
from dotenv import load_dotenv
from anthropic import Anthropic

from agentguard import AgentGuard
from agentguard.wrappers.anthropic import wrap_anthropic

load_dotenv()

# --- Setup ---
guard = AgentGuard(
    system_name="anthropic-test-bot",
    provider_name="My Company",
    risk_level="limited",
    audit_backend="file",
    audit_path="./agentguard_audit",
    human_escalation="low_confidence",
    confidence_threshold=0.7,
    sensitive_keywords=["legal", "medical", "financial advice"],
)

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
client = wrap_anthropic(client, guard, user_id="test-user")

# --- Test 1: Basic non-streaming call ---
print("=" * 50)
print("Test 1: Non-streaming call")
print("=" * 50)

message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=256,
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

print(f"Response: {message.content[0].text}")
print(f"Interaction ID: {message._agentguard['interaction_id']}")
print(f"Escalated: {message._agentguard['escalated']}")
print(f"Latency: {message._agentguard['latency_ms']:.1f}ms")
print(f"Headers: {message._agentguard['disclosure']}")
print()

# --- Test 2: Streaming call ---
print("=" * 50)
print("Test 2: Streaming call")
print("=" * 50)

stream = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=256,
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    stream=True,
)

print("Streamed response: ", end="")
for event in stream:
    if hasattr(event, "type") and event.type == "content_block_delta":
        if hasattr(event.delta, "text"):
            print(event.delta.text, end="", flush=True)
print()
print(f"Interaction ID: {stream._agentguard['interaction_id']}")
print(f"Escalated: {stream._agentguard['escalated']}")
print()

# --- Test 3: Escalation trigger ---
print("=" * 50)
print("Test 3: Escalation trigger (sensitive keyword)")
print("=" * 50)

message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=256,
    messages=[{"role": "user", "content": "I need legal advice about my contract."}],
)

print(f"Response: {message.content[0].text[:100]}...")
print(f"Escalated: {message._agentguard['escalated']}")
print(f"Reason: {message._agentguard['escalation_reason']}")
print()

# --- Test 4: Compliance report ---
print("=" * 50)
print("Test 4: Compliance report")
print("=" * 50)

print(guard.generate_report_markdown())
