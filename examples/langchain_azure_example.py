"""
Simple test script for the LangChain callback handler with Azure OpenAI.

Setup:
    1. Fill in your Azure credentials in .env
    2. pip install -e ".[all]" langchain-openai
    3. python examples/langchain_azure_example.py
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

from agentguard import AgentGuard
from agentguard.wrappers.langchain import AgentGuardCallback

load_dotenv()

# --- Setup ---
guard = AgentGuard(
    system_name="langchain-azure-bot",
    provider_name="My Company",
    risk_level="limited",
    audit_backend="file",
    audit_path="./agentguard_audit",
    human_escalation="low_confidence",
    confidence_threshold=0.7,
    sensitive_keywords=["legal", "medical", "financial advice"],
)

callback = AgentGuardCallback(guard, user_id="test-user")

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    callbacks=[callback],
)

# --- Test 1: Basic invoke ---
print("=" * 50)
print("Test 1: Basic invoke")
print("=" * 50)

response = llm.invoke("What is the capital of Germany?")

print(f"Response: {response.content}")
print(f"Interaction ID: {callback.last_result['interaction_id']}")
print(f"Escalated: {callback.last_result['escalated']}")
print(f"Latency: {callback.last_result['latency_ms']:.1f}ms")
print(f"Headers: {callback.last_result['disclosure']}")
print()

# --- Test 2: Streaming ---
print("=" * 50)
print("Test 2: Streaming")
print("=" * 50)

print("Streamed response: ", end="")
for chunk in llm.stream("Count from 1 to 5."):
    print(chunk.content, end="", flush=True)
print()
print(f"Interaction ID: {callback.last_result['interaction_id']}")
print(f"Escalated: {callback.last_result['escalated']}")
print()

# --- Test 3: Escalation trigger ---
print("=" * 50)
print("Test 3: Escalation trigger (sensitive keyword)")
print("=" * 50)

response = llm.invoke("I need legal advice about my employment contract.")

print(f"Response: {response.content[:100]}...")
print(f"Escalated: {callback.last_result['escalated']}")
print(f"Reason: {callback.last_result['escalation_reason']}")
print()

# --- Test 4: Multiple calls tracked ---
print("=" * 50)
print("Test 4: All runs tracked")
print("=" * 50)

print(f"Total runs tracked: {len(callback.results)}")
for run_id, result in callback.results.items():
    print(f"  {run_id[:8]}... escalated={result['escalated']}")
print()

# --- Test 5: Compliance report ---
print("=" * 50)
print("Test 5: Compliance report")
print("=" * 50)

print(guard.generate_report_markdown())
