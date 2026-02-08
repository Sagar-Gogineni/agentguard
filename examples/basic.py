"""Basic test: verify content is untouched and compliance dict is populated."""

import os

from dotenv import load_dotenv
from openai import AzureOpenAI

from agentguard import AgentGuard, InputPolicy, CustomRule, PolicyAction, OutputPolicy
from agentguard.wrappers.azure_openai import wrap_azure_openai

load_dotenv()

guard = AgentGuard(
    system_name="my-chatbot",
    provider_name="My Company",
    risk_level="limited",
    disclosure_mode="contextual",
    language="en",
    audit_backend="sqlite",
    audit_path="./agentguard_audit",
    human_escalation="low_confidence",
    confidence_threshold=0.7,
    input_policy=InputPolicy(
        block_categories=["weapons", "self_harm"],
        flag_categories=["medical", "legal"],
        custom_rules=[
            CustomRule(
                name="prompt_injection",
                pattern=r"ignore previous instructions",
                action=PolicyAction.BLOCK,
                message="Blocked: prompt injection detected.",
            ),
        ],
    ),
    output_policy=OutputPolicy(
        scan_categories=["medical", "legal", "financial"],
        add_disclaimer=True,
    ),
)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)
agent_client = wrap_azure_openai(client, guard, user_id="demo-user")

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")

response = agent_client.chat.completions.create(
    model=deployment,
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

# Content is clean — untouched
print("Content:", response.choices[0].message.content)

# All compliance data is in metadata
print("\nCompliance dict:")
print(f"  interaction_id: {response.compliance['interaction_id'][:12]}...")
print(f"  disclosure.text: {response.compliance['disclosure']['text'][:60]}...")
print(f"  disclosure.method: {response.compliance['disclosure']['method']}")
print(f"  policy.input_action: {response.compliance['policy']['input_action']}")
print(f"  escalation.escalated: {response.compliance['escalation']['escalated']}")
print(f"  audit.logged: {response.compliance['audit']['logged']}")
print(f"  content_label.ai_generated: {response.compliance['content_label']['ai_generated']}")

guard.close()
