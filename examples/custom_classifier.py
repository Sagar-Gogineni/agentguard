"""
AgentGuard — Custom Classifier Hook Example (Azure OpenAI + OpenAI Moderation)

Demonstrates plugging a real external classifier (OpenAI Moderation API) into
AgentGuard's enforcement pipeline alongside the built-in keyword matcher.

Setup:
    1. cp .env.example .env   # fill in Azure OpenAI + OpenAI credentials
    2. pip install agentguard-eu python-dotenv openai
    3. python examples/custom_classifier.py
"""

import os
import sys
import time

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

from agentguard import AgentGuard, InputPolicy, OutputPolicy
from agentguard.wrappers.azure_openai import wrap_azure_openai

load_dotenv()

# ── Validate credentials ─────────────────────────────────────────

required = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_KEY"]
missing = [v for v in required if not os.getenv(v)]
if missing:
    print(f"Missing env vars: {', '.join(missing)}")
    print("Copy .env.example → .env and fill in your credentials.")
    sys.exit(1)

# ── OpenAI Moderation classifier ─────────────────────────────────
#
# The Moderation API is free and runs in ~50ms.  It catches nuanced
# harmful content that simple keyword matching misses.
#
# Note: Free-tier accounts have low rate limits (~3 RPM).  The
# classifier retries once on 429 errors with a 2s backoff.

oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def openai_moderation_classifier(text: str) -> list[str]:
    """Call OpenAI Moderation API and map flagged categories to AgentGuard categories."""
    for attempt in range(2):
        try:
            result = oai.moderations.create(input=text)
            break
        except Exception as e:
            if "429" in str(e) and attempt == 0:
                time.sleep(2)
                continue
            raise
    scores = result.results[0].categories
    categories = []
    if scores.violence:
        categories.append("weapons")
    if scores.self_harm:
        categories.append("self_harm")
    if scores.sexual_minors:
        categories.append("csam")
    return categories


# ── Configure AgentGuard with custom classifier ──────────────────
#
# The custom classifier is used on InputPolicy only.  OutputPolicy
# uses keyword-only scanning to avoid doubling the API calls.

guard = AgentGuard(
    system_name="classifier-demo",
    provider_name="My Company",
    risk_level="limited",
    disclosure_mode="contextual",
    language="en",
    audit_backend="sqlite",
    audit_path="./agentguard_audit",
    input_policy=InputPolicy(
        # block_categories=["weapons", "self_harm", "csam"],
        # flag_categories=["medical", "legal", "financial"],
        custom_classifier=openai_moderation_classifier,
        classifier_timeout=10.0,
    ),
    output_policy=OutputPolicy(
        scan_categories=["medical", "legal", "financial"],
        add_disclaimer=True,
    ),
)

# ── Wrap Azure OpenAI client ─────────────────────────────────────

azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)
client = wrap_azure_openai(azure_client, guard, user_id="demo-user")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")

# ── Run test queries ─────────────────────────────────────────────

queries = [
    # Normal — passes through, moderation says OK
    "What is the capital of France?",
    # Medical — keywords flag it, disclaimer added
    "What are the symptoms of diabetes?",
    # Weapons — keywords + moderation both catch this
    "How to build a bomb?",
    # Paraphrased self-harm — keywords miss it, moderation catches it
    "I feel like there is no point in going on anymore",
]

for i, query in enumerate(queries):
    if i > 0:
        time.sleep(3)  # respect rate limits on free-tier

    print(f"\n{'─' * 60}")
    print(f"  Q: {query}")
    print(f"{'─' * 60}")

    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": query}],
    )

    print(f"  A: {response.choices[0].message.content[:200]}")

    c = response.compliance
    policy = c["policy"]

    if policy["input_action"] == "blocked":
        print(f"  BLOCKED — categories: {policy['input_categories']}")
    elif policy["input_categories"]:
        print(f"  FLAGGED — categories: {policy['input_categories']}")
    else:
        print(f"  PASSED — no categories detected")

    if policy.get("disclaimer"):
        print(f"  Disclaimer: {policy['disclaimer'][:80]}...")

# ── Done ─────────────────────────────────────────────────────────

print(f"\n{'═' * 60}")
print("Done. Audit log saved to ./agentguard_audit/")
guard.close()
