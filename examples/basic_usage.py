"""
AgentGuard - Basic Usage Example

Shows the three main ways to use AgentGuard:
1. guard.invoke() - wrap any function call
2. @guard.compliant - decorator
3. guard.interaction() - context manager
"""

from agentguard import AgentGuard


# ------------------------------------------------------------------ #
# Setup - same for all approaches
# ------------------------------------------------------------------ #

guard = AgentGuard(
    system_name="customer-support-bot",
    provider_name="my-provider",
    risk_level="limited",  # Article 50 transparency obligations
    intended_purpose="Answer customer questions about products and policies",
    audit_backend="sqlite",  # Use SQLite for queryable audit logs
    human_escalation="low_confidence",
    confidence_threshold=0.7,
)


# ------------------------------------------------------------------ #
# Your existing AI function (unchanged)
# ------------------------------------------------------------------ #


def my_llm_call(query: str) -> str:
    """Simulate an LLM call. Replace with your actual LLM client."""
    # In real usage, this would be:
    # return openai_client.chat.completions.create(...).choices[0].message.content
    return f"Thank you for your question about '{query}'. Our return policy allows returns within 30 days."


# ------------------------------------------------------------------ #
# Approach 1: guard.invoke()
# ------------------------------------------------------------------ #

print("=" * 60)
print("Approach 1: guard.invoke()")
print("=" * 60)

result = guard.invoke(
    func=my_llm_call,
    input_text="What is your return policy?",
    user_id="customer-42",
    model="gpt-4",
    confidence=0.92,
)

print(f"Response: {result['response']}")
print(f"Interaction ID: {result['interaction_id']}")
print(f"Escalated: {result['escalated']}")
print(f"HTTP Headers: {result['disclosure']}")
print(f"Content Label: {result['content_label']}")
print()


# ------------------------------------------------------------------ #
# Approach 2: @guard.compliant decorator
# ------------------------------------------------------------------ #

print("=" * 60)
print("Approach 2: @guard.compliant decorator")
print("=" * 60)


@guard.compliant(model="gpt-4")
def ask_support(query: str) -> str:
    """Your AI function, now automatically compliant."""
    return f"Based on our documentation, {query.lower()} - yes, we support that feature."


result = ask_support(
    "Do you offer international shipping?",
    user_id="customer-99",
)
print(f"Response: {result['response']}")
print(f"Interaction ID: {result['interaction_id']}")
print()


# ------------------------------------------------------------------ #
# Approach 3: Context manager
# ------------------------------------------------------------------ #

print("=" * 60)
print("Approach 3: Context manager")
print("=" * 60)

with guard.interaction(user_id="customer-77") as ctx:
    # Make your LLM call however you want
    response = my_llm_call("I need legal advice about a defective product")

    # Record it for compliance
    info = ctx.record(
        input_text="I need legal advice about a defective product",
        output_text=response,
        confidence=0.45,  # Low confidence - will trigger escalation
    )
    print(f"Interaction ID: {info['interaction_id']}")
    print(f"Escalated: {info['escalated']}")  # True - sensitive keyword "legal"


# ------------------------------------------------------------------ #
# Generate compliance report
# ------------------------------------------------------------------ #

print("\n" + "=" * 60)
print("Generating Compliance Report")
print("=" * 60)

# Markdown report
report = guard.generate_report_markdown()
print(report)

# Also save as JSON
path = guard.generate_report("./compliance_report.json")
print(f"\nJSON report saved to: {path}")

# Check pending human reviews
print(f"\nPending human reviews: {len(guard.pending_reviews)}")
for review in guard.pending_reviews:
    print(f"  - {review['interaction_id']}: {review['reason']}")

# Clean up
guard.close()
