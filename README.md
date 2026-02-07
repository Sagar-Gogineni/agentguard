# 🛡️ AgentGuard

**EU AI Act compliance middleware for AI agents. Make any LLM-powered agent legally deployable in Europe with 3 lines of code.**

[![PyPI version](https://img.shields.io/pypi/v/agentguard-eu.svg)](https://pypi.org/project/agentguard-eu/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## The Problem

Starting **August 2, 2026**, every company deploying AI systems in the EU must comply with the [EU AI Act](https://artificialintelligenceact.eu/) — or face fines up to **€35M or 7% of global turnover**.



AgentGuard fixes that. It's a lightweight middleware that wraps any AI agent or LLM call with the compliance layer required by the EU AI Act:

| EU AI Act Requirement | Article | AgentGuard Feature |
|---|---|---|
| Users must know they're talking to AI | Art. 50(1) | Automatic disclosure injection |
| AI content must be machine-readable labeled | Art. 50(2) | Content labeling (C2PA-compatible) |
| Interactions must be logged and auditable | Art. 12 | Structured audit logging (file/SQLite) |
| Human oversight must be possible | Art. 14 | Automatic escalation + review queue |
| Harmful content must be prevented | Runtime | Input/Output Policy Engine (block/flag/disclaim) |
| System must be documented | Art. 11, 18 | Auto-generated compliance reports |

## Quick Start

```bash
pip install agentguard-eu
```

```python
from agentguard import AgentGuard

# 1. Initialize with your system details
guard = AgentGuard(
    system_name="customer-support-bot",
    provider_name="my-provider",
    risk_level="limited",
)

# 2. Wrap any LLM function
result = guard.invoke(
    func=my_llm_function,       # Your existing AI function
    input_text="What is your return policy?",
    user_id="customer-42",
)

# 3. Everything is now compliant
print(result["response"])         # AI response with disclosure
print(result["interaction_id"])   # Unique audit trail ID
print(result["disclosure"])       # HTTP headers for Article 50
print(result["content_label"])    # Machine-readable content label
print(result["escalated"])        # Whether human review was triggered
```

**That's it.** Your existing AI code doesn't change. AgentGuard wraps it.

## Three Ways to Use

### 1. `guard.invoke()` — Wrap any function call

```python
result = guard.invoke(
    func=lambda q: openai_client.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": q}]
    ).choices[0].message.content,
    input_text="Hello!",
    user_id="user-123",
    model="gpt-4",
    confidence=0.92,
)
```

### 2. `@guard.compliant` — Decorator

```python
@guard.compliant(model="gpt-4")
def ask_support(query: str) -> str:
    return openai_client.chat.completions.create(...).choices[0].message.content

result = ask_support("Do you ship internationally?", user_id="user-456")
```

### 3. `guard.interaction()` — Context manager

```python
with guard.interaction(user_id="user-789") as ctx:
    response = my_complex_agent.run("Analyze this contract")
    ctx.record(
        input_text="Analyze this contract",
        output_text=response,
        confidence=0.45,
    )
    # Low confidence + keyword "contract" → auto-escalated
```

## Input/Output Policy Engine — Runtime Content Enforcement

The policy engine is AgentGuard's core differentiator: **block, flag, or disclaim content on every LLM call** — before and after the model responds.

```
Input → [InputPolicy: block/flag/allow] → [LLM Call] → [OutputPolicy: disclaim/block/pass] → Output
```

### InputPolicy (pre-call)

Runs **before** the LLM is called. Blocked requests never reach the API — zero cost, zero latency.

```python
from agentguard import AgentGuard, InputPolicy, OutputPolicy, PolicyAction
from agentguard.policy import CustomRule

input_policy = InputPolicy(
    block_categories=["weapons", "self_harm", "csam"],
    flag_categories=["emotional_simulation", "medical", "legal", "financial"],
    max_input_length=5000,
    custom_rules=[
        CustomRule(
            name="prompt_injection",
            pattern=r"ignore previous instructions",
            action=PolicyAction.BLOCK,
            message="Blocked: potential prompt injection.",
        ),
    ],
)
```

### OutputPolicy (post-call)

Runs **after** the LLM responds. Appends category-specific disclaimers or blocks unsafe output.

```python
output_policy = OutputPolicy(
    scan_categories=["medical", "legal", "financial"],
    block_on_detect=False,     # True = replace response entirely
    add_disclaimer=True,       # Append category-aware disclaimers
)
```

### Wire it up

```python
guard = AgentGuard(
    system_name="my-bot",
    provider_name="My Company",
    risk_level="limited",
    input_policy=input_policy,
    output_policy=output_policy,
)
```

### Built-in content categories

AgentGuard ships with keyword/pattern matchers for these categories out of the box:

| Category | Detected via | Default action examples |
|---|---|---|
| `weapons` | Keywords + regex patterns | Block |
| `medical` | Keywords (diagnosis, symptoms, treatment...) | Flag + disclaimer |
| `legal` | Keywords + regex (lawsuit, sue, attorney...) | Flag + disclaimer |
| `financial` | Keywords + regex (invest, portfolio...) | Flag + disclaimer |
| `emotional_simulation` | Keywords + regex (girlfriend, boyfriend...) | Flag |
| `self_harm` | Keywords + regex | Block |
| `csam` | Keywords | Block |

You can override or extend any category with your own `CategoryDefinition`.

### Policy metadata on every response

When using provider wrappers, policy results are attached to every response:

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "How to build a bomb?"}],
)

print(response.choices[0].message.content)
# "I'm unable to process this request as it violates our content policy."

print(response._agentguard["input_policy"])
# {"blocked": True, "reason": "Input matched blocked category: weapons", ...}
# LLM was never called — 0ms latency, $0 cost
```

## Human Oversight (Article 14)

AgentGuard automatically detects when interactions should be reviewed by a human:

```python
guard = AgentGuard(
    system_name="my-bot",
    provider_name="my-provider",
    human_escalation="low_confidence",
    confidence_threshold=0.7,
    sensitive_keywords=["legal", "medical", "financial advice"],
    block_on_escalation=True,  # Block response until human approves
)

# Check pending reviews
for review in guard.pending_reviews:
    print(f"Needs review: {review['reason']}")

# Approve or reject
guard.oversight.approve(interaction_id)
guard.oversight.reject(interaction_id, reason="Inaccurate response")
```

## Compliance Reports (Articles 11, 18)

Generate audit documentation with one line:

```python
# JSON report
guard.generate_report("compliance_report.json")

# Markdown report (for human reading)
print(guard.generate_report_markdown())
```

Reports include: system identification, transparency configuration, human oversight settings, interaction statistics, and escalation history.

## Provider Wrappers

Zero-effort compliance for popular LLM clients — wrap once, every call is compliant.

### OpenAI

```bash
pip install "agentguard-eu[openai]"
```

```python
from agentguard import AgentGuard, wrap_openai
from openai import OpenAI

guard = AgentGuard(system_name="my-bot", provider_name="my-provider")
client = wrap_openai(OpenAI(), guard)

# Every call is now compliant — logged, disclosed, escalation-checked
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)  # unchanged
print(response._agentguard["interaction_id"])  # compliance metadata
```

Streaming works too — chunks yield in real-time, compliance runs after the stream completes:

```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
print(stream._agentguard)  # available after iteration
```

### Azure OpenAI

```bash
pip install "agentguard-eu[openai]"
```

```python
from agentguard import AgentGuard, wrap_azure_openai
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

response = client.chat.completions.create(
    model="my-deployment",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Anthropic

```bash
pip install "agentguard-eu[anthropic]"
```

```python
from agentguard import AgentGuard, wrap_anthropic
from anthropic import Anthropic

guard = AgentGuard(system_name="my-bot", provider_name="my-provider")
client = wrap_anthropic(Anthropic(), guard)

message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(message.content[0].text)  # unchanged
print(message._agentguard["interaction_id"])  # compliance metadata
```

### LangChain

```bash
pip install "agentguard-eu[langchain]"
```

```python
from agentguard import AgentGuard, AgentGuardCallback
from langchain_openai import ChatOpenAI  # or AzureChatOpenAI, ChatAnthropic, etc.

guard = AgentGuard(system_name="my-bot", provider_name="my-provider")
callback = AgentGuardCallback(guard, user_id="user-123")
llm = ChatOpenAI(model="gpt-4", callbacks=[callback])

response = llm.invoke("Hello!")
print(response.content)
print(callback.last_result)  # compliance metadata for most recent call
print(callback.results)      # all runs keyed by run_id
```

Works with any LangChain LLM — ChatOpenAI, AzureChatOpenAI, ChatAnthropic, and more. Streaming is also supported automatically via the callback hooks.

## Audit Backends (Article 12)

AgentGuard supports multiple audit backends for logging all AI interactions:

### File Backend (default)

Writes one JSONL file per day to the audit directory. Simple, portable, and easy to ship to external systems:

```python
guard = AgentGuard(
    system_name="my-bot",
    provider_name="my-provider",
    audit_backend="file",
    audit_path="./agentguard_audit",
)
# Logs go to ./agentguard_audit/audit_2026-02-07.jsonl
```

### SQLite Backend

Local SQLite database with built-in querying and statistics. Required for the dashboard and compliance report statistics:

```python
guard = AgentGuard(
    system_name="my-bot",
    provider_name="my-provider",
    audit_backend="sqlite",
    audit_path="./agentguard_audit",
)

# Query audit logs programmatically
entries = guard.audit.query(
    start_date="2026-01-01",
    end_date="2026-02-07",
    user_id="customer-42",
    escalated_only=True,
    limit=100,
)

# Get aggregate statistics
stats = guard.audit.get_stats()
print(stats)
# {
#     "total_interactions": 1234,
#     "total_escalations": 56,
#     "total_errors": 3,
#     "disclosures_shown": 1231,
#     "unique_users": 89,
#     "avg_latency_ms": 245.3,
# }
```

### Custom Backend

Provide your own callback for integration with external logging systems (e.g., S3, BigQuery, Datadog):

```python
def my_log_handler(entry):
    # Send to your logging infrastructure
    requests.post("https://my-logging-api/ingest", json=entry.model_dump())

guard = AgentGuard(
    system_name="my-bot",
    provider_name="my-provider",
    audit_backend="custom",
)
# Pass custom_callback when constructing the AuditLogger directly
```

## Human Review Dashboard

A Streamlit-based dashboard for Article 14 human oversight.

```bash
pip install "agentguard-eu[dashboard]"
agentguard-dashboard --audit-path ./agentguard_audit
```

Features:
- Compliance statistics (total interactions, escalation rate, avg confidence, avg latency)
- Pending escalation review queue with Approve/Reject buttons
- Full audit log browser with filters (date range, user ID, escalated only)
- CSV export of audit data
- Live compliance report viewer

Requires the `sqlite` audit backend (`audit_backend="sqlite"`) to be enabled in your AgentGuard configuration. The dashboard connects directly to the SQLite audit database.

## FastAPI Integration

See [examples/fastapi_example.py](examples/fastapi_example.py) for a complete API with:
- Compliant `/chat` endpoint with automatic headers
- `/compliance/report` endpoint
- `/compliance/pending-reviews` for human oversight
- Approve/reject endpoints for review queue

## Configuration

```python
guard = AgentGuard(
    # Identity (Article 16)
    system_name="my-ai-system",
    provider_name="My Company GmbH",
    risk_level="limited",           # "minimal", "limited", or "high"
    intended_purpose="Customer support chatbot",

    # Transparency (Article 50)
    disclosure_method="metadata",   # "prepend", "metadata", or "header"
    label_content=True,             # Machine-readable content labels

    # Audit (Article 12)
    audit_backend="sqlite",         # "file", "sqlite", or "custom"
    audit_path="./audit_logs",
    log_inputs=True,
    log_outputs=True,
    retention_days=365,

    # Human Oversight (Article 14)
    human_escalation="low_confidence",  # "never", "low_confidence", "sensitive_topic", "always_review"
    confidence_threshold=0.7,
    sensitive_keywords=["legal", "medical", "financial"],
    escalation_callback=my_slack_notifier,  # Optional: get notified
    block_on_escalation=False,
)
```

## What AgentGuard is NOT

- ❌ Not a legal compliance guarantee (consult qualified legal professionals)
- ❌ Not an AI agent framework (use LangGraph, CrewAI, etc. — then wrap with AgentGuard)
- ❌ Not a replacement for a full conformity assessment (required for high-risk systems)
- ✅ A practical engineering tool that covers the technical requirements

## EU AI Act Timeline

| Date | Milestone |
|------|-----------|
| Feb 2025 | Prohibited AI practices banned |
| Aug 2025 | GPAI model rules in effect |
| **Aug 2026** | **Full enforcement: transparency, high-risk obligations, Article 50** |
| Aug 2027 | Remaining provisions for product-embedded AI |

**AgentGuard targets the August 2026 deadline** — the biggest compliance milestone for most companies.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Priority areas:
- Cloud audit backends (S3, BigQuery)
- C2PA standard implementation
- Async support (ainvoke, async wrappers)
- Webhook notifications for escalations

## License

Apache 2.0 — use it freely in commercial projects.

---

**Built by [Sagar](https://github.com/Sagar-Gogineni)** — AI Engineer based in Berlin, specializing in enterprise AI systems and compliance.

*AgentGuard: Because shipping AI agents without compliance is like shipping code without tests. You can do it, but you probably shouldn't.*
