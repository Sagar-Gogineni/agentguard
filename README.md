# 🛡️ AgentGuard

**EU AI Act compliance middleware for AI agents. Make any LLM-powered agent legally deployable in Europe with 3 lines of code.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## The Problem

Starting **August 2, 2026**, every company deploying AI systems in the EU must comply with the [EU AI Act](https://artificialintelligenceact.eu/) — or face fines up to **€35M or 7% of global turnover**.

There are hundreds of AI agent frameworks on GitHub. Almost none of them can be legally deployed in Europe.

AgentGuard fixes that. It's a lightweight middleware that wraps any AI agent or LLM call with the compliance layer required by the EU AI Act:

| EU AI Act Requirement | Article | AgentGuard Feature |
|---|---|---|
| Users must know they're talking to AI | Art. 50(1) | Automatic disclosure injection |
| AI content must be machine-readable labeled | Art. 50(2) | Content labeling (C2PA-compatible) |
| Interactions must be logged and auditable | Art. 12 | Structured audit logging (file/SQLite) |
| Human oversight must be possible | Art. 14 | Automatic escalation + review queue |
| System must be documented | Art. 11, 18 | Auto-generated compliance reports |

## Quick Start

```bash
pip install agentguard
```

```python
from agentguard import AgentGuard

# 1. Initialize with your system details
guard = AgentGuard(
    system_name="customer-support-bot",
    provider_name="Acme Corp GmbH",
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

## Human Oversight (Article 14)

AgentGuard automatically detects when interactions should be reviewed by a human:

```python
guard = AgentGuard(
    system_name="my-bot",
    provider_name="Acme Corp",
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
pip install "agentguard[openai]"
```

```python
from agentguard import AgentGuard, wrap_openai
from openai import OpenAI

guard = AgentGuard(system_name="my-bot", provider_name="Acme Corp")
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
pip install "agentguard[openai]"
```

```python
from agentguard import AgentGuard, wrap_azure_openai
from openai import AzureOpenAI

guard = AgentGuard(system_name="my-bot", provider_name="Acme Corp")
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
pip install "agentguard[anthropic]"
```

```python
from agentguard import AgentGuard, wrap_anthropic
from anthropic import Anthropic

guard = AgentGuard(system_name="my-bot", provider_name="Acme Corp")
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
pip install "agentguard[langchain]"
```

```python
from agentguard import AgentGuard, AgentGuardCallback
from langchain_openai import ChatOpenAI  # or AzureChatOpenAI, ChatAnthropic, etc.

guard = AgentGuard(system_name="my-bot", provider_name="Acme Corp")
callback = AgentGuardCallback(guard, user_id="user-123")
llm = ChatOpenAI(model="gpt-4", callbacks=[callback])

response = llm.invoke("Hello!")
print(response.content)
print(callback.last_result)  # compliance metadata for most recent call
print(callback.results)      # all runs keyed by run_id
```

Works with any LangChain LLM — ChatOpenAI, AzureChatOpenAI, ChatAnthropic, and more. Streaming is also supported automatically via the callback hooks.

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
- Dashboard UI for human review queue
- C2PA standard implementation
- Async support (ainvoke, async wrappers)
- Webhook notifications for escalations

## License

Apache 2.0 — use it freely in commercial projects.

---

**Built by [Sagar](https://github.com/sagar)** — AI Engineer based in Berlin, specializing in enterprise AI systems and compliance.

*AgentGuard: Because shipping AI agents without compliance is like shipping code without tests. You can do it, but you probably shouldn't.*
