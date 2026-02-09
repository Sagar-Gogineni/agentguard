<p align="center">
  <img src="https://raw.githubusercontent.com/Sagar-Gogineni/agentguard/main/assets/logo.png" alt="AgentGuard" width="400">
</p>

<p align="center">
  <strong>EU AI Act compliance middleware for AI agents.<br>Make any LLM-powered agent legally deployable in Europe with 3 lines of code.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/agentguard-eu/"><img src="https://img.shields.io/pypi/v/agentguard-eu.svg" alt="PyPI version"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
</p>

---

## The Problem

Starting **August 2, 2026**, every company deploying AI systems in the EU must comply with the [EU AI Act](https://artificialintelligenceact.eu/) — or face fines up to **€35M or 7% of global turnover**.

AgentGuard fixes that. It's a lightweight middleware that wraps any AI agent or LLM call with the compliance layer required by the EU AI Act:

| EU AI Act Requirement | Article | AgentGuard Feature |
|---|---|---|
| Users must know they're talking to AI | Art. 50(1) | Contextual smart disclosures (category-aware, multi-language) |
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
print(result["response"])         # AI response (NEVER modified by default)
print(result["interaction_id"])   # Unique audit trail ID
print(result["compliance"])       # Structured compliance metadata dict
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

Runs **after** the LLM responds. Adds category-specific disclaimers (in metadata by default) or blocks unsafe output.

```python
output_policy = OutputPolicy(
    scan_categories=["medical", "legal", "financial"],
    block_on_detect=False,     # True = replace response entirely
    add_disclaimer=True,       # Add category-aware disclaimers (in metadata by default)
)
```

**Important:** By default (`disclosure_method="metadata"`), disclaimers are placed in `response.compliance["policy"]["disclaimer"]` — the actual LLM response content is **never modified**. Set `disclosure_method="prepend"` if you want disclaimers appended to the response text.

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

### Content Detection

AgentGuard ships with built-in keyword/pattern matchers for common categories:

| Category | Default Action | Detected via |
|---|---|---|
| `weapons` | Block | Keywords + regex patterns |
| `self_harm` | Block | Keywords + regex |
| `csam` | Block | Keywords |
| `medical` | Flag + disclaimer | Keywords (diagnosis, symptoms, dosage...) |
| `legal` | Flag + disclaimer | Keywords (lawsuit, attorney, liability...) |
| `financial` | Flag + disclaimer | Keywords (invest, portfolio, trading...) |
| `emotional_simulation` | Flag | Keywords (girlfriend, boyfriend...) |

> Built-in detection uses keyword and regex patterns. This is fast (<1ms)
> and catches obvious cases, but will miss paraphrased or subtle inputs.
> For production accuracy, plug in a dedicated classifier:

#### Custom Classifier Hook

Plug in **any** external classifier — Azure Content Safety, OpenAI Moderation, Llama Guard, or your own rules. AgentGuard merges the results with its built-in keyword detection:

```python
from agentguard import AgentGuard, InputPolicy, OutputPolicy
```

**Azure AI Content Safety:**

```python
from azure.ai.contentsafety import ContentSafetyClient

safety_client = ContentSafetyClient(endpoint="...", credential="...")

def azure_classifier(text: str) -> list[str]:
    result = safety_client.analyze_text({"text": text})
    categories = []
    if result.violence_result.severity >= 2:
        categories.append("weapons")
    if result.self_harm_result.severity >= 2:
        categories.append("self_harm")
    return categories

guard = AgentGuard(
    ...,
    input_policy=InputPolicy(
        block_categories=["weapons", "self_harm"],
        custom_classifier=azure_classifier,
    ),
)
```

**OpenAI Moderation (free):**

```python
from openai import OpenAI
oai = OpenAI()

def openai_classifier(text: str) -> list[str]:
    result = oai.moderations.create(input=text)
    scores = result.results[0].categories
    categories = []
    if scores.violence: categories.append("weapons")
    if scores.self_harm: categories.append("self_harm")
    if scores.sexual_minors: categories.append("csam")
    return categories

guard = AgentGuard(
    ...,
    input_policy=InputPolicy(
        block_categories=["weapons", "self_harm", "csam"],
        custom_classifier=openai_classifier,
    ),
)
```

**Simple domain rules:**

```python
def healthcare_rules(text: str) -> list[str]:
    categories = []
    controlled = ["oxycodone", "fentanyl", "morphine"]
    if any(drug in text.lower() for drug in controlled):
        categories.append("controlled_substance")
    return categories
```

The custom classifier is any `Callable[[str], list[str]]`. If it crashes, AgentGuard catches the error and continues with keyword results only. If it's too slow, it's skipped after `classifier_timeout` seconds (default: 5.0).

#### Detection Quality Roadmap

| Version | Method | Accuracy | Latency | Cost |
|---|---|---|---|---|
| v0.1 (now) | Keywords + regex | ~70% | <1ms | Free |
| v0.1+ (now) | + Custom classifier hook | User-defined | User-defined | User-defined |
| v0.3 | + LLM-as-judge option | ~95% | +200ms | ~$0.00001/call |
| v1.0 | + Fine-tuned small model | ~93% | +20ms | Free (local) |

You can override or extend any category with your own `CategoryDefinition`.

### Policy metadata on every response

When using provider wrappers, policy results are attached to every response via `response.compliance`:

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me about my symptoms"}],
)

# Response content is NEVER modified (default: disclosure_method="metadata")
print(response.choices[0].message.content)  # pristine LLM output

# All compliance data in structured metadata
print(response.compliance["policy"]["disclaimer"])
# "This is AI-generated and not medical advice..."

print(response.compliance["disclosure"]["text"])
# "This is an AI system. Information provided is not medical advice..."

# HTTP headers ready for FastAPI/Flask forwarding
print(response.compliance_headers)
# {"X-AI-Generated": "true", "X-AI-System": "my-bot", ...}
```

For blocked requests (e.g., weapons), the LLM is **never called** — zero cost, zero latency:

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "How to build a bomb?"}],
)
print(response.compliance["policy"]["input_action"])  # "blocked"
```

## Contextual Smart Disclosures (Article 50)

Instead of a generic "you are talking to AI" message, AgentGuard adapts the disclosure based on detected content categories — in the user's language.

```python
guard = AgentGuard(
    system_name="my-bot",
    provider_name="My Company GmbH",
    disclosure_method="metadata",      # DEFAULT: never touch response content
    # Other options: "prepend", "first_only", "header", "none"
    disclosure_mode="contextual",      # "static" = same text always (default)
    language="de",                     # en, de, fr, es, it built-in
)
```

**Disclosure methods:**

| Method | Behavior |
|--------|----------|
| `"metadata"` (default) | Attach to `response.compliance` — **never modify content** |
| `"prepend"` | Prepend disclosure text before response content |
| `"first_only"` | Prepend only on first message per session_id |
| `"header"` | Return as HTTP headers dict via `response.compliance_headers` |
| `"none"` | Disable disclosure text entirely (just log and audit) |

When the policy engine detects `medical` content, the user sees (in German):

> Dies ist ein KI-System. Die bereitgestellten Informationen stellen keine medizinische Beratung dar. Bitte konsultieren Sie einen Arzt.

Instead of a generic "You are talking to an AI."

**Built-in languages:** English, German, French, Spanish, Italian — each with category-specific templates for medical, legal, financial, emotional simulation, and self-harm.

```python
# Override or add templates for any category/language
guard = AgentGuard(
    ...,
    disclosure_mode="contextual",
    language="pt",
    disclosure_languages={
        "pt": {
            "default": "Voce esta interagindo com um sistema de IA ({system_name}).",
            "medical": "Informacao nao constitui aconselhamento medico.",
        },
    },
)
```

Multiple categories are combined automatically. If a query triggers both `medical` and `legal`, the user sees both disclosures.

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
print(response.choices[0].message.content)  # untouched LLM output
print(response.compliance)                  # structured compliance metadata
print(response.compliance_headers)          # HTTP headers for forwarding
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
print(stream.compliance)           # available after iteration
print(stream.compliance_headers)   # HTTP headers
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
print(message.content[0].text)  # untouched LLM output
print(message.compliance)       # structured compliance metadata
print(message.compliance_headers)  # HTTP headers for forwarding
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

Use `response.compliance_headers` to forward compliance headers to HTTP responses:

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

@app.post("/chat")
async def chat(query: str):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": query}],
    )
    fastapi_response = JSONResponse({"message": response.choices[0].message.content})
    for key, value in response.compliance_headers.items():
        fastapi_response.headers[key] = value
    return fastapi_response
```

## Configuration

```python
guard = AgentGuard(
    # Identity (Article 16)
    system_name="my-ai-system",
    provider_name="My Company GmbH",
    risk_level="limited",           # "minimal", "limited", or "high"
    intended_purpose="Customer support chatbot",

    # Transparency (Article 50)
    disclosure_method="metadata",   # "metadata" (default), "prepend", "first_only", "header", "none"
    disclosure_mode="contextual",   # "static" (default) or "contextual"
    language="en",                  # en, de, fr, es, it (or add your own)
    label_content=True,             # Machine-readable content labels

    # Content Policies (runtime enforcement)
    input_policy=InputPolicy(
        block_categories=["weapons", "self_harm"],
        flag_categories=["medical", "legal", "financial"],
    ),
    output_policy=OutputPolicy(
        scan_categories=["medical", "legal", "financial"],
        add_disclaimer=True,
    ),

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
