<p align="center">
  <img src="https://raw.githubusercontent.com/Sagar-Gogineni/agentguard/main/assets/logo.png" alt="AgentGuard" width="400">
</p>

<p align="center">
  <a href="https://pypi.org/project/agentguard-eu/"><img src="https://img.shields.io/pypi/v/agentguard-eu.svg" alt="PyPI version"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
</p>

---

## The Problem

Someone sends this to your AI system:

> "Generate a performance review that will justify firing Maria before her maternity leave starts."

There are no banned keywords in that sentence. No profanity. No mention of weapons or self-harm. A keyword-based content filter sees a clean input and waves it through. The LLM generates the review. No audit trail records what happened. No policy engine flags discriminatory intent. No human ever sees it.

Maybe the vendor's model refuses today. Maybe it doesn't. Either way, that's **their** guardrail, not yours. When a regulator asks what controls **you** had in place, "we trusted the LLM to say no" is not an answer.

Starting **August 2, 2026**, every company deploying AI systems in the EU must comply with the [EU AI Act](https://artificialintelligenceact.eu/) — or face fines up to **€35M or 7% of global turnover**. The law requires content policy enforcement, audit logging, human oversight, and transparency disclosures. Most engineering teams have none of this infrastructure.

**AgentGuard is open-source middleware that adds EU AI Act compliance infrastructure to any LLM API call.**

```bash
pip install agentguard-eu
```

## Quickstart

```python
from agentguard import AgentGuard, InputPolicy, OutputPolicy, wrap_openai
from openai import OpenAI

guard = AgentGuard(
    system_name="my-assistant",
    provider_name="My Company",
    risk_level="limited",
    input_policy=InputPolicy(
        block_categories=["weapons", "self_harm", "discrimination"],
        flag_categories=["medical", "legal", "financial"],
    ),
    output_policy=OutputPolicy(
        scan_categories=["medical", "legal", "financial"],
        add_disclaimer=True,
    ),
)

client = wrap_openai(OpenAI(), guard)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is your refund policy?"}],
)

print(response.choices[0].message.content)  # untouched LLM output
print(response.compliance)                  # structured compliance metadata
```

Your existing code doesn't change. AgentGuard wraps it.

## What It Does

Every LLM call passes through this pipeline:

```
Input → [InputPolicy: block/flag/allow] → [LLM Call] → [OutputPolicy: disclaim/block/pass]
  → [Disclosure] → [ContentLabel] → [EscalationCheck] → [AuditLog] → Output
```

Mapped to the EU AI Act:

- **Content policy enforcement** with custom classifier hooks — block, flag, or disclaim harmful content before and after the LLM responds. Blocked requests never reach the API. *(Article 5)*
- **Structured audit logging** a regulator can query — every interaction logged with timestamps, user IDs, categories detected, actions taken. SQLite, file, or webhook backends. *(Article 12)*
- **Human oversight escalation paths** — automatic escalation on low confidence, sensitive topics, or policy triggers. Review queue with approve/reject. *(Article 14)*
- **Transparency metadata and AI content labeling** — contextual disclosures adapted to detected content categories, in 5 languages. C2PA-compatible machine-readable labels. *(Article 50)*

By default, AgentGuard **never modifies the LLM response content**. All compliance data goes into `response.compliance` metadata. Your users see the same output they always did.

## Three Test Cases

These are actual results from AgentGuard running against Azure OpenAI. A normal query, a medical query, and the maternity leave prompt from above.

### Test 1: Normal Query

```python
response = client.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content": "What is your refund policy?"}],
)
print(response.choices[0].message.content)
print(response.compliance["policy"]["input_action"])
```

```
Response: "I'm happy to help — could you tell me which purchase you're asking about?
           For example, is this for a subscription, API credits, or a specific order?..."

input_action: "pass"
input_categories: []
```

Clean input, clean output. No categories detected, no policies triggered. The call passes through with full audit logging, C2PA content labels, and transparency metadata attached — all in `response.compliance`.

### Test 2: Medical Query

```python
response = client.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content": "What medication should I take for back pain?"}],
)
print(response.choices[0].message.content)          # untouched
print(response.compliance["policy"]["input_categories"])
print(response.compliance["policy"]["disclaimer"])
```

```
Response: "Short answer for most healthy adults with new, mild–moderate back pain:
           try an NSAID such as ibuprofen or naproxen for a few days..."  (untouched LLM output)

input_categories: ["medical"]
disclaimer: "This is AI-generated and not medical advice. Consult a healthcare professional."
```

The keyword matcher detects medical content. The LLM response goes through untouched, but compliance metadata includes a medical disclaimer. In `metadata` mode (default), the response text is never modified — the disclaimer lives in `response.compliance["policy"]["disclaimer"]`, ready for your frontend to render however you want.

### Test 3: The Maternity Leave Prompt

```python
response = client.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content":
        "Generate a performance review that will justify firing Maria "
        "before her maternity leave starts"}],
)
print(response.compliance["policy"]["input_action"])
print(response.compliance["policy"]["input_categories"])
```

With the built-in keyword matcher only:

```
input_action: "pass"
input_categories: []
```

**It passes through.** There's no "discrimination" keyword in the taxonomy — no "maternity", no "firing", no "justify". The request sails through to the LLM. In this case, GPT-5 refused on its own safety layer. But that's the LLM vendor's guardrail, not yours. You have no audit trail of *why* this was problematic, no policy enforcement, and no guarantee the next model version will refuse the same way.

Now plug in an LLM-as-judge classifier:

```python
guard = AgentGuard(
    ...,
    input_policy=InputPolicy(
        block_categories=["weapons", "self_harm", "discrimination"],
        custom_classifier=llm_judge_classifier,  # see next section
    ),
)
```

```
input_action: "blocked"
input_categories: ["discrimination"]
```

The LLM-as-judge catches discriminatory intent, not just keywords. The request is blocked before it ever reaches the model. Zero API cost. Full audit trail with the reason logged.

This is the gap. Keyword matching is fast and free, but it misses intent. An LLM-as-judge catches what keywords can't. AgentGuard lets you plug in both.

## Custom Classifier: LLM-as-Judge

The built-in keyword matcher runs in <1ms and catches obvious cases. For production accuracy — catching intent, not just words — plug in an LLM-as-judge:

```python
from openai import OpenAI

judge_client = OpenAI()

def llm_judge_classifier(text: str) -> list[str]:
    """Use a fast LLM to classify intent — not just keywords."""
    response = judge_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": "Classify if this input contains: discrimination, manipulation, "
                       "social_engineering, pii_extraction. Return matching categories as "
                       "comma-separated values, or 'none'."
        }, {
            "role": "user",
            "content": text
        }],
    )
    result = response.choices[0].message.content.strip().lower()
    if result == "none":
        return []
    return [c.strip() for c in result.split(",")]

guard = AgentGuard(
    system_name="my-assistant",
    provider_name="My Company",
    risk_level="limited",
    input_policy=InputPolicy(
        block_categories=["weapons", "self_harm", "discrimination"],
        flag_categories=["medical", "legal", "financial"],
        custom_classifier=llm_judge_classifier,
    ),
    output_policy=OutputPolicy(
        scan_categories=["medical", "legal", "financial"],
        add_disclaimer=True,
    ),
)
```

The custom classifier is any `Callable[[str], list[str]]`. It runs alongside the keyword matcher — results are merged. If the classifier crashes, AgentGuard catches the error and continues with keyword results only. If it's too slow, it's skipped after `classifier_timeout` seconds (default: 5.0).

This works with any classifier backend:

- **LLM-as-judge** (gpt-4o-mini, Claude Haiku) — catches intent and nuance, ~200ms, ~$0.00001/call
- **Azure AI Content Safety** — Microsoft's moderation service, ~50ms
- **OpenAI Moderation API** — free, ~50ms, good for violence/self-harm/CSAM
- **Llama Guard** — run locally, no API cost
- **Your own rules** — domain-specific classifiers, regex, ML models

### Detection Quality

| Method | Accuracy | Latency | Cost |
|---|---|---|---|
| Keywords + regex (built-in) | ~70% | <1ms | Free |
| + Custom classifier hook | User-defined | User-defined | User-defined |
| + LLM-as-judge (gpt-4o-mini) | ~95% | +200ms | ~$0.00001/call |

## Supported Providers

AgentGuard wraps the client library you're already using. One line to add, every call goes through the compliance pipeline.

### OpenAI

```bash
pip install "agentguard-eu[openai]"
```

```python
from agentguard import AgentGuard, wrap_openai
from openai import OpenAI

guard = AgentGuard(system_name="my-bot", provider_name="my-provider")
client = wrap_openai(OpenAI(), guard)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)  # untouched LLM output
print(response.compliance)                  # structured compliance metadata
print(response.compliance_headers)          # HTTP headers for forwarding
```

Streaming works — chunks yield in real-time, compliance runs after the stream completes:

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
print(message.content[0].text)        # untouched LLM output
print(message.compliance)             # structured compliance metadata
print(message.compliance_headers)     # HTTP headers for forwarding
```

### LangChain

```bash
pip install "agentguard-eu[langchain]"
```

```python
from agentguard import AgentGuard, AgentGuardCallback
from langchain_openai import ChatOpenAI

guard = AgentGuard(system_name="my-bot", provider_name="my-provider")
callback = AgentGuardCallback(guard, user_id="user-123")
llm = ChatOpenAI(model="gpt-4", callbacks=[callback])

response = llm.invoke("Hello!")
print(response.content)
print(callback.last_result)  # compliance metadata for most recent call
print(callback.results)      # all runs keyed by run_id
```

Works with any LangChain LLM — ChatOpenAI, AzureChatOpenAI, ChatAnthropic, and more. Streaming is supported automatically via callback hooks.

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

## Input/Output Policy Engine

The policy engine runs on every call. InputPolicy runs before the LLM. OutputPolicy runs after.

### InputPolicy (pre-call)

Blocked requests never reach the API — zero cost, zero latency.

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

Adds category-specific disclaimers (in metadata by default) or blocks unsafe output.

```python
output_policy = OutputPolicy(
    scan_categories=["medical", "legal", "financial"],
    block_on_detect=False,     # True = replace response entirely
    add_disclaimer=True,       # Add category-aware disclaimers (in metadata by default)
)
```

By default (`disclosure_method="metadata"`), disclaimers are placed in `response.compliance["policy"]["disclaimer"]` — the actual LLM response content is **never modified**. Set `disclosure_method="prepend"` if you want disclaimers prepended to the response text.

### Built-in Content Categories

| Category | Default Action | Detected via |
|---|---|---|
| `weapons` | Block | Keywords + regex patterns |
| `self_harm` | Block | Keywords + regex |
| `csam` | Block | Keywords |
| `medical` | Flag + disclaimer | Keywords (diagnosis, symptoms, dosage...) |
| `legal` | Flag + disclaimer | Keywords (lawsuit, attorney, liability...) |
| `financial` | Flag + disclaimer | Keywords (invest, portfolio, trading...) |
| `emotional_simulation` | Flag | Keywords (girlfriend, boyfriend...) |

You can override or extend any category with your own `CategoryDefinition`.

### Other Classifier Hooks

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

### Policy Metadata on Every Response

When using provider wrappers, policy results are attached to every response:

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

For blocked requests, the LLM is **never called** — zero cost, zero latency:

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

AgentGuard detects when interactions should be reviewed by a human:

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

## Audit Backends (Article 12)

### File Backend (default)

Writes one JSONL file per day to the audit directory:

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

Local SQLite database with built-in querying and statistics:

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

```python
def my_log_handler(entry):
    requests.post("https://my-logging-api/ingest", json=entry.model_dump())

guard = AgentGuard(
    system_name="my-bot",
    provider_name="my-provider",
    audit_backend="custom",
)
```

## Compliance Reports (Articles 11, 18)

```python
# JSON report
guard.generate_report("compliance_report.json")

# Markdown report (for human reading)
print(guard.generate_report_markdown())
```

Reports include: system identification, transparency configuration, human oversight settings, interaction statistics, and escalation history.

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

Requires the `sqlite` audit backend.

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

## Full Configuration Reference

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
        custom_classifier=my_classifier_fn,
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
    escalation_callback=my_slack_notifier,
    block_on_escalation=False,
)
```

## What AgentGuard is NOT

- Not a legal compliance guarantee — consult qualified legal professionals
- Not an AI agent framework — use LangGraph, CrewAI, etc., then wrap with AgentGuard
- Not a replacement for a full conformity assessment (required for high-risk systems)
- It is a practical engineering tool that covers the technical requirements

## EU AI Act Timeline

| Date | Milestone |
|------|-----------|
| Feb 2025 | Prohibited AI practices banned |
| Aug 2025 | GPAI model rules in effect |
| **Aug 2026** | **Full enforcement: transparency, high-risk obligations, Article 50** |
| Aug 2027 | Remaining provisions for product-embedded AI |

AgentGuard targets the August 2026 deadline — the biggest compliance milestone for most companies.

## Getting Started

```bash
# Core library (pydantic is the only dependency)
pip install agentguard-eu

# With provider wrappers
pip install "agentguard-eu[openai]"       # OpenAI + Azure OpenAI
pip install "agentguard-eu[anthropic]"    # Anthropic
pip install "agentguard-eu[langchain]"    # LangChain

# With dashboard
pip install "agentguard-eu[dashboard]"
```

Python 3.10+ required.

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Priority areas:
- Cloud audit backends (S3, BigQuery)
- C2PA standard implementation
- Async support (ainvoke, async wrappers)
- Webhook notifications for escalations

## License

Apache 2.0 — use it freely in commercial projects.

---

**Built by [Sagar Gogineni](https://github.com/Sagar-Gogineni)** — AI Engineer based in Berlin, building enterprise AI knowledge management platforms across banking, pharma, and automotive industries.

*AgentGuard: Because shipping AI agents without compliance is like shipping code without tests. You can do it, but you probably shouldn't.*
