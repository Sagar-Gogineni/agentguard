<p align="center">
  <img src="https://raw.githubusercontent.com/Sagar-Gogineni/agentguard/main/assets/logo.png" alt="agentguard-eu" width="400">
</p>

<p align="center">
  <a href="https://pypi.org/project/agentguard-eu/"><img src="https://img.shields.io/pypi/v/agentguard-eu.svg" alt="PyPI version"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
</p>

---

## Your AI system might already be illegal in the EU

Not in August 2026. Right now.

**[Take the free 2-minute risk assessment →](https://navario.tech/quiz)**

---

## Three scenarios you haven't thought about

### Automotive: Your voice agent quietly changed risk tiers

You ship an in-car AI voice assistant. Navigation, music, calls, messages. Limited risk under Article 50. Then three things happen over 12 months of OTA updates.

Update 1: The assistant starts generating text messages on behalf of the driver ("tell my wife I'll be late"). Under Article 50(2), AI-generated content sent to third parties must be machine-readable labeled as AI-generated. The wife receives a message that looks like it came from the driver. No labeling. That's a transparency violation.

Update 2: The assistant starts analyzing driving patterns and suggesting routes based on habits. That's profiling of natural persons. Under Article 6(3), systems that perform profiling are always classified as high-risk, even if they'd otherwise qualify for an exception. Your system just jumped from LIMITED to HIGH risk. New obligations: Art. 9 risk management, Art. 11 technical documentation, Art. 14 human oversight, Art. 49 EU database registration.

Update 3: Product adds a "driving style score" that gets shared with the fleet management dashboard. Now the AI is evaluating worker performance. Annex III, paragraph 4(b): AI systems used for performance monitoring of workers. HIGH risk with employment-specific obligations.

Three incremental product decisions. None of them felt like compliance events. All of them changed the regulatory classification. The assessment done at launch is now wrong.

### Pharma: Your medical information AI is a regulatory time bomb

Your medical affairs team deploys an AI system that answers healthcare professionals' questions about your drug products. It's trained on approved prescribing information, clinical trial summaries, and your product label. Seems safe.

A physician asks about using your drug for a condition it's not approved for. The AI synthesizes data from a clinical trial and suggests potential efficacy. That's off-label promotion. Under EU pharmaceutical law, that's already illegal. Under the EU AI Act, this AI system is influencing clinical decisions and may qualify as HIGH risk under Annex III (medical devices). Your compliance team assessed it as a "simple Q&A tool" and classified it as LIMITED.

The high-risk classification isn't automatic. Article 6(3) allows providers to rebut the presumption if the system doesn't pose a significant risk of harm, and a Q&A tool answering trained healthcare professionals (not patients) has a case for rebuttal. But you need to document that assessment per Article 6(4) and register it before deployment. If you haven't done that analysis, you're operating in a gray zone with no paper trail.

On top of that: the AI generates responses that look like they came from your medical science liaison team. No AI disclosure. Under Article 50(1), providers must design AI systems so users know they're interacting with AI. Your HCPs don't know. That's a transparency violation layered on the off-label risk.

### Financial services: Your credit model silently became high-risk

You build a lending API. At launch, it's a rule-based scoring system with an ML component for fraud detection. Your team classifies it as LIMITED risk because the ML model is "just one input among many" and a human reviews every decision.

Over 18 months, the product team improves the model. The ML score starts carrying 80% of the weight. The human reviewer rubber-stamps 97% of decisions. The system now makes autonomous credit decisions about natural persons. That's Annex III, paragraph 5(b): creditworthiness assessment. HIGH risk. Full obligations: Art. 9 risk management, Art. 10 bias audits, Art. 14 meaningful human oversight (not rubber-stamping), Art. 27 fundamental rights impact assessment.

Nobody re-classified. The assessment from two years ago still says LIMITED. The regulator won't care what it said then. They care what the system does now.

---

## The pattern

All three scenarios share the same failure: **the AI system's classification drifted after the initial assessment.** Features were added. Usage patterns changed. The product evolved. The compliance status didn't.

agentguard-eu is open-source compliance infrastructure for AI systems. It does two things:

1. **Classifies your AI system** against the EU AI Act, including per-feature risk analysis, provider vs deployer obligations, and sector-specific requirements
2. **Enforces compliance at runtime** with audit logging, content policies, transparency disclosures, and human escalation

**[Check your risk tier in 2 minutes → navario.tech/quiz](https://navario.tech/quiz)**

---

## What is the EU AI Act?

Regulation (EU) 2024/1689. The world's first comprehensive AI regulation. It defines risk tiers and imposes different obligations depending on whether you build AI (provider) or use it (deployer).

**Already in force:**
- Feb 2025: Prohibited practices (Art. 5) — social scoring, manipulative AI, workplace emotion recognition
- Aug 2025: GPAI model obligations (Art. 51-56)

**Coming Aug 2026:**
- Transparency obligations (Art. 50) + high-risk system requirements (Art. 6-49) — employment, credit, healthcare, education, law enforcement

**Coming Aug 2027:**
- Remaining provisions for AI systems embedded in products covered by EU harmonisation legislation

Penalties: up to 35,000,000 EUR or 7% of total worldwide annual turnover, whichever is higher.

---

## How agentguard-eu works

Every LLM call passes through this pipeline:

```
Input → [InputPolicy: block/flag/allow] → [LLM Call] → [OutputPolicy: disclaim/block/pass]
  → [Disclosure] → [ContentLabel] → [EscalationCheck] → [AuditLog] → Output
```

**By default, the LLM response content is never modified.** All compliance data goes into `response.compliance` metadata.

```bash
pip install agentguard-eu
```

```python
from agentguard import AgentGuard, InputPolicy, wrap_openai
from openai import OpenAI

guard = AgentGuard(
    system_name="my-assistant",
    provider_name="My Company",
    risk_level="high",
    input_policy=InputPolicy(
        block_categories=["weapons", "self_harm"],
        flag_categories=["medical", "legal", "financial"],
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

Your existing code doesn't change. agentguard-eu wraps it.

### What it covers

| EU AI Act Article | What agentguard-eu does | How |
|---|---|---|
| **Art. 5** | Content policy enforcement | Block, flag, or escalate harmful content before and after the LLM responds. Blocked requests never reach the API. |
| **Art. 12** | Structured audit logging | Every interaction logged with timestamps, user IDs, categories detected, actions taken. SQLite, file, or webhook backends. |
| **Art. 14** | Human oversight escalation | Automatic escalation on low confidence, sensitive topics, or policy triggers. Review queue with approve/reject. |
| **Art. 50** | Transparency and content labeling | Contextual disclosures adapted to detected content categories, in 5 languages. C2PA-compatible machine-readable labels. |

### What it doesn't cover (be honest with your team about this)

| Obligation | Why agentguard-eu can't do this |
|---|---|
| **Art. 9** Risk management system | Organizational process requiring human-led risk assessment |
| **Art. 10** Bias audits | Requires testing training data for representativeness. Pre-deployment, not runtime. |
| **Art. 11** Technical documentation | A written deliverable describing system design, not a code feature |
| **Art. 27** Fundamental rights impact assessment | Human-led assessment of impact on affected persons |

agentguard-eu covers the runtime enforcement obligations. The organizational obligations are on you. The [free assessment](https://navario.tech/quiz) shows you the full breakdown.

---

## Smart content classification

The built-in keyword matcher catches obvious cases in <1ms. For production accuracy, plug in an LLM-as-judge:

```python
def llm_judge_classifier(text: str) -> list[str]:
    """Use a fast LLM to classify intent, not just keywords."""
    response = judge_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": "Classify if this input contains: discrimination, "
                       "manipulation, off_label_promotion, pii_extraction. "
                       "Return matching categories or 'none'."
        }, {"role": "user", "content": text}],
    )
    result = response.choices[0].message.content.strip().lower()
    return [] if result == "none" else [c.strip() for c in result.split(",")]

guard = AgentGuard(
    ...,
    input_policy=InputPolicy(
        block_categories=["weapons", "self_harm", "discrimination"],
        custom_classifier=llm_judge_classifier,
    ),
)
```

The custom classifier is any `Callable[[str], list[str]]`. It runs alongside the keyword matcher. Results are merged. If the classifier crashes or times out (default: 5s), agentguard-eu continues with keyword results only.

Works with any classifier backend: LLM-as-judge, Azure AI Content Safety, OpenAI Moderation API, Llama Guard, or your own domain rules.

| Method | Accuracy | Latency | Cost |
|---|---|---|---|
| Keywords + regex (built-in) | ~70% | <1ms | Free |
| + LLM-as-judge (gpt-4o-mini) | ~95% | +200ms | ~$0.00001/call |

---

## Supported providers

One line to add. Every call goes through the compliance pipeline.

```bash
pip install "agentguard-eu[openai]"       # OpenAI + Azure OpenAI
pip install "agentguard-eu[anthropic]"    # Anthropic
pip install "agentguard-eu[langchain]"    # LangChain
```

```python
# OpenAI
client = wrap_openai(OpenAI(), guard)

# Azure OpenAI
client = wrap_azure_openai(AzureOpenAI(...), guard)

# Anthropic
client = wrap_anthropic(Anthropic(), guard)

# LangChain (callback)
callback = AgentGuardCallback(guard, user_id="user-123")
llm = ChatOpenAI(model="gpt-4", callbacks=[callback])
```

Streaming works. Chunks yield in real-time. Compliance runs after the stream completes.

---

## Contextual disclosures (Article 50)

Instead of a generic "you are talking to AI" message, agentguard-eu adapts the disclosure based on detected content categories, in the user's language.

```python
guard = AgentGuard(
    ...,
    disclosure_method="metadata",   # never touch response content (default)
    disclosure_mode="contextual",   # adapt to detected categories
    language="de",                  # en, de, fr, es, it built-in
)
```

When the policy engine detects `medical` content in German:

> Dies ist ein KI-System. Die bereitgestellten Informationen stellen keine medizinische Beratung dar. Bitte konsultieren Sie einen Arzt.

Instead of a generic "You are talking to an AI."

| Disclosure method | Behavior |
|---|---|
| `"metadata"` (default) | Attach to `response.compliance`. Never modify content. |
| `"prepend"` | Prepend disclosure text before response |
| `"first_only"` | Prepend only on first message per session |
| `"header"` | Return as HTTP headers via `response.compliance_headers` |
| `"none"` | Disable disclosure text (audit only) |

---

## Audit logging (Article 12)

Every interaction logged with full context. A regulator can query what happened.

```python
guard = AgentGuard(
    ...,
    audit_backend="sqlite",     # or "file" (JSONL) or "custom"
    audit_path="./audit_logs",
)

# Query audit logs
entries = guard.audit.query(
    start_date="2026-01-01",
    user_id="customer-42",
    escalated_only=True,
)

# Aggregate statistics
stats = guard.audit.get_stats()
# {"total_interactions": 1234, "total_escalations": 56, "avg_latency_ms": 245.3, ...}
```

---

## Human oversight (Article 14)

Automatic escalation on low confidence, sensitive topics, or policy triggers.

```python
guard = AgentGuard(
    ...,
    human_escalation="sensitive_topic",
    sensitive_keywords=["legal", "medical", "financial advice"],
    escalation_callback=my_slack_notifier,
    block_on_escalation=True,   # block response until human approves
)

# Review queue
for review in guard.pending_reviews:
    print(f"Needs review: {review['reason']}")

guard.oversight.approve(interaction_id)
guard.oversight.reject(interaction_id, reason="Inaccurate response")
```

---

## Input/Output policy engine

InputPolicy runs before the LLM. Blocked requests never reach the API. Zero cost. Zero latency.

```python
input_policy = InputPolicy(
    block_categories=["weapons", "self_harm", "csam"],
    flag_categories=["medical", "legal", "financial"],
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

output_policy = OutputPolicy(
    scan_categories=["medical", "legal", "financial"],
    add_disclaimer=True,
)
```

### Built-in content categories

| Category | Default action | Detection |
|---|---|---|
| `weapons` | Block | Keywords + regex |
| `self_harm` | Block | Keywords + regex |
| `csam` | Block | Keywords |
| `medical` | Flag + disclaimer | Keywords |
| `legal` | Flag + disclaimer | Keywords |
| `financial` | Flag + disclaimer | Keywords |
| `emotional_simulation` | Flag | Keywords |

Extend with your own `CategoryDefinition` or plug in a custom classifier for any category.

---

## FastAPI integration

Forward compliance headers to HTTP responses:

```python
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

---

## EU AI Act timeline

| Date | What happens |
|---|---|
| **Feb 2025** | Prohibited AI practices in force (Art. 5) |
| **Aug 2025** | GPAI model obligations in force (Art. 51-56) |
| **Aug 2026** | Transparency (Art. 50) + high-risk system requirements (Art. 6-49) |
| **Aug 2027** | Remaining provisions for product-embedded AI |

**Not sure what applies to you?** [Take the free assessment →](https://navario.tech/quiz)

---

## Getting started

```bash
# Core library (pydantic is the only dependency)
pip install agentguard-eu

# With provider wrappers
pip install "agentguard-eu[openai]"
pip install "agentguard-eu[anthropic]"
pip install "agentguard-eu[langchain]"

# With review dashboard
pip install "agentguard-eu[dashboard]"
```

Python 3.10+ required.

## What agentguard-eu is NOT

- Not a legal compliance guarantee. Consult qualified legal professionals.
- Not an AI agent framework. Use LangGraph, CrewAI, etc., then wrap with agentguard-eu.
- Not a replacement for organizational compliance (risk management, bias audits, documentation).
- It is a practical engineering tool that covers the runtime enforcement requirements.

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Priority areas:
- Cloud audit backends (S3, BigQuery, webhook)
- C2PA standard implementation
- Async support (ainvoke, async wrappers)
- Webhook notifications for escalations

## License

Apache 2.0. Use it freely in commercial projects.

---

**Built by [Navario](https://navario.tech)** — AI compliance infrastructure for developers.

**[Free EU AI Act Risk Assessment → navario.tech/quiz](https://navario.tech/quiz)**
