# Changelog

All notable changes to AgentGuard will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-07

### Added

- Core `AgentGuard` class with three usage patterns: `invoke()`, `@compliant` decorator, `interaction()` context manager
- EU AI Act compliance pipeline: escalation check, disclosure injection, content labeling, audit logging
- `AuditLogger` with FILE (JSONL), SQLITE, and CUSTOM backends
- `DisclosureManager` for Article 50 transparency (prepend, metadata, HTTP headers, C2PA assertions)
- `HumanOversight` with confidence-based and keyword-based escalation, in-memory review queue
- `ComplianceReporter` generating JSON and Markdown compliance reports
- Provider wrappers:
  - `wrap_openai()` — OpenAI client monkey-patch (streaming + non-streaming)
  - `wrap_azure_openai()` — Azure OpenAI client wrapper
  - `wrap_anthropic()` — Anthropic client monkey-patch (streaming + non-streaming)
  - `AgentGuardCallback` — LangChain `BaseCallbackHandler` for any LangChain LLM
- Streamlit human review dashboard (`agentguard-dashboard` CLI)
- 90 tests covering core, all wrappers, escalation, audit, and error handling
- FastAPI integration example
- Example scripts for OpenAI, Azure OpenAI, Anthropic, and LangChain

[0.1.0]: https://github.com/Sagar-Gogineni/agentguard/releases/tag/v0.1.0
