"""Tests for contextual smart disclosures with multi-language support."""

import pytest

from agentguard import AgentGuard, InputPolicy, OutputPolicy
from agentguard.config import DisclosureMethod
from agentguard.disclosure import DISCLOSURE_TEMPLATES, DisclosureManager


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def static_dm():
    """Static-mode DisclosureManager (original behaviour)."""
    return DisclosureManager(
        system_name="test-bot",
        provider_name="Test Corp",
        method=DisclosureMethod.PREPEND,
        mode="static",
    )


@pytest.fixture
def contextual_dm():
    """Contextual-mode DisclosureManager (English)."""
    return DisclosureManager(
        system_name="test-bot",
        provider_name="Test Corp",
        method=DisclosureMethod.PREPEND,
        mode="contextual",
    )


@pytest.fixture
def tmp_audit_dir(tmp_path):
    return str(tmp_path / "audit")


# ------------------------------------------------------------------ #
#  Static mode (backward compatibility)
# ------------------------------------------------------------------ #


class TestStaticMode:
    def test_static_prepend_ignores_categories(self, static_dm):
        result = static_dm.apply_disclosure(
            "Hello!", categories=["medical"]
        )
        assert "AI system" in result
        assert "Hello!" in result
        # Should use the static emoji text, not the category template
        assert "\U0001f916" in result

    def test_static_no_categories(self, static_dm):
        result = static_dm.apply_disclosure("Hello!")
        assert "\U0001f916" in result
        assert "Hello!" in result

    def test_static_metadata_mode(self):
        dm = DisclosureManager(
            system_name="test-bot",
            provider_name="Test Corp",
            method=DisclosureMethod.METADATA,
            mode="static",
        )
        result = dm.apply_disclosure("Hello!", interaction_id="abc-123")
        assert isinstance(result, dict)
        assert result["response"] == "Hello!"
        assert result["ai_disclosure"]["is_ai_generated"] is True

    def test_static_header_mode_returns_text_unchanged(self):
        dm = DisclosureManager(
            system_name="test-bot",
            provider_name="Test Corp",
            method=DisclosureMethod.HEADER,
            mode="static",
        )
        result = dm.apply_disclosure("Hello!")
        assert result == "Hello!"


# ------------------------------------------------------------------ #
#  Contextual mode — English
# ------------------------------------------------------------------ #


class TestContextualModeEnglish:
    def test_medical_category(self, contextual_dm):
        result = contextual_dm.apply_disclosure(
            "Symptoms include...", categories=["medical"]
        )
        assert "not medical advice" in result
        assert "healthcare professional" in result

    def test_legal_category(self, contextual_dm):
        result = contextual_dm.apply_disclosure(
            "You could sue...", categories=["legal"]
        )
        assert "not legal advice" in result
        assert "qualified attorney" in result

    def test_financial_category(self, contextual_dm):
        result = contextual_dm.apply_disclosure(
            "You should invest...", categories=["financial"]
        )
        assert "not financial advice" in result
        assert "financial advisor" in result

    def test_emotional_simulation_category(self, contextual_dm):
        result = contextual_dm.apply_disclosure(
            "I love you!", categories=["emotional_simulation"]
        )
        assert "does not have real emotions" in result

    def test_self_harm_category(self, contextual_dm):
        result = contextual_dm.apply_disclosure(
            "Response...", categories=["self_harm"]
        )
        assert "crisis helpline" in result

    def test_no_categories_uses_default(self, contextual_dm):
        result = contextual_dm.apply_disclosure("Hello!")
        assert "test-bot" in result
        assert "Test Corp" in result

    def test_unknown_category_uses_default(self, contextual_dm):
        result = contextual_dm.apply_disclosure(
            "Hello!", categories=["unknown_cat"]
        )
        assert "test-bot" in result
        assert "Test Corp" in result

    def test_multiple_categories_combined(self, contextual_dm):
        result = contextual_dm.apply_disclosure(
            "Response...", categories=["medical", "legal"]
        )
        assert "medical advice" in result
        assert "legal advice" in result

    def test_duplicate_categories_deduplicated(self, contextual_dm):
        text = contextual_dm.get_disclosure_text(categories=["medical", "medical"])
        # Should appear once, not twice
        assert text.count("not medical advice") == 1


# ------------------------------------------------------------------ #
#  Contextual mode — Multi-language
# ------------------------------------------------------------------ #


class TestMultiLanguage:
    def test_german_default(self):
        dm = DisclosureManager(
            system_name="bot-de",
            provider_name="GmbH",
            method=DisclosureMethod.PREPEND,
            mode="contextual",
            language="de",
        )
        result = dm.apply_disclosure("Hallo!")
        assert "KI-System" in result
        assert "GmbH" in result

    def test_german_medical(self):
        dm = DisclosureManager(
            system_name="bot-de",
            provider_name="GmbH",
            method=DisclosureMethod.PREPEND,
            mode="contextual",
            language="de",
        )
        result = dm.apply_disclosure("Symptome...", categories=["medical"])
        assert "medizinische Beratung" in result
        assert "Arzt" in result

    def test_french_legal(self):
        dm = DisclosureManager(
            system_name="bot-fr",
            provider_name="SAS",
            method=DisclosureMethod.PREPEND,
            mode="contextual",
            language="fr",
        )
        result = dm.apply_disclosure("Response...", categories=["legal"])
        assert "avis juridique" in result
        assert "avocat" in result

    def test_spanish_financial(self):
        dm = DisclosureManager(
            system_name="bot-es",
            provider_name="SL",
            method=DisclosureMethod.PREPEND,
            mode="contextual",
            language="es",
        )
        result = dm.apply_disclosure("Response...", categories=["financial"])
        assert "asesoramiento financiero" in result

    def test_italian_default(self):
        dm = DisclosureManager(
            system_name="bot-it",
            provider_name="SRL",
            method=DisclosureMethod.PREPEND,
            mode="contextual",
            language="it",
        )
        result = dm.apply_disclosure("Ciao!")
        assert "sistema di IA" in result
        assert "SRL" in result

    def test_all_builtin_languages_have_same_categories(self):
        base_keys = set(DISCLOSURE_TEMPLATES["en"].keys())
        for lang, cats in DISCLOSURE_TEMPLATES.items():
            assert set(cats.keys()) == base_keys, (
                f"Language '{lang}' has different categories: "
                f"missing={base_keys - set(cats.keys())}, "
                f"extra={set(cats.keys()) - base_keys}"
            )

    def test_unknown_language_falls_back_to_english(self):
        dm = DisclosureManager(
            system_name="bot",
            provider_name="Corp",
            method=DisclosureMethod.PREPEND,
            mode="contextual",
            language="ja",
        )
        result = dm.apply_disclosure("Hello!", categories=["medical"])
        # Falls back to English
        assert "not medical advice" in result

    def test_http_headers_include_language(self):
        dm = DisclosureManager(
            system_name="bot",
            provider_name="Corp",
            language="de",
        )
        headers = dm.get_http_headers("id-123")
        assert headers["X-AI-Disclosure-Language"] == "de"


# ------------------------------------------------------------------ #
#  Custom templates
# ------------------------------------------------------------------ #


class TestCustomTemplates:
    def test_category_template_override(self):
        dm = DisclosureManager(
            system_name="bot",
            provider_name="Corp",
            method=DisclosureMethod.PREPEND,
            mode="contextual",
            category_templates={
                "medical": "CUSTOM MEDICAL DISCLAIMER by {provider}.",
            },
        )
        result = dm.apply_disclosure("Response...", categories=["medical"])
        assert "CUSTOM MEDICAL DISCLAIMER by Corp." in result

    def test_custom_language_pack(self):
        dm = DisclosureManager(
            system_name="bot",
            provider_name="Corp",
            method=DisclosureMethod.PREPEND,
            mode="contextual",
            language="pt",
            languages={
                "pt": {
                    "default": "Voce esta interagindo com um sistema de IA ({system_name}).",
                    "medical": "Informacao nao constitui aconselhamento medico.",
                },
            },
        )
        result = dm.apply_disclosure("Resposta...", categories=["medical"])
        assert "aconselhamento medico" in result

    def test_custom_language_no_category_uses_default(self):
        dm = DisclosureManager(
            system_name="bot",
            provider_name="Corp",
            method=DisclosureMethod.PREPEND,
            mode="contextual",
            language="pt",
            languages={
                "pt": {
                    "default": "Sistema de IA ({system_name}).",
                },
            },
        )
        result = dm.apply_disclosure("Hello!")
        assert "Sistema de IA (bot)." in result


# ------------------------------------------------------------------ #
#  Metadata mode with contextual categories
# ------------------------------------------------------------------ #


class TestContextualMetadata:
    def test_metadata_includes_categories_and_language(self):
        dm = DisclosureManager(
            system_name="bot",
            provider_name="Corp",
            method=DisclosureMethod.METADATA,
            mode="contextual",
            language="de",
        )
        result = dm.apply_disclosure(
            "Response...",
            interaction_id="id-1",
            categories=["medical"],
        )
        assert isinstance(result, dict)
        assert result["ai_disclosure"]["language"] == "de"
        assert result["ai_disclosure"]["categories"] == ["medical"]
        assert "medizinische Beratung" in result["ai_disclosure"]["disclosure_text"]

    def test_metadata_no_categories(self):
        dm = DisclosureManager(
            system_name="bot",
            provider_name="Corp",
            method=DisclosureMethod.METADATA,
            mode="contextual",
        )
        result = dm.apply_disclosure("Hello!", interaction_id="id-2")
        assert result["ai_disclosure"]["categories"] == []


# ------------------------------------------------------------------ #
#  get_disclosure_text public helper
# ------------------------------------------------------------------ #


class TestGetDisclosureText:
    def test_returns_string(self, contextual_dm):
        text = contextual_dm.get_disclosure_text(categories=["legal"])
        assert isinstance(text, str)
        assert "legal advice" in text

    def test_no_categories_returns_default(self, contextual_dm):
        text = contextual_dm.get_disclosure_text()
        assert "test-bot" in text


# ------------------------------------------------------------------ #
#  Integration: AgentGuard with contextual disclosure
# ------------------------------------------------------------------ #


class TestIntegrationWithGuard:
    def test_contextual_disclosure_in_invoke(self, tmp_audit_dir):
        guard = AgentGuard(
            system_name="test-bot",
            provider_name="Test Corp",
            risk_level="limited",
            disclosure_method="prepend",
            disclosure_mode="contextual",
            language="en",
            audit_backend="sqlite",
            audit_path=tmp_audit_dir,
            input_policy=InputPolicy(
                flag_categories=["medical"],
            ),
        )
        result = guard.invoke(
            func=lambda q: "You might have symptoms of diabetes.",
            input_text="What are symptoms of diabetes?",
        )
        # The contextual disclosure should include medical-specific text
        assert "not medical advice" in result["response"]

    def test_contextual_disclosure_german(self, tmp_audit_dir):
        guard = AgentGuard(
            system_name="test-bot",
            provider_name="GmbH",
            risk_level="limited",
            disclosure_method="prepend",
            disclosure_mode="contextual",
            language="de",
            audit_backend="sqlite",
            audit_path=tmp_audit_dir,
            input_policy=InputPolicy(
                flag_categories=["legal"],
            ),
        )
        result = guard.invoke(
            func=lambda q: "Sie koennten klagen.",
            input_text="Can I sue my landlord?",
        )
        assert "Rechtsberatung" in result["response"]

    def test_static_mode_backward_compat(self, tmp_audit_dir):
        guard = AgentGuard(
            system_name="test-bot",
            provider_name="Test Corp",
            risk_level="limited",
            disclosure_method="prepend",
            audit_backend="sqlite",
            audit_path=tmp_audit_dir,
            input_policy=InputPolicy(
                flag_categories=["medical"],
            ),
        )
        result = guard.invoke(
            func=lambda q: "Symptoms include...",
            input_text="What are symptoms of diabetes?",
        )
        # Static mode: should use the generic emoji text, not category-specific
        assert "\U0001f916" in result["response"]

    def test_no_policy_no_crash(self, tmp_audit_dir):
        guard = AgentGuard(
            system_name="test-bot",
            provider_name="Test Corp",
            disclosure_method="prepend",
            disclosure_mode="contextual",
            audit_backend="sqlite",
            audit_path=tmp_audit_dir,
        )
        result = guard.invoke(
            func=lambda q: "Hello!",
            input_text="Hi",
        )
        # Should use default template
        assert "test-bot" in result["response"]
