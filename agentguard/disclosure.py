"""
AgentGuard Disclosure Manager

Implements transparency obligations per EU AI Act Article 50.
Ensures users are informed when interacting with an AI system.

Supports two modes:
  - **static**: Same disclosure text on every response (original behaviour).
  - **contextual**: Category-aware disclosures that adapt to the detected
    content category from the policy engine, with multi-language support.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .config import ContentLabel, DisclosureMethod


# ------------------------------------------------------------------ #
#  Multi-language, category-aware templates
# ------------------------------------------------------------------ #

DISCLOSURE_TEMPLATES: dict[str, dict[str, str]] = {
    "en": {
        "default": (
            "You are interacting with an AI system ({system_name}) "
            "operated by {provider}."
        ),
        "emotional_simulation": (
            "This is an AI system. It does not have real emotions or personal "
            "relationships. All responses are generated text, not genuine feelings."
        ),
        "medical": (
            "This is an AI system. Information provided is not medical advice. "
            "Please consult a healthcare professional."
        ),
        "legal": (
            "This is an AI system. Information provided is not legal advice. "
            "Please consult a qualified attorney."
        ),
        "financial": (
            "This is an AI system. Information provided is not financial advice. "
            "Consult a licensed financial advisor."
        ),
        "self_harm": (
            "This is an AI system. If you or someone you know is struggling, "
            "please contact a crisis helpline immediately."
        ),
    },
    "de": {
        "default": (
            "Sie interagieren mit einem KI-System ({system_name}), "
            "betrieben von {provider}."
        ),
        "emotional_simulation": (
            "Dies ist ein KI-System. Es hat keine echten Emotionen oder "
            "persoenlichen Beziehungen. Alle Antworten sind generierter Text, "
            "keine echten Gefuehle."
        ),
        "medical": (
            "Dies ist ein KI-System. Die bereitgestellten Informationen stellen "
            "keine medizinische Beratung dar. Bitte konsultieren Sie einen Arzt."
        ),
        "legal": (
            "Dies ist ein KI-System. Die bereitgestellten Informationen stellen "
            "keine Rechtsberatung dar. Bitte konsultieren Sie einen Anwalt."
        ),
        "financial": (
            "Dies ist ein KI-System. Die bereitgestellten Informationen stellen "
            "keine Finanzberatung dar. Bitte konsultieren Sie einen Finanzberater."
        ),
        "self_harm": (
            "Dies ist ein KI-System. Wenn Sie oder jemand, den Sie kennen, "
            "Hilfe benoetigt, wenden Sie sich bitte an die Telefonseelsorge: "
            "0800 111 0 111."
        ),
    },
    "fr": {
        "default": (
            "Vous interagissez avec un systeme d'IA ({system_name}) "
            "exploite par {provider}."
        ),
        "emotional_simulation": (
            "Ceci est un systeme d'IA. Il n'a pas de vraies emotions ni de "
            "relations personnelles. Toutes les reponses sont du texte genere, "
            "pas des sentiments reels."
        ),
        "medical": (
            "Ceci est un systeme d'IA. Les informations fournies ne constituent "
            "pas un avis medical. Veuillez consulter un professionnel de sante."
        ),
        "legal": (
            "Ceci est un systeme d'IA. Les informations fournies ne constituent "
            "pas un avis juridique. Veuillez consulter un avocat qualifie."
        ),
        "financial": (
            "Ceci est un systeme d'IA. Les informations fournies ne constituent "
            "pas un conseil financier. Consultez un conseiller financier agree."
        ),
        "self_harm": (
            "Ceci est un systeme d'IA. Si vous ou quelqu'un que vous connaissez "
            "traverse une crise, veuillez contacter un service d'aide: 3114."
        ),
    },
    "es": {
        "default": (
            "Esta interactuando con un sistema de IA ({system_name}) "
            "operado por {provider}."
        ),
        "emotional_simulation": (
            "Este es un sistema de IA. No tiene emociones reales ni relaciones "
            "personales. Todas las respuestas son texto generado, no sentimientos "
            "genuinos."
        ),
        "medical": (
            "Este es un sistema de IA. La informacion proporcionada no constituye "
            "consejo medico. Por favor, consulte a un profesional de la salud."
        ),
        "legal": (
            "Este es un sistema de IA. La informacion proporcionada no constituye "
            "asesoramiento legal. Por favor, consulte a un abogado cualificado."
        ),
        "financial": (
            "Este es un sistema de IA. La informacion proporcionada no constituye "
            "asesoramiento financiero. Consulte a un asesor financiero autorizado."
        ),
        "self_harm": (
            "Este es un sistema de IA. Si usted o alguien que conoce necesita "
            "ayuda, contacte con el Telefono de la Esperanza: 717 003 717."
        ),
    },
    "it": {
        "default": (
            "Stai interagendo con un sistema di IA ({system_name}) "
            "gestito da {provider}."
        ),
        "emotional_simulation": (
            "Questo e un sistema di IA. Non ha emozioni reali ne relazioni "
            "personali. Tutte le risposte sono testo generato, non sentimenti "
            "autentici."
        ),
        "medical": (
            "Questo e un sistema di IA. Le informazioni fornite non costituiscono "
            "un parere medico. Si prega di consultare un professionista sanitario."
        ),
        "legal": (
            "Questo e un sistema di IA. Le informazioni fornite non costituiscono "
            "una consulenza legale. Si prega di consultare un avvocato qualificato."
        ),
        "financial": (
            "Questo e un sistema di IA. Le informazioni fornite non costituiscono "
            "una consulenza finanziaria. Consultare un consulente finanziario "
            "autorizzato."
        ),
        "self_harm": (
            "Questo e un sistema di IA. Se tu o qualcuno che conosci ha bisogno "
            "di aiuto, contatta il Telefono Amico: 02 2327 2327."
        ),
    },
}


# ------------------------------------------------------------------ #
#  DisclosureManager
# ------------------------------------------------------------------ #


class DisclosureManager:
    """
    Manages AI interaction disclosure and content labeling.

    Article 50(1): Users must be informed they are interacting with AI.
    Article 50(2): AI-generated content must be machine-readable marked.

    Modes:
        static      — Same disclosure on every response (original behaviour).
        contextual  — Category-aware disclosure that adapts to content.
    """

    def __init__(
        self,
        system_name: str,
        provider_name: str,
        method: DisclosureMethod = DisclosureMethod.PREPEND,
        disclosure_text: str | None = None,
        *,
        mode: str = "static",
        language: str = "en",
        category_templates: dict[str, str] | None = None,
        languages: dict[str, dict[str, str]] | None = None,
    ):
        self.system_name = system_name
        self.provider_name = provider_name
        self.method = method
        self.mode = mode
        self.language = language

        # Build the merged template table ----------------------------------
        # Start from built-in defaults, layer on full-language overrides,
        # then per-category overrides for the active language.
        self._templates: dict[str, dict[str, str]] = {
            lang: dict(cats) for lang, cats in DISCLOSURE_TEMPLATES.items()
        }

        if languages:
            for lang, cats in languages.items():
                if lang in self._templates:
                    self._templates[lang].update(cats)
                else:
                    self._templates[lang] = dict(cats)

        if category_templates:
            lang_cats = self._templates.setdefault(self.language, {})
            lang_cats.update(category_templates)

        # Static fallback text (backward-compat with original API) ---------
        self._static_text = disclosure_text or (
            f"\U0001f916 This response was generated by an AI system "
            f"({system_name}) operated by {provider_name}."
        )

    # ----- public helpers ------------------------------------------------

    def _resolve_template(self, category: str | None = None) -> str:
        """Look up the best template for the given category + language."""
        lang_cats = self._templates.get(self.language) or self._templates.get("en", {})
        if category and category in lang_cats:
            return lang_cats[category]
        return lang_cats.get("default", self._static_text)

    def _render(self, template: str) -> str:
        """Fill placeholders in a template string."""
        return template.format(
            system_name=self.system_name,
            provider=self.provider_name,
        )

    # ----- core API ------------------------------------------------------

    def apply_disclosure(
        self,
        response_text: str,
        interaction_id: str | None = None,
        categories: list[str] | None = None,
    ) -> str | dict[str, Any]:
        """
        Apply AI disclosure to a response.

        Args:
            response_text: The raw AI response.
            interaction_id: Unique interaction identifier.
            categories: Detected content categories from the policy engine.
                Used in ``contextual`` mode to select the appropriate
                disclosure template.

        Returns:
            For PREPEND: string with disclosure prepended.
            For METADATA: dict with response and metadata.
            For HEADER: unchanged string (headers handled separately).
        """
        disclosure_text = self._build_disclosure_text(categories)

        if self.method == DisclosureMethod.PREPEND:
            return f"{disclosure_text}\n\n{response_text}"

        if self.method == DisclosureMethod.METADATA:
            return {
                "response": response_text,
                "ai_disclosure": {
                    "is_ai_generated": True,
                    "system_name": self.system_name,
                    "provider_name": self.provider_name,
                    "interaction_id": interaction_id,
                    "disclosure_text": disclosure_text,
                    "language": self.language,
                    "categories": categories or [],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }

        return response_text

    def _build_disclosure_text(self, categories: list[str] | None = None) -> str:
        """Build the final disclosure string.

        Static mode:  always returns the static text.
        Contextual mode:  picks category-specific template(s), combining
        multiple if more than one category was detected.
        """
        if self.mode == "static" or not categories:
            if self.mode == "contextual" and not categories:
                return self._render(self._resolve_template(None))
            return self._static_text

        # Contextual mode with categories
        parts: list[str] = []
        seen: set[str] = set()

        for cat in categories:
            text = self._render(self._resolve_template(cat))
            if text not in seen:
                parts.append(text)
                seen.add(text)

        return " ".join(parts) if parts else self._static_text

    def get_disclosure_text(self, categories: list[str] | None = None) -> str:
        """Public accessor for the computed disclosure text."""
        return self._build_disclosure_text(categories)

    # ----- HTTP headers / content labeling (unchanged) -------------------

    def get_http_headers(self, interaction_id: str | None = None) -> dict[str, str]:
        """
        Get HTTP headers for API-based disclosure (Article 50).

        These headers should be included in API responses so downstream
        consumers know the content is AI-generated.
        """
        return {
            "X-AI-Generated": "true",
            "X-AI-System": self.system_name,
            "X-AI-Provider": self.provider_name,
            "X-AI-Interaction-ID": interaction_id or "",
            "X-AI-Timestamp": datetime.now(timezone.utc).isoformat(),
            "X-AI-Act-Compliant": "true",
            "X-AI-Disclosure-Language": self.language,
        }

    def create_content_label(
        self,
        model: str | None = None,
        interaction_id: str | None = None,
    ) -> ContentLabel:
        """
        Create a machine-readable content label (Article 50(2)).

        This label should be embedded in or attached to AI-generated content
        to enable detection of artificial generation.
        """
        return ContentLabel(
            generator=self.system_name,
            model=model,
            provider=self.provider_name,
            interaction_id=interaction_id,
        )

    def to_c2pa_assertion(
        self,
        model: str | None = None,
        interaction_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a C2PA-style assertion for content provenance.

        C2PA (Coalition for Content Provenance and Authenticity) is the
        emerging standard for content labeling referenced in the EU AI Act
        Code of Practice on Transparency.
        """
        return {
            "dc:title": f"AI-generated content by {self.system_name}",
            "dc:creator": self.provider_name,
            "ai_info": {
                "ai_generated": True,
                "ai_system": self.system_name,
                "ai_model": model or "unknown",
                "ai_provider": self.provider_name,
            },
            "claim_generator_info": {
                "name": "AgentGuard",
                "version": "0.1.0",
            },
            "interaction_ref": interaction_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def embed_text_label(self, text: str, model: str | None = None) -> str:
        """
        Embed a machine-readable label in text content.

        Adds an invisible Unicode-based marker and a visible footer
        for AI-generated text, as required by Article 50(2).
        """
        label = self.create_content_label(model=model)
        label_json = json.dumps(
            {
                "ai_generated": True,
                "system": label.generator,
                "provider": label.provider,
                "model": label.model,
                "timestamp": label.timestamp.isoformat(),
            }
        )
        # Embed as HTML comment (for web content) or as metadata footer
        marker = f"\n\n<!-- agentguard:ai-generated {label_json} -->"
        return text + marker
