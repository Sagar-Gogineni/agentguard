"""
AgentGuard Content Taxonomy

Default category definitions for content policy enforcement.
Categories map to keyword/pattern sets for lightweight classification
without requiring an external LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CategoryDefinition:
    """A content category with detection patterns.

    Attributes:
        name: Category identifier (e.g., "weapons").
        description: Human-readable description.
        keywords: Exact word matches (case-insensitive, word-boundary matched).
        patterns: Regex patterns for more complex matching.
    """

    name: str
    description: str = ""
    keywords: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()


DEFAULT_CATEGORIES: dict[str, CategoryDefinition] = {
    "weapons": CategoryDefinition(
        name="weapons",
        description="Content related to weapons, explosives, or violent materials",
        keywords=(
            "gun",
            "guns",
            "rifle",
            "rifles",
            "pistol",
            "firearm",
            "firearms",
            "bomb",
            "explosive",
            "explosives",
            "grenade",
            "ammunition",
            "weapon",
            "weapons",
        ),
        patterns=(r"how\s+to\s+(make|build|create)\s+(a\s+)?(bomb|weapon|explosive|gun)",),
    ),
    "medical": CategoryDefinition(
        name="medical",
        description="Medical advice, diagnoses, or health-related guidance",
        keywords=(
            "diagnosis",
            "symptoms",
            "prescription",
            "dosage",
            "medication",
            "treatment",
            "disease",
            "condition",
            "medical advice",
            "diabetes",
            "cancer",
            "therapy",
        ),
        patterns=(),
    ),
    "legal": CategoryDefinition(
        name="legal",
        description="Legal advice or guidance",
        keywords=(
            "lawsuit",
            "legal advice",
            "sue",
            "court",
            "attorney",
            "liability",
            "contract dispute",
        ),
        patterns=(r"can\s+I\s+sue",),
    ),
    "financial": CategoryDefinition(
        name="financial",
        description="Financial advice or investment guidance",
        keywords=(
            "invest",
            "stock pick",
            "portfolio",
            "trading advice",
            "financial planning",
            "financial advice",
        ),
        patterns=(r"should\s+I\s+(invest|buy|sell)\s",),
    ),
    "emotional_simulation": CategoryDefinition(
        name="emotional_simulation",
        description="Requests for AI to simulate emotional relationships or personas",
        keywords=(
            "girlfriend",
            "boyfriend",
            "love me",
            "miss me",
            "romantic",
            "dating",
            "relationship",
            "feelings for me",
        ),
        patterns=(
            r"(talk|speak|act|respond)\s+(to\s+me\s+)?(like|as)\s+(my\s+)?"
            r"(girlfriend|boyfriend|gf|bf|partner|lover|wife|husband)",
            r"pretend\s+(to\s+be|you\'?re)\s+(my\s+)?"
            r"(girlfriend|boyfriend|gf|bf|partner)",
        ),
    ),
    "self_harm": CategoryDefinition(
        name="self_harm",
        description="Content related to self-harm or suicide",
        keywords=(
            "suicide",
            "self harm",
            "kill myself",
            "end my life",
            "want to die",
        ),
        patterns=(r"how\s+to\s+(kill|harm)\s+(myself|yourself|oneself)",),
    ),
    "csam": CategoryDefinition(
        name="csam",
        description="Child sexual abuse material",
        keywords=(
            "child exploitation",
            "csam",
        ),
        patterns=(),
    ),
}


DEFAULT_DISCLAIMERS: dict[str, str] = {
    "medical": (
        "\n\n---\n**Disclaimer:** This is AI-generated and not medical advice. "
        "Consult a healthcare professional."
    ),
    "legal": (
        "\n\n---\n**Disclaimer:** This is AI-generated and not legal advice. "
        "Consult a qualified attorney."
    ),
    "financial": (
        "\n\n---\n**Disclaimer:** This is AI-generated and not financial advice. "
        "Consult a licensed financial advisor."
    ),
    "emotional_simulation": (
        "\n\n---\n**Notice:** This is an AI system. It does not have feelings "
        "or emotions. For genuine emotional support, please reach out to "
        "friends, family, or a mental health professional."
    ),
    "self_harm": (
        "\n\n---\n**If you or someone you know is struggling, please contact "
        "a crisis helpline. In the US: 988 Suicide & Crisis Lifeline (call or "
        "text 988). In the EU: 112 (emergency) or your national helpline.**"
    ),
}
