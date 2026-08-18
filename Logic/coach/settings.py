"""Environment-backed settings for the unified Study Lab coach."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class CoachSettings:
    provider: str = os.getenv("COACH_LLM_PROVIDER", "groq")
    provider_order: str = os.getenv("COACH_PROVIDER_ORDER", os.getenv("COACH_LLM_PROVIDER", "groq"))
    fast_model: str = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-120b")
    tutor_model: str = os.getenv("GROQ_TUTOR_MODEL", "openai/gpt-oss-120b")
    # Deep tier: numerical, exam, and strict-grounding turns. Point this (or
    # OPENROUTER_DEEP_MODEL / OPENAI_DEEP_MODEL with provider order) at a
    # frontier model to upgrade reasoning-heavy turns without touching code.
    deep_model: str = os.getenv("GROQ_DEEP_MODEL", os.getenv("GROQ_TUTOR_MODEL", "openai/gpt-oss-120b"))
    review_model: str = os.getenv("GROQ_REVIEW_MODEL", "openai/gpt-oss-120b")
    # Groq currently has no supported image-input chat model. Leave this
    # unconfigured unless a supported vision route is supplied explicitly.
    vision_model: str = os.getenv("GROQ_VISION_MODEL", "")
    fallback_model: str = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-120b")
    route_preference: str = os.getenv("COACH_ROUTE_PREFERENCE", "balanced")
    budget_routing: bool = os.getenv("COACH_BUDGET_ROUTING", "true").lower() == "true"
    turn_budget_usd: float = float(os.getenv("COACH_TURN_BUDGET_USD", "0"))
    daily_budget_usd: float = float(os.getenv("COACH_DAILY_BUDGET_USD", "0"))
    route_output_token_estimate: int = int(os.getenv("COACH_ROUTE_OUTPUT_TOKEN_ESTIMATE", "700"))
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    llm_timeout_seconds: float = float(os.getenv("COACH_LLM_TIMEOUT_SECONDS", "22"))
    llm_max_attempts: int = int(os.getenv("COACH_LLM_MAX_ATTEMPTS", "2"))
    max_retrieval_chars: int = int(os.getenv("COACH_MAX_RETRIEVAL_CHARS", "5000"))
    max_retrieval_paragraphs: int = int(os.getenv("COACH_MAX_RETRIEVAL_PARAGRAPHS", "8"))
    memory_limit: int = int(os.getenv("COACH_MEMORY_LIMIT", "6"))
    interaction_limit: int = int(os.getenv("COACH_INTERACTION_LIMIT", "8"))
    max_attachments: int = int(os.getenv("COACH_MAX_ATTACHMENTS", "5"))
    max_image_attachments: int = int(os.getenv("COACH_MAX_IMAGE_ATTACHMENTS", "5"))
    max_image_bytes: int = int(os.getenv("COACH_MAX_IMAGE_BYTES", str(4 * 1024 * 1024)))
    max_document_bytes: int = int(os.getenv("COACH_MAX_DOCUMENT_BYTES", str(6 * 1024 * 1024)))
    max_attachment_chars: int = int(os.getenv("COACH_MAX_ATTACHMENT_CHARS", "12000"))
    max_pdf_pages: int = int(os.getenv("COACH_MAX_PDF_PAGES", "8"))
    strict_grounding_default: bool = os.getenv("COACH_STRICT_GROUNDING", "false").lower() == "true"
    not_found_message: str = os.getenv(
        "COACH_NOT_FOUND_MESSAGE",
        "I could not find this in your study material. Please upload or select the correct chapter/data.",
    )


coach_settings = CoachSettings()
