"""Strict platform-data retriever for Study Lab coach answers."""

import logging
import re
from typing import Any, Dict

from Logic.knowledge_graph import knowledge_graph
from Logic.content_pipeline import search_approved_content
from Logic.tools.knowledge_search import SECTION_FILE_MAP, search_knowledge_base

from .models import RetrievalResult
from .settings import coach_settings

logger = logging.getLogger("ai_educator.coach.retriever")


def _scope_value(scope: Dict[str, Any], key: str) -> str:
    return str(scope.get(key) or "").strip()


def _section_id(scope: Dict[str, Any]) -> str:
    return (
        _scope_value(scope, "section_id")
        or _scope_value(scope, "topic")
        or _scope_value(scope, "chapter")
        or "general"
    )


class GroundedRetriever:
    """Retrieves only ingested platform data and never falls back to outside facts."""

    _STOPWORDS = {
        "what", "why", "how", "explain", "define", "describe", "tell", "give",
        "with", "from", "about", "than", "more", "less", "into", "this", "that",
        "these", "those", "your", "please", "simple", "simply", "reactive", "reaction",
        "the", "and", "are", "is", "was", "were", "for", "does", "can",
    }

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")

    def _candidate_sections(self, question: str) -> list[str]:
        normalized_question = self._normalize(question).replace("_", " ")
        question_terms = {
            term
            for term in normalized_question.split()
            if len(term) > 2 and term not in self._STOPWORDS
        }
        candidates: list[str] = []

        matched_markdown = []
        for section_id in SECTION_FILE_MAP:
            readable = section_id.replace("_", " ")
            singular = readable[:-1] if readable.endswith("s") else readable
            if readable in question_terms or singular in question_terms:
                position = min(
                    index
                    for index in (
                        normalized_question.find(readable),
                        normalized_question.find(singular),
                    )
                    if index >= 0
                )
                matched_markdown.append((position, section_id))
        candidates.extend(section_id for _, section_id in sorted(matched_markdown))

        for concept in knowledge_graph.search_by_keyword(" ".join(question_terms), limit=1):
            concept_id = str(concept.get("concept_id") or "").strip()
            if concept_id and concept_id not in candidates:
                candidates.append(concept_id)

        return candidates

    def _retrieve_section(self, section_id: str, question: str, scope: Dict[str, Any]) -> RetrievalResult:
        planning_scope = _scope_value(scope, "catalog_source").lower() == "planning_manifest"
        if not planning_scope:
            try:
                approved = search_approved_content(
                    section_id=section_id,
                    question=question or section_id,
                    scope=scope,
                    max_chars=coach_settings.max_retrieval_chars,
                )
                approved_context = str(approved.get("context") or "").strip()
                if approved_context:
                    return RetrievalResult(
                        context=approved_context,
                        section_id=str(approved.get("section_id") or section_id),
                        source=str(approved.get("source") or "approved_content_pipeline"),
                        paragraphs_found=int(approved.get("paragraphs_found") or 0),
                        keywords_used=list(approved.get("keywords_used") or []),
                        scope={
                            "subject": _scope_value(scope, "subject"),
                            "chapter": _scope_value(scope, "chapter"),
                            "topic": _scope_value(scope, "topic"),
                            "section_id": section_id,
                            "source_pages": list(approved.get("source_pages") or []),
                        },
                        supported=True,
                    )
            except Exception:
                # Approved content is the primary source; a failure here silently
                # downgrades answers to the markdown knowledge base, so make it loud.
                logger.exception(
                    "Approved-content search failed; falling back to markdown knowledge base | section_id=%s",
                    section_id,
                )

        result = search_knowledge_base(
            section_id=section_id,
            question=question or section_id,
            max_paragraphs=coach_settings.max_retrieval_paragraphs,
            max_chars=coach_settings.max_retrieval_chars,
            scope=scope,
        )
        context = str(result.get("context") or "").strip()
        error = str(result.get("error") or "").strip()
        return RetrievalResult(
            context=context,
            section_id=str(result.get("section_id") or section_id),
            source=str(result.get("source") or ""),
            paragraphs_found=int(result.get("paragraphs_found") or 0),
            keywords_used=list(result.get("keywords_used") or []),
            scope={
                "subject": _scope_value(scope, "subject"),
                "chapter": _scope_value(scope, "chapter"),
                "topic": _scope_value(scope, "topic"),
                "section_id": section_id,
                "catalog_source": _scope_value(scope, "catalog_source"),
                "class_level": _scope_value(scope, "class_level"),
                "source_pages": list(result.get("source_pages") or []),
            },
            supported=bool(context and not error),
            error=error,
        )

    def retrieve(self, question: str, scope: Dict[str, Any]) -> RetrievalResult:
        section_id = _section_id(scope)
        if section_id not in {"general", "open", "any", "all"}:
            return self._retrieve_section(section_id, question, scope)

        candidates = self._candidate_sections(question)
        matches = []
        for candidate in candidates:
            result = self._retrieve_section(candidate, question, scope)
            if result.supported:
                matches.append(result)
            if len(matches) >= 3:
                break

        if len(matches) == 1:
            matches[0].scope["section_id"] = matches[0].section_id
            return matches[0]
        if matches:
            section_ids = [match.section_id for match in matches]
            return RetrievalResult(
                context="\n\n".join(
                    f"## Source: {match.section_id}\n{match.context}"
                    for match in matches
                )[: coach_settings.max_retrieval_chars],
                section_id=" + ".join(section_ids),
                source="hybrid_platform_data",
                paragraphs_found=sum(match.paragraphs_found for match in matches),
                keywords_used=sorted({keyword for match in matches for keyword in match.keywords_used}),
                scope={
                    "subject": _scope_value(scope, "subject"),
                    "chapter": "",
                    "topic": " + ".join(section_ids),
                    "section_id": " + ".join(section_ids),
                },
                supported=True,
            )

        return RetrievalResult(
            section_id="general",
            scope={
                "subject": _scope_value(scope, "subject"),
                "chapter": "",
                "topic": "",
                "section_id": "general",
            },
            error="No ingested study source matched the student's question.",
        )


grounded_retriever = GroundedRetriever()
