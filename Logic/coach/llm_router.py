"""Cost-aware, multi-provider model routing for Study Lab coach calls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
import logging
import os
from types import SimpleNamespace
import threading
import time
from typing import Any, Dict, Iterable, Iterator, List

from groq import Groq
import requests
from sqlalchemy import func
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from .costing import estimate_messages_tokens, estimate_model_cost_usd, estimate_text_tokens
from .settings import coach_settings

logger = logging.getLogger("ai_educator.coach.llm_router")


PROVIDER_ALIASES = {
    "groq": "groq",
    "openrouter": "openrouter",
    "openai": "openai",
}

# Groq periodically retires model ids. Keep old deployment variables usable
# by routing known retired ids to the current non-agentic production model.
RETIRED_GROQ_MODELS = {
    "meta-llama/llama-4-scout-17b-16e-instruct": "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile": "openai/gpt-oss-120b",
}


@dataclass(frozen=True)
class ModelRoute:
    provider: str
    model: str
    reason: str
    estimated_route_cost_usd: float


@dataclass
class ModelCallRecord:
    role: str
    model: str
    provider: str
    mode: str
    status: str
    latency_ms: int
    attempt: int
    fallback: bool = False
    error: str = ""
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    estimated_route_cost_usd: float = 0.0
    route_reason: str = ""
    budget_action: str = "allowed"
    truncated: bool = False


class LLMRouter:
    """
    Route cheap tasks cheaply, fail over across providers, and retain safe traces.

    Provider setup is intentionally environment-backed:
    - COACH_PROVIDER_ORDER=groq,openrouter,openai
    - OPENROUTER_API_KEY / OPENROUTER_TUTOR_MODEL / OPENROUTER_FAST_MODEL ...
    - OPENAI_API_KEY / OPENAI_TUTOR_MODEL / OPENAI_FAST_MODEL ...
    """

    def __init__(self) -> None:
        self._groq = None
        self._provider_clients: dict[str, Any] = {}
        self._local = threading.local()

    # =====================================================
    # Env helpers
    # =====================================================
    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_float(name: str, default: float = 0.0) -> float:
        try:
            return float(os.getenv(name, str(default)) or default)
        except ValueError:
            return default

    @staticmethod
    def _env_int(name: str, default: int = 0) -> int:
        try:
            return int(os.getenv(name, str(default)) or default)
        except ValueError:
            return default

    def _provider_order(self) -> List[str]:
        raw = os.getenv("COACH_PROVIDER_ORDER") or coach_settings.provider or "groq"
        providers: List[str] = []
        for item in raw.split(","):
            provider = PROVIDER_ALIASES.get(item.strip().lower(), item.strip().lower())
            if provider and provider not in providers:
                providers.append(provider)
        return providers or ["groq"]

    @staticmethod
    def _role_key(role: str, complexity: str) -> str:
        if role == "vision":
            return "VISION"
        if role == "reviewer":
            return "REVIEW"
        if role == "profiler" or complexity == "fast":
            return "FAST"
        if complexity == "deep":
            return "DEEP"
        return "TUTOR"

    def _provider_api_key(self, provider: str) -> str:
        if provider == "groq":
            return os.getenv("GROQ_API_KEY", "")
        return os.getenv(f"{provider.upper()}_API_KEY", "")

    def _provider_base_url(self, provider: str) -> str:
        if provider == "openrouter":
            return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        if provider == "openai":
            return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        return os.getenv(f"{provider.upper()}_BASE_URL", "").rstrip("/")

    def _provider_available(self, provider: str) -> bool:
        if provider in self._provider_clients:
            return True
        if provider == "groq" and self._groq is not None:
            return True
        return bool(self._provider_api_key(provider))

    def _provider_model(self, provider: str, role: str, complexity: str) -> str:
        role_key = self._role_key(role, complexity)
        prefix = provider.upper()
        if provider == "groq":
            return self.model_for(role, complexity=complexity)
        return (
            os.getenv(f"{prefix}_{role_key}_MODEL")
            or os.getenv(f"{prefix}_MODEL")
            or ""
        ).strip()

    def _provider_fallback_model(self, provider: str) -> str:
        if provider == "groq":
            model = os.getenv("GROQ_FALLBACK_MODEL", coach_settings.fallback_model)
            return self._normalize_groq_model(model, role="fallback")
        return (
            os.getenv(f"{provider.upper()}_FALLBACK_MODEL")
            or os.getenv(f"{provider.upper()}_MODEL")
            or ""
        ).strip()

    # =====================================================
    # Clients
    # =====================================================
    def _groq_client(self) -> Groq:
        if self._groq is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY is not configured.")
            self._groq = Groq(api_key=api_key, timeout=coach_settings.llm_timeout_seconds)
        return self._groq

    def _headers_for(self, provider: str) -> Dict[str, str]:
        api_key = self._provider_api_key(provider)
        if not api_key:
            raise RuntimeError(f"{provider.upper()}_API_KEY is not configured.")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if provider == "openrouter":
            site_url = os.getenv("OPENROUTER_SITE_URL", "https://agentifyai.in")
            app_name = os.getenv("OPENROUTER_APP_NAME", "AgentifyAI")
            headers["HTTP-Referer"] = site_url
            headers["X-Title"] = app_name
        return headers

    @staticmethod
    def _completion_namespace(content: str, finish_reason: str = "") -> Any:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason=finish_reason or None,
                )
            ]
        )

    @staticmethod
    def _chunk_namespace(content: str, finish_reason: str = "") -> Any:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=content),
                    finish_reason=finish_reason or None,
                )
            ]
        )

    def _http_complete(
        self,
        *,
        provider: str,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool,
        **kwargs: Any,
    ) -> Any:
        base_url = self._provider_base_url(provider)
        if not base_url:
            raise RuntimeError(f"{provider} base URL is not configured.")
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs,
        }
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=self._headers_for(provider),
            json=payload,
            timeout=coach_settings.llm_timeout_seconds,
            stream=stream,
        )
        response.raise_for_status()
        if stream:
            return self._iter_http_stream(response)
        data = response.json()
        choice = data.get("choices", [{}])[0]
        content = str(choice.get("message", {}).get("content") or "").strip()
        return self._completion_namespace(content, str(choice.get("finish_reason") or ""))

    def _iter_http_stream(self, response: Any) -> Iterator[Any]:
        with response:
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line:
                    continue
                if line == "[DONE]":
                    break
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                choice = payload.get("choices", [{}])[0]
                delta = choice.get("delta", {}).get("content") or ""
                finish_reason = str(choice.get("finish_reason") or "")
                if delta or finish_reason:
                    yield self._chunk_namespace(delta, finish_reason)

    @staticmethod
    def _is_transient_error(exc: BaseException) -> bool:
        status = getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        return status in {429, 500, 502, 503, 504} or isinstance(
            exc, (requests.Timeout, requests.ConnectionError)
        )

    @retry(
        retry=retry_if_exception(_is_transient_error.__func__),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=4.0),
        reraise=True,
    )
    def _create_completion(
        self,
        *,
        route: ModelRoute,
        messages: List[Dict[str, Any]],
        stream: bool,
        **kwargs: Any,
    ) -> Any:
        provider = route.provider
        if provider in self._provider_clients:
            return self._provider_clients[provider].chat.completions.create(
                model=route.model,
                messages=messages,
                stream=stream,
                **kwargs,
            )
        if provider == "groq":
            return self._groq_client().chat.completions.create(
                model=route.model,
                messages=messages,
                stream=stream,
                **kwargs,
            )
        return self._http_complete(
            provider=provider,
            model=route.model,
            messages=messages,
            stream=stream,
            **kwargs,
        )

    # =====================================================
    # Public trace API
    # =====================================================
    def begin_turn(self, turn_id: str) -> None:
        self._local.turn_id = turn_id
        self._local.records = []
        self._local.turn_estimated_cost_usd = 0.0

    def records(self) -> List[Dict[str, Any]]:
        return [asdict(record) for record in list(getattr(self._local, "records", []))]

    def _record(self, record: ModelCallRecord) -> None:
        records = list(getattr(self._local, "records", []))
        records.append(record)
        self._local.records = records[-32:]

    # =====================================================
    # Routing and budgets
    # =====================================================
    @staticmethod
    def _normalize_groq_model(model: str, *, role: str) -> str:
        normalized = str(model or "").strip()
        if normalized not in RETIRED_GROQ_MODELS:
            return normalized
        # The retired Scout id was also the old image route. Mapping image
        # input to a text-only model would be unsafe and misleading; fail fast
        # so attachment handling can return its honest material warning.
        if role == "vision":
            return ""
        return RETIRED_GROQ_MODELS[normalized]

    def model_for(self, role: str, complexity: str = "balanced") -> str:
        if role == "vision":
            model = coach_settings.vision_model
        elif role == "profiler" or complexity == "fast":
            model = coach_settings.fast_model
        elif role == "reviewer":
            model = coach_settings.review_model
        elif complexity == "deep":
            model = coach_settings.deep_model
        else:
            model = coach_settings.tutor_model
        return self._normalize_groq_model(model, role=role)

    def _estimate_route_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        return estimate_model_cost_usd(
            model,
            input_tokens,
            output_tokens,
            provider=provider,
        )

    def _daily_spend_usd(self) -> float:
        daily_budget = self._env_float("COACH_DAILY_BUDGET_USD", 0.0)
        if daily_budget <= 0:
            return 0.0
        try:
            from database import SessionLocal
            from models import ModelToolTrace

            db = SessionLocal()
            try:
                cutoff = datetime.utcnow() - timedelta(hours=24)
                value = (
                    db.query(func.sum(ModelToolTrace.estimated_cost_usd))
                    .filter(ModelToolTrace.trace_type == "model")
                    .filter(ModelToolTrace.created_at >= cutoff)
                    .scalar()
                )
                return float(value or 0.0)
            finally:
                db.close()
        except Exception:
            return 0.0

    def _budget_allows(self, route: ModelRoute) -> bool:
        if not self._env_bool("COACH_BUDGET_ROUTING", True):
            return True
        turn_budget = self._env_float("COACH_TURN_BUDGET_USD", 0.0)
        daily_budget = self._env_float("COACH_DAILY_BUDGET_USD", 0.0)
        turn_spend = float(getattr(self._local, "turn_estimated_cost_usd", 0.0) or 0.0)
        if turn_budget > 0 and turn_spend + route.estimated_route_cost_usd > turn_budget:
            return False
        if daily_budget > 0 and self._daily_spend_usd() + route.estimated_route_cost_usd > daily_budget:
            return False
        return True

    def _candidate_routes(
        self,
        *,
        role: str,
        complexity: str,
        input_tokens: int,
        output_tokens: int,
    ) -> List[ModelRoute]:
        primary_routes: List[ModelRoute] = []
        fallback_routes: List[ModelRoute] = []
        for provider in self._provider_order():
            if not self._provider_available(provider):
                continue
            primary = self._provider_model(provider, role, complexity)
            if primary:
                primary_routes.append(ModelRoute(
                    provider=provider,
                    model=primary,
                    reason=f"{provider}:{self._role_key(role, complexity).lower()}",
                    estimated_route_cost_usd=self._estimate_route_cost(provider, primary, input_tokens, output_tokens),
                ))
            if role != "vision":
                fallback = self._provider_fallback_model(provider)
                if fallback and fallback != primary:
                    fallback_routes.append(ModelRoute(
                        provider=provider,
                        model=fallback,
                        reason=f"{provider}:fallback",
                        estimated_route_cost_usd=self._estimate_route_cost(provider, fallback, input_tokens, output_tokens),
                    ))

        routes = primary_routes + fallback_routes
        max_attempts = max(1, self._env_int("COACH_LLM_MAX_ATTEMPTS", coach_settings.llm_max_attempts))
        preference = os.getenv("COACH_ROUTE_PREFERENCE", "balanced").strip().lower()
        if preference == "lowest_cost" or role == "profiler" or complexity == "fast":
            routes.sort(key=lambda route: route.estimated_route_cost_usd)
        return routes[: max_attempts]

    def _record_call(
        self,
        *,
        role: str,
        route: ModelRoute,
        mode: str,
        status: str,
        started_at: float,
        attempt: int,
        input_tokens: int,
        output_text: Any = "",
        fallback: bool = False,
        error: str = "",
        budget_action: str = "allowed",
        truncated: bool = False,
    ) -> None:
        output_tokens = estimate_text_tokens(output_text)
        actual_cost = estimate_model_cost_usd(
            route.model,
            input_tokens,
            output_tokens,
            provider=route.provider,
        )
        if status == "success":
            self._local.turn_estimated_cost_usd = (
                float(getattr(self._local, "turn_estimated_cost_usd", 0.0) or 0.0) + actual_cost
            )
        self._record(ModelCallRecord(
            role=role,
            model=route.model,
            provider=route.provider,
            mode=mode,
            status=status,
            latency_ms=round((time.perf_counter() - started_at) * 1000),
            attempt=attempt,
            fallback=fallback,
            error=error[:240],
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            estimated_cost_usd=actual_cost,
            estimated_route_cost_usd=route.estimated_route_cost_usd,
            route_reason=route.reason,
            budget_action=budget_action,
            truncated=truncated,
        ))

    @staticmethod
    def _chunk_text(chunk: Any) -> str:
        try:
            return chunk.choices[0].delta.content or ""
        except Exception:
            return ""

    @staticmethod
    def _chunk_finish_reason(chunk: Any) -> str:
        try:
            return str(chunk.choices[0].finish_reason or "")
        except Exception:
            return ""

    def complete(
        self,
        role: str,
        messages: Iterable[Dict[str, Any]],
        complexity: str = "balanced",
        **kwargs: Any,
    ) -> str:
        last_error: Exception | None = None
        message_rows = list(messages)
        input_tokens = estimate_messages_tokens(message_rows)
        expected_output_tokens = int(kwargs.get("max_tokens") or self._env_int("COACH_ROUTE_OUTPUT_TOKEN_ESTIMATE", 700))
        candidates = self._candidate_routes(
            role=role,
            complexity=complexity,
            input_tokens=input_tokens,
            output_tokens=expected_output_tokens,
        )
        if not candidates:
            raise RuntimeError("No configured LLM provider route is available.")

        skipped_for_budget = 0
        for attempt, route in enumerate(candidates, start=1):
            if not self._budget_allows(route):
                skipped_for_budget += 1
                self._record_call(
                    role=role,
                    route=route,
                    mode="complete",
                    status="skipped",
                    started_at=time.perf_counter(),
                    attempt=attempt,
                    input_tokens=input_tokens,
                    fallback=attempt > 1,
                    error="Route skipped because it would exceed the configured coach budget.",
                    budget_action="skipped",
                )
                continue

            started_at = time.perf_counter()
            try:
                response = self._create_completion(
                    route=route,
                    messages=message_rows,
                    stream=False,
                    **kwargs,
                )
                content = (response.choices[0].message.content or "").strip()
                finish_reason = str(getattr(response.choices[0], "finish_reason", "") or "")
                truncated = finish_reason == "length"
                if truncated:
                    logger.warning(
                        "Model output truncated at max_tokens | role=%s model=%s", role, route.model
                    )
                self._record_call(
                    role=role,
                    route=route,
                    mode="complete",
                    status="success",
                    started_at=started_at,
                    attempt=attempt,
                    input_tokens=input_tokens,
                    output_text=content,
                    fallback=attempt > 1,
                    truncated=truncated,
                )
                return content
            except Exception as exc:
                last_error = exc
                self._record_call(
                    role=role,
                    route=route,
                    mode="complete",
                    status="error",
                    started_at=started_at,
                    attempt=attempt,
                    input_tokens=input_tokens,
                    fallback=attempt > 1,
                    error=str(exc),
                )

        if skipped_for_budget == len(candidates):
            raise RuntimeError("Coach LLM budget exhausted before a safe route could be selected.")
        raise last_error or RuntimeError(f"No model route was available for role '{role}'.")

    def stream(
        self,
        role: str,
        messages: Iterable[Dict[str, Any]],
        complexity: str = "balanced",
        **kwargs: Any,
    ) -> Iterator[Any]:
        message_rows = list(messages)
        input_tokens = estimate_messages_tokens(message_rows)
        expected_output_tokens = int(kwargs.get("max_tokens") or self._env_int("COACH_ROUTE_OUTPUT_TOKEN_ESTIMATE", 900))
        candidates = self._candidate_routes(
            role=role,
            complexity=complexity,
            input_tokens=input_tokens,
            output_tokens=expected_output_tokens,
        )
        if not candidates:
            raise RuntimeError("No configured streaming LLM provider route is available.")

        last_error: Exception | None = None
        emitted_chunks = 0
        skipped_for_budget = 0

        for attempt, route in enumerate(candidates, start=1):
            if not self._budget_allows(route):
                skipped_for_budget += 1
                self._record_call(
                    role=role,
                    route=route,
                    mode="stream",
                    status="skipped",
                    started_at=time.perf_counter(),
                    attempt=attempt,
                    input_tokens=input_tokens,
                    fallback=attempt > 1,
                    error="Route skipped because it would exceed the configured coach budget.",
                    budget_action="skipped",
                )
                continue

            started_at = time.perf_counter()
            output_parts: list[str] = []
            finish_reason = ""
            try:
                response = self._create_completion(
                    route=route,
                    messages=message_rows,
                    stream=True,
                    **kwargs,
                )
                for chunk in response:
                    emitted_chunks += 1
                    chunk_text = self._chunk_text(chunk)
                    if chunk_text:
                        output_parts.append(chunk_text)
                    finish_reason = self._chunk_finish_reason(chunk) or finish_reason
                    yield chunk
                truncated = finish_reason == "length"
                if truncated:
                    logger.warning(
                        "Streamed output truncated at max_tokens | role=%s model=%s", role, route.model
                    )
                self._record_call(
                    role=role,
                    route=route,
                    mode="stream",
                    status="success",
                    started_at=started_at,
                    attempt=attempt,
                    input_tokens=input_tokens,
                    output_text="".join(output_parts),
                    fallback=attempt > 1,
                    truncated=truncated,
                )
                return
            except Exception as exc:
                last_error = exc
                self._record_call(
                    role=role,
                    route=route,
                    mode="stream",
                    status="error",
                    started_at=started_at,
                    attempt=attempt,
                    input_tokens=input_tokens,
                    output_text="".join(output_parts),
                    fallback=attempt > 1,
                    error=str(exc),
                )
                if emitted_chunks:
                    raise

        if skipped_for_budget == len(candidates):
            raise RuntimeError("Coach streaming LLM budget exhausted before a safe route could be selected.")
        raise last_error or RuntimeError(f"No streaming model route was available for role '{role}'.")


llm_router = LLMRouter()
