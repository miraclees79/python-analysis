from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Protocol, cast

from moj_system.config import OUTPUT_DIR
from moj_system.core.benchmark_catalog import INDEX_CATALOG

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_REVIEW_MODEL = "gemini-3.8-flash"
DEFAULT_FALLBACK_MODELS = ("gemini-3.1-flash-lite",)
_OUTPUT_TOKEN_RESERVE = 4096


@dataclass(frozen=True)
class _ModelQuota:
    requests_per_minute: int
    tokens_per_minute: int
    requests_per_day: int


_MODEL_QUOTAS = {
    "gemini-3.8-flash": _ModelQuota(5, 250_000, 20),
    "gemini-3.5-flash-lite": _ModelQuota(15, 250_000, 503),
    "gemini-3.1-flash-lite": _ModelQuota(15, 250_000, 503),
    "gemma-4-26b-a4b-it": _ModelQuota(30, 16_000, 14_400),
}
_UNKNOWN_MODEL_QUOTA = _ModelQuota(5, 16_000, 20)


class _GeminiResponse(Protocol):
    text: str | None


class _GeminiModels(Protocol):
    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: dict[str, object],
    ) -> _GeminiResponse: ...

class _GeminiClient(Protocol):
    models: _GeminiModels


_COMPONENT_SCHEMA = {
    "type": "object",
    "properties": {
        "index_key": {"type": "string", "enum": sorted(INDEX_CATALOG)},
        "weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "spread_bps": {"type": "number", "minimum": 0.0, "maximum": 10000.0},
        "hedged_to_pln": {"type": "boolean"},
    },
    "required": ["index_key", "weight", "spread_bps", "hedged_to_pln"],
}
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "components": {"type": "array", "items": _COMPONENT_SCHEMA},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
        "portfolio_spread_bps": {"type": "number", "minimum": 0.0, "maximum": 10000.0},
    },
    "required": ["components", "confidence", "reason"],
}
_BATCH_ITEM_SCHEMA = {
    "type": "object",
    "properties": {"id": {"type": "string"}, **_RESPONSE_SCHEMA["properties"]},
    "required": ["id", *_RESPONSE_SCHEMA["required"]],
}
_BATCH_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"results": {"type": "array", "items": _BATCH_ITEM_SCHEMA}},
    "required": ["results"],
}


class GeminiBenchmarkResolver:
    """Opt-in, cached Gemini interpretation with strict catalog validation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        fallback_models: tuple[str, ...] | None = None,
        cache_path: Path | None = None,
        client: _GeminiClient | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        state_changed: Callable[[], None] | None = None,
    ) -> None:
        self.api_key = os.environ.get("GEMINI_API_KEY") if api_key is None else api_key
        self.model = model or os.environ.get("GEMINI_BULK_MODEL", DEFAULT_MODEL)
        self.review_model = os.environ.get("GEMINI_REVIEW_MODEL", DEFAULT_REVIEW_MODEL)
        if fallback_models is None:
            configured_fallbacks = os.environ.get("GEMINI_FALLBACK_MODELS")
            fallback_models = (
                tuple(item.strip() for item in configured_fallbacks.split(",") if item.strip())
                if configured_fallbacks
                else DEFAULT_FALLBACK_MODELS
            )
        self.models = tuple(dict.fromkeys((self.model, *fallback_models)))
        self.cache_path = cache_path or OUTPUT_DIR / "benchmark_llm_cache.json"
        self.quota_path = self.cache_path.with_name("benchmark_llm_quota.json")
        self.client: _GeminiClient | None = client
        self._sleep = sleep
        self._clock = clock
        self._state_changed = state_changed
        self._last_request_at: dict[str, float] = {}
        self._server_error_counts: dict[str, int] = {}
        self._disabled_models: set[str] = set()
        self._cache: dict[str, dict[str, object]] | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key or self.client)

    def resolve_kid_text(self, benchmark_text: str) -> dict[str, object] | None:
        prompt = (
            "Parse this Polish fund benchmark description into index components. "
            "Use only index_key values in the supplied schema. Preserve all weights and basis-point spreads. "
            "A rate spread belongs to its rate component; "
            "an unassigned final spread belongs to the whole portfolio. "
            "Set hedged_to_pln only when the text explicitly says it is hedged to PLN. "
            "Do not infer missing constituents. Return low confidence if the text is ambiguous.\n\n"
            f"BENCHMARK: {benchmark_text}"
        )
        return self._resolve("kid", {"benchmark": benchmark_text}, prompt)

    def propose_proxy(
        self,
        name: str,
        category: str,
        classification: str,
        risk_level: str,
    ) -> dict[str, object] | None:
        payload = {
            "name": name,
            "category": category,
            "classification": classification,
            "risk_level": risk_level,
        }
        prompt = (
            "Propose a transparent market proxy benchmark for this Polish investment fund. "
            "Use only index_key values in the supplied schema, and use available broad proxies if a precise "
            "index is not represented. Do not claim that the proxy is the official fund benchmark. "
            "Weights must sum to 1.0. Use hedged_to_pln only when evidence is explicit; otherwise false. "
            "Return low confidence if the strategy is too broad or alternative to proxy responsibly.\n\n"
            f"FUND METADATA: {json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
        )
        return self._resolve("proxy", payload, prompt)

    def resolve_many(
        self,
        requests: dict[str, tuple[str, dict[str, str]]],
        batch_size: int = 25,
        primary_model: str | None = None,
    ) -> dict[str, dict[str, object]]:
        """Resolve KID benchmarks/proxies in quota-efficient, independently validated batches."""
        if not self.enabled or not requests:
            return {}

        resolved: dict[str, dict[str, object]] = {}
        pending_by_key: dict[str, tuple[str, dict[str, str]]] = {}
        ids_by_key: dict[str, list[str]] = {}
        for item_id, (operation, payload) in requests.items():
            cache_key = self._cache_key(operation, payload)
            ids_by_key.setdefault(cache_key, []).append(item_id)
            cached = self._read_cache().get(cache_key)
            if cached is not None:
                validated = self._validate_response(cached)
                if validated is not None:
                    for duplicate_id in ids_by_key[cache_key]:
                        resolved[duplicate_id] = {**cached, **validated}
                    continue
                logging.warning("Ignoring invalid cached Gemini benchmark result.")
            pending_by_key[cache_key] = (operation, payload)

        pending_keys = list(pending_by_key)
        if not pending_keys:
            return resolved
        client = self._get_client()
        if client is None:
            return resolved

        batch_size = max(1, batch_size)
        model_chain = tuple(dict.fromkeys(((primary_model,) if primary_model else ()) + self.models))
        for model_name in model_chain:
            if model_name in self._disabled_models:
                continue
            for offset in range(0, len(pending_keys), batch_size):
                batch_keys = [
                    key for key in pending_keys[offset : offset + batch_size] if key in pending_by_key
                ]
                if not batch_keys:
                    continue
                records = [
                    {
                        "id": key,
                        "operation": pending_by_key[key][0],
                        "payload": pending_by_key[key][1],
                    }
                    for key in batch_keys
                ]
                prompt = (
                    "Resolve each fund independently. Parse 'kid' benchmarks; for 'proxy', propose a "
                    "transparent market proxy, not an official benchmark. Use only schema index_key values. "
                    "Preserve weights and spreads. Set hedged_to_pln only when explicit. "
                    "Do not infer missing "
                    "constituents. Return one result for each input id. "
                    "If an item is ambiguous, return confidence below 0.65 for that item.\n\n"
                    f"ITEMS: {json.dumps(records, ensure_ascii=False, sort_keys=True)}"
                )
                response = self._request_with_retries(
                    client,
                    model_name,
                    prompt,
                    response_schema=_BATCH_RESPONSE_SCHEMA,
                )
                if response is None or not isinstance(response.text, str):
                    continue
                try:
                    response_data = json.loads(response.text)
                except (json.JSONDecodeError, TypeError):
                    continue
                batch_results = response_data.get("results") if isinstance(response_data, dict) else None
                if not isinstance(batch_results, list):
                    continue
                valid_ids: set[str] = set()
                for item in batch_results:
                    if not isinstance(item, dict):
                        continue
                    cache_key = item.get("id")
                    if not isinstance(cache_key, str) or cache_key not in batch_keys:
                        continue
                    validated = self._validate_response(item)
                    if validated is None:
                        continue
                    operation, _ = pending_by_key[cache_key]
                    cache_record = {
                        **validated,
                        "operation": operation,
                        "model": model_name,
                        "cached_at": datetime.now(UTC).isoformat(),
                    }
                    self._read_cache()[cache_key] = cache_record
                    for item_id in ids_by_key[cache_key]:
                        resolved[item_id] = cache_record
                    valid_ids.add(cache_key)
                if valid_ids:
                    self._write_cache()
                    for cache_key in valid_ids:
                        pending_by_key.pop(cache_key, None)
            pending_keys = [key for key in pending_keys if key in pending_by_key]
            if not pending_keys:
                break
        return resolved

    def _get_client(self) -> _GeminiClient | None:
        if self.client is not None:
            return self.client
        try:
            from google import genai

            self.client = cast(_GeminiClient, genai.Client(api_key=self.api_key))
        except Exception as exc:
            logging.warning("Gemini client initialization failed (%s).", type(exc).__name__)
        return self.client

    @staticmethod
    def _cache_key(operation: str, payload: dict[str, str]) -> str:
        return hashlib.sha256(
            json.dumps(
                {"version": 2, "operation": operation, "payload": payload},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8"),
        ).hexdigest()

    def _resolve(self, operation: str, payload: dict[str, str], prompt: str) -> dict[str, object] | None:
        if not self.enabled:
            return None
        cache_key = self._cache_key(operation, payload)
        cached = self._read_cache().get(cache_key)
        if cached is not None:
            validated_cache = self._validate_response(cached)
            if validated_cache is not None:
                return {**cached, **validated_cache}
            logging.warning("Ignoring invalid cached Gemini benchmark result.")

        client = self._get_client()
        if client is None:
            return None

        model_chain = (
            (self.review_model, *self.models)
            if operation == "proxy"
            else self.models
        )
        for model_name in dict.fromkeys(model_chain):
            if model_name in self._disabled_models:
                continue
            response = self._request_with_retries(client, model_name, prompt)
            if response is None:
                continue
            response_text = response.text
            if not isinstance(response_text, str):
                continue
            try:
                result = self._validate_response(json.loads(response_text))
            except (json.JSONDecodeError, TypeError):
                result = None
            if result is None:
                logging.warning("Gemini returned an invalid benchmark response (%s).", model_name)
                continue

            cache_record = {
                **result,
                "operation": operation,
                "model": model_name,
                "cached_at": datetime.now(UTC).isoformat(),
            }
            self._read_cache()[cache_key] = cache_record
            self._write_cache()
            return cache_record
        return None

    def _request_with_retries(
        self,
        client: _GeminiClient,
        model_name: str,
        prompt: str,
        response_schema: dict[str, object] = _RESPONSE_SCHEMA,
    ) -> _GeminiResponse | None:
        quota = _MODEL_QUOTAS.get(model_name, _UNKNOWN_MODEL_QUOTA)
        for attempt in range(3):
            if not self._reserve_quota(model_name, prompt, quota):
                return None
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "temperature": 0,
                        "max_output_tokens": _OUTPUT_TOKEN_RESERVE,
                        "response_mime_type": "application/json",
                        "response_schema": response_schema,
                    },
                )
                self._server_error_counts[model_name] = 0
                return response
            except Exception as exc:
                status = self._status_code(exc)
                if status == 429:
                    logging.warning("Gemini quota/rate limit reached for %s (HTTP 429).", model_name)
                    self._disabled_models.add(model_name)
                    return None
                if status in {500, 503, 504}:
                    self._server_error_counts[model_name] = self._server_error_counts.get(model_name, 0) + 1
                    if self._server_error_counts[model_name] >= 3:
                        self._disabled_models.add(model_name)
                if status not in {500, 503, 504} or attempt == 2:
                    logging.warning(
                        "Gemini request failed for %s (HTTP %s, %s).",
                        model_name,
                        status or "unknown",
                        type(exc).__name__,
                    )
                    return None
                delay = self._retry_after(exc)
                if delay is None:
                    delay = min(30.0, (2**attempt) + random.uniform(0.0, 1.0))
                logging.warning(
                    "Gemini returned HTTP %s for %s; retry %d/2 in %.1f seconds.",
                    status,
                    model_name,
                    attempt + 1,
                    delay,
                )
                self._sleep(delay)
        return None

    def _reserve_quota(self, model_name: str, prompt: str, quota: _ModelQuota) -> bool:
        estimated_tokens = math.ceil(len(prompt) / 4) + _OUTPUT_TOKEN_RESERVE
        if estimated_tokens > quota.tokens_per_minute:
            logging.warning("Gemini prompt exceeds the conservative TPM budget for %s.", model_name)
            return False

        while True:
            now = self._clock()
            day_key = datetime.fromtimestamp(now, UTC).date().isoformat()
            minute_key = str(int(now // 60))
            usage = self._read_quota_usage(day_key)
            request_counts = cast(dict[str, int], usage["requests"])
            minute_counts = cast(dict[str, dict[str, dict[str, int]]], usage["minutes"])
            last_requests = cast(dict[str, float], usage["last_request_at"])
            daily_requests = request_counts.get(model_name, 0)
            if daily_requests >= quota.requests_per_day:
                logging.warning("Gemini daily request budget exhausted for %s.", model_name)
                self._disabled_models.add(model_name)
                return False

            minute_usage = minute_counts.get(minute_key, {}).get(model_name, {})
            request_count = minute_usage.get("requests", 0)
            token_count = minute_usage.get("tokens", 0)
            last_request_at = max(
                self._last_request_at.get(model_name, 0.0),
                last_requests.get(model_name, 0.0),
            ) or None
            wait_seconds = 0.0
            if last_request_at is not None:
                elapsed = now - last_request_at
                wait_seconds = max(wait_seconds, (60.0 / quota.requests_per_minute) - elapsed)
            if (
                request_count >= quota.requests_per_minute
                or token_count + estimated_tokens > quota.tokens_per_minute
            ):
                wait_seconds = max(wait_seconds, 60.0 - (now % 60.0))
            if wait_seconds > 0.0:
                self._sleep(wait_seconds)
                continue

            request_counts[model_name] = daily_requests + 1
            model_minute = minute_counts.setdefault(minute_key, {}).setdefault(
                model_name,
                {"requests": 0, "tokens": 0},
            )
            model_minute["requests"] += 1
            model_minute["tokens"] += estimated_tokens
            self._last_request_at[model_name] = now
            last_requests[model_name] = now
            self._write_quota_usage(usage)
            return True

    def _read_quota_usage(self, day_key: str) -> dict[str, object]:
        try:
            raw = json.loads(self.quota_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            raw = {}
        if not isinstance(raw, dict) or raw.get("day") != day_key:
            raw = {"day": day_key}
        raw_requests = raw.get("requests")
        requests = {
            str(model_name): int(count)
            for model_name, count in raw_requests.items()
            if isinstance(raw_requests, dict) and isinstance(count, int) and count >= 0
        } if isinstance(raw_requests, dict) else {}
        raw_minutes = raw.get("minutes")
        minutes: dict[str, dict[str, dict[str, int]]] = {}
        if isinstance(raw_minutes, dict):
            for minute_key, model_data in raw_minutes.items():
                if not isinstance(model_data, dict):
                    continue
                valid_models: dict[str, dict[str, int]] = {}
                for model_name, values in model_data.items():
                    if not isinstance(values, dict):
                        continue
                    request_count = values.get("requests")
                    token_count = values.get("tokens")
                    if (
                        isinstance(request_count, int)
                        and request_count >= 0
                        and isinstance(token_count, int)
                        and token_count >= 0
                    ):
                        valid_models[str(model_name)] = {
                            "requests": request_count,
                            "tokens": token_count,
                        }
                if valid_models:
                    minutes[str(minute_key)] = valid_models
        raw_last_requests = raw.get("last_request_at")
        last_request_at = {
            str(model_name): float(timestamp)
            for model_name, timestamp in raw_last_requests.items()
            if isinstance(raw_last_requests, dict)
            and isinstance(timestamp, (int, float))
            and timestamp >= 0.0
        } if isinstance(raw_last_requests, dict) else {}
        return {
            "day": day_key,
            "requests": requests,
            "minutes": minutes,
            "last_request_at": last_request_at,
        }

    def _write_quota_usage(self, usage: dict[str, object]) -> None:
        self.quota_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.quota_path.with_suffix(f"{self.quota_path.suffix}.tmp")
        temp_path.write_text(json.dumps(usage, indent=2), encoding="utf-8")
        temp_path.replace(self.quota_path)
        self._notify_state_changed()

    def _notify_state_changed(self) -> None:
        if self._state_changed is None:
            return
        try:
            self._state_changed()
        except Exception as exc:
            logging.warning("Could not synchronize Gemini state (%s).", type(exc).__name__)

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        response = getattr(exc, "response", None)
        for candidate in (
            getattr(exc, "code", None),
            getattr(exc, "status_code", None),
            getattr(response, "status_code", None),
        ):
            try:
                if candidate is not None:
                    return int(candidate)
            except (TypeError, ValueError):
                continue
        match = re.search(r"\b(429|500|503|504)\b", str(exc))
        return int(match.group(1)) if match else None

    @staticmethod
    def _retry_after(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {})
        value = headers.get("Retry-After") if hasattr(headers, "get") else None
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(value))
                return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

    @staticmethod
    def _validate_response(data: object) -> dict[str, object] | None:
        if not isinstance(data, dict):
            return None
        components = data.get("components")
        confidence = data.get("confidence")
        reason = data.get("reason")
        if not isinstance(components, list) or not components or not isinstance(confidence, (int, float)):
            return None
        if not 0.65 <= float(confidence) <= 1.0 or not isinstance(reason, str):
            return None

        validated: list[dict[str, object]] = []
        for component in components:
            if not isinstance(component, dict):
                return None
            index_key = component.get("index_key")
            weight = component.get("weight")
            spread_bps = component.get("spread_bps", 0.0)
            hedged_to_pln = component.get("hedged_to_pln", False)
            if index_key not in INDEX_CATALOG:
                return None
            if not isinstance(weight, (int, float)) or not 0.0 < float(weight) <= 1.0:
                return None
            if not isinstance(spread_bps, (int, float)) or not 0.0 <= float(spread_bps) <= 10000.0:
                return None
            if not isinstance(hedged_to_pln, bool):
                return None
            validated.append(
                {
                    "index_key": index_key,
                    "weight": float(weight),
                    "spread_bps": float(spread_bps),
                    "hedged_to_pln": hedged_to_pln,
                },
            )
        weight_sum = sum(cast(float, component["weight"]) for component in validated)
        if abs(weight_sum - 1.0) > 0.01:
            return None
        return {
            "components": validated,
            "confidence": float(confidence),
            "reason": reason,
            "portfolio_spread_bps": float(data.get("portfolio_spread_bps", 0.0)),
        }

    def _read_cache(self) -> dict[str, dict[str, object]]:
        if self._cache is not None:
            return self._cache
        self._cache = {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._cache = {key: value for key, value in data.items() if isinstance(value, dict)}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        return self._cache

    def _write_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.cache_path.with_suffix(f"{self.cache_path.suffix}.tmp")
        temp_path.write_text(json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.cache_path)
        self._notify_state_changed()
