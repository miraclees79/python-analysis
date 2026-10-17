from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from moj_system.config import OUTPUT_DIR
from moj_system.core.benchmark_catalog import INDEX_CATALOG

DEFAULT_MODEL = "gemini-3.8-flash"


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


class GeminiBenchmarkResolver:
    """Opt-in, cached Gemini interpretation with strict catalog validation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        cache_path: Path | None = None,
        client: _GeminiClient | None = None,
    ) -> None:
        self.api_key = os.environ.get("GEMINI_API_KEY") if api_key is None else api_key
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self.cache_path = cache_path or OUTPUT_DIR / "benchmark_llm_cache.json"
        self.client: _GeminiClient | None = client
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

    def _resolve(self, operation: str, payload: dict[str, str], prompt: str) -> dict[str, object] | None:
        if not self.enabled:
            return None
        cache_key = hashlib.sha256(
            json.dumps(
                {"version": 1, "model": self.model, "operation": operation, "payload": payload},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8"),
        ).hexdigest()
        cached = self._read_cache().get(cache_key)
        if cached is not None:
            validated_cache = self._validate_response(cached)
            if validated_cache is not None:
                return {**cached, **validated_cache}
            logging.warning("Ignoring invalid cached Gemini benchmark result.")

        try:
            client = self.client
            if client is None:
                from google import genai

                client = cast(_GeminiClient, genai.Client(api_key=self.api_key))
                self.client = client
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": 0,
                    "response_mime_type": "application/json",
                    "response_schema": _RESPONSE_SCHEMA,
                },
            )
            response_text = response.text
            if not isinstance(response_text, str):
                return None
            result = self._validate_response(json.loads(response_text))
        except Exception as exc:
            logging.warning("Gemini benchmark resolution failed (%s).", type(exc).__name__)
            return None

        if result is None:
            return None
        cache_record = {
            **result,
            "operation": operation,
            "model": self.model,
            "cached_at": datetime.now(UTC).isoformat(),
        }
        self._read_cache()[cache_key] = cache_record
        self._write_cache()
        return cache_record

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
        weight_sum = sum(component["weight"] for component in validated)
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
