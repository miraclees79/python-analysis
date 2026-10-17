from __future__ import annotations

import re
from dataclasses import dataclass

from moj_system.core.benchmark_catalog import normalize_index_text, resolve_index_key


@dataclass(frozen=True)
class BenchmarkComponent:
    index_key: str
    weight: float
    spread_bps: float = 0.0
    hedged_to_pln: bool = False


@dataclass(frozen=True)
class ParsedBenchmark:
    components: tuple[BenchmarkComponent, ...]
    unresolved: tuple[str, ...]
    weight_sum: float
    is_no_benchmark: bool = False
    portfolio_spread_bps: float = 0.0

    @property
    def is_resolved(self) -> bool:
        return bool(self.components) and not self.unresolved and abs(self.weight_sum - 1.0) <= 0.01


_WEIGHT_RE = re.compile(
    r"(?<![\d.,])(\d+(?:[,.]\d+)?)\s*(%|proc\.?|procent(?:u|owa|owy|owej)?)", re.IGNORECASE
)
_SPREAD_RE = re.compile(
    r"(?:\+\s*|powiększon\w*\s+o\s+|powiekszon\w*\s+o\s+)"
    r"(\d+(?:[,.]\d+)?)\s*(bp|bps|pb|punkt(?:y|ów|ow)?\s+bazowych|%)",
    re.IGNORECASE,
)
_NO_BENCHMARK_RE = re.compile(
    r"\b(?:NIE\s+POSIADA\s+BENCHMARKU|BRAK\s+BENCHMARKU|NIE\s+DOTYCZY)\b", re.IGNORECASE
)
_RATE_RE = re.compile(r"\b(?:WIBOR|WIBID|WIRON|POLSTR|POLONIA|EURIBOR)", re.IGNORECASE)
_HEDGED_PLN_RE = re.compile(
    r"(?:zabezpieczon\w*\s+do\s+(?:(?:polskiego\s+)?zlot\w*|pln)|\bpln\s+hedged\b|\bhedged\s+to\s+pln\b)",
    re.IGNORECASE,
)


def _parse_spread(text: str) -> float:
    match = _SPREAD_RE.search(text)
    if not match:
        return 0.0
    amount = float(match.group(1).replace(",", "."))
    unit = match.group(2).lower()
    if unit == "%":
        return amount * 100.0
    return amount


def _is_spread_percent(text: str, match_start: int) -> bool:
    prefix = text[:match_start]
    plus_position = prefix.rfind("+")
    if plus_position < 0 or prefix[plus_position + 1 :].strip():
        return False
    rate_prefix = normalize_index_text(prefix[:plus_position])[-50:]
    return bool(
        re.search(
            r"\b(?:WIBOR|WIBID|WIRON|POLSTR|POLONIA)\s*(?:\d+\s*(?:M|Y)|O\s*/?\s*N)?$|"
            r"\bEURIBOR\s*\d+\s*M$",
            rate_prefix,
            re.IGNORECASE,
        ),
    )


def _is_terminal_portfolio_spread(text: str, matches: list[re.Match[str]], match: re.Match[str]) -> bool:
    if match is not matches[-1] or text[match.end() :].strip():
        return False
    if not text[: match.start()].rstrip().endswith("+"):
        return False
    prior_weight = sum(float(item.group(1).replace(",", ".")) / 100.0 for item in matches[:-1])
    return abs(prior_weight - 1.0) <= 0.01


def parse_benchmark_text(text: str | None) -> ParsedBenchmark:
    if text is None or not text.strip():
        return ParsedBenchmark((), ("empty benchmark text",), 0.0)
    if _NO_BENCHMARK_RE.search(text):
        return ParsedBenchmark((), (), 0.0, True)

    all_weight_matches = list(_WEIGHT_RE.finditer(text))
    weight_matches = [
        match
        for match in all_weight_matches
        if not _is_spread_percent(text, match.start())
        and not _is_terminal_portfolio_spread(text, all_weight_matches, match)
    ]
    if not weight_matches:
        index_key = resolve_index_key(text)
        if index_key is None:
            return ParsedBenchmark((), (text.strip(),), 0.0)
        component = BenchmarkComponent(
            index_key=index_key,
            weight=1.0,
            spread_bps=_parse_spread(text),
            hedged_to_pln=bool(
                _HEDGED_PLN_RE.search(text)
            ),
        )
        return ParsedBenchmark((component,), (), 1.0)

    components: list[BenchmarkComponent] = []
    unresolved: list[str] = []
    weight_sum = 0.0
    for index, match in enumerate(weight_matches):
        segment_end = weight_matches[index + 1].start() if index + 1 < len(weight_matches) else len(text)
        segment = text[match.end() : segment_end].strip(" \t,;:+*()")
        weight = float(match.group(1).replace(",", ".")) / 100.0
        weight_sum += weight
        index_key = resolve_index_key(segment)
        if index_key is None:
            unresolved.append(segment or text[: match.start()].strip())
            continue
        components.append(
            BenchmarkComponent(
                index_key=index_key,
                weight=weight,
                spread_bps=_parse_spread(segment),
                hedged_to_pln=bool(_HEDGED_PLN_RE.search(segment)),
            ),
        )

    portfolio_spread_bps = 0.0
    if components:
        final_segment = text[weight_matches[-1].end() :].strip(" \t,;:+*()")
        spread_match = _SPREAD_RE.search(final_segment)
        if spread_match and spread_match.end() == len(final_segment):
            before_spread = final_segment[: spread_match.start()]
            parenthesis_depth = before_spread.count("(") - before_spread.count(")")
            if parenthesis_depth <= 0 and not _RATE_RE.search(before_spread):
                portfolio_spread_bps = _parse_spread(final_segment)
                last = components[-1]
                components[-1] = BenchmarkComponent(
                    index_key=last.index_key,
                    weight=last.weight,
                    spread_bps=0.0,
                    hedged_to_pln=last.hedged_to_pln,
                )

    if abs(weight_sum - 1.0) > 0.01:
        unresolved.append(f"component weights sum to {weight_sum:.4f}, not 1.0")
    return ParsedBenchmark(
        tuple(components),
        tuple(unresolved),
        weight_sum,
        portfolio_spread_bps=portfolio_spread_bps,
    )
