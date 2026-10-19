"""Deterministic scoring for failure-code classification (FCC) scenarios.

A scenario opts in through ``scenario_meta.json`` with one of two shapes::

    {"scoring_method": "fcc", "match": "numeric", "numeric_tolerance": 1}
    {"scoring_method": "fcc", "match": "set_f1", "f1_pass_threshold": 0.8}

``numeric`` passes when the answer is within ``numeric_tolerance`` of the
gold number: a count, a percentage such as ``40.0%``, or a list of numbers
compared element-wise.  ``set_f1`` scores a list of failure codes, or of
``{"wo_id": ..., "failure_code": ...}`` objects, by F1 over distinct items
and passes at ``f1_pass_threshold``.  Both comparisons are inclusive and are
made on values rounded to 6 decimals, so float error such as
``43.3 - 40.0 = 3.299999999999997`` cannot move an answer across a boundary.

Bad metadata or ground truth fails the scenario with a rationale instead of
raising, so one malformed file cannot abort an evaluation run.
"""

from __future__ import annotations

import math
import re
from typing import Any

from ..models import Scenario, ScorerResult
from . import register
from .static_json import normalize_value, parse_structured_answer

_NAME = "fcc"
_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
# A first line that is only the value: a number, a percentage, or a list.
_BARE_VALUE_RE = re.compile(r"\[.*\]|[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*%?")
_PARSE_ERRORS = (TypeError, ValueError, RecursionError)


def _metadata(scenario: Scenario, key: str) -> dict[str, Any]:
    metadata = scenario.evaluation_metadata or {}
    value = metadata.get(key, {})
    return value if isinstance(value, dict) else {}


def _parse(value: Any) -> Any:
    # ast.literal_eval raises TypeError on literals such as "{[1]: 2}".
    try:
        return parse_structured_answer(value)
    except _PARSE_ERRORS:
        return value


def _normalize(value: Any) -> str:
    try:
        return normalize_value(value)
    except _PARSE_ERRORS:
        return str(value).strip().lower()


def _number(value: Any) -> float | None:
    """Read one finite number; a string may carry a trailing ``%``."""
    if isinstance(value, str):
        value = value.strip().removesuffix("%").strip()
        if not _NUMBER_RE.fullmatch(value):
            return None
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _numbers(value: Any, want_list: bool) -> float | list[float] | None:
    if not want_list:
        return _number(value)
    if not isinstance(value, (list, tuple)):
        return None
    numbers = [_number(item) for item in value]
    return None if any(number is None for number in numbers) else numbers


def _setting(meta: dict[str, Any], key: str) -> float | None:
    value = meta.get(key)
    return _number(value) if isinstance(value, (int, float)) else None


def _read_model_numbers(
    answer: str, want_list: bool
) -> tuple[float | list[float] | None, str]:
    """Return the answer's number(s) and where they were read from."""
    lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]
    first_line = lines[0] if lines else ""
    # The questions put the value alone on the first line, and some allow a
    # free-text line after it.  Parsing the whole answer would read "(3)"
    # out of "12\nFlag: A (3) + B (9) = 12 is above C (10)".
    if _BARE_VALUE_RE.fullmatch(first_line):
        value = _numbers(_parse(first_line), want_list)
        if value is not None:
            return value, "first_line"
    value = _numbers(_parse(answer), want_list)
    if value is not None:
        return value, "answer"
    value = _numbers(_parse(first_line), want_list)
    if value is not None:
        return value, "first_line"
    return None, "unreadable"


def _format(value: float | list[float] | None) -> str:
    if value is None:
        return "none"
    if isinstance(value, list):
        return "[" + ", ".join(f"{item:g}" for item in value) + "]"
    return f"{value:g}"


def _failed(rationale: str, **details: Any) -> ScorerResult:
    return ScorerResult(
        scorer=_NAME, passed=False, score=0.0, rationale=rationale, details=details
    )


def _score_numeric(
    scenario: Scenario, answer: str, meta: dict[str, Any]
) -> ScorerResult:
    tolerance = _setting(meta, "numeric_tolerance")
    if tolerance is None or tolerance < 0:
        return _failed(
            "scenario_meta.json needs a non-negative numeric_tolerance for "
            f"match 'numeric'; got {meta.get('numeric_tolerance')!r}.",
            error="invalid_scenario_meta",
        )

    parsed_gold = _parse(scenario.expected_answer)
    want_list = isinstance(parsed_gold, (list, tuple))
    gold = _numbers(parsed_gold, want_list)
    if gold is None or gold == []:
        return _failed(
            "FCC ground truth is not a number, percentage, or non-empty list of numbers.",
            error="invalid_ground_truth",
        )

    model, source = _read_model_numbers(answer, want_list)
    details: dict[str, Any] = {
        "match": "numeric",
        "gold": gold,
        "model": model,
        "model_source": source,
        "deltas": [],
        "numeric_tolerance": tolerance,
        "tolerance_operator": "<=",
    }
    gold_values = gold if isinstance(gold, list) else [gold]
    model_values = model if isinstance(model, list) else [model]
    if model is None:
        expected = "a list of numbers" if want_list else "a number"
        return _failed(f"Could not read {expected} from the answer.", **details)
    if len(model_values) != len(gold_values):
        return _failed(
            f"Answer has {len(model_values)} values, gold has {len(gold_values)}; "
            f"answer={_format(model)}, gold={_format(gold)}.",
            **details,
        )

    deltas = [round(abs(m - g), 6) for m, g in zip(model_values, gold_values)]
    passed = all(delta <= tolerance for delta in deltas)
    details["deltas"] = deltas
    rationale = (
        f"Max |answer - gold| = {max(deltas):g} "
        f"({'<=' if passed else '>'} tolerance {tolerance:g}); "
        f"answer={_format(model)}, gold={_format(gold)}."
    )
    return ScorerResult(
        scorer=_NAME,
        passed=passed,
        score=1.0 if passed else 0.0,
        rationale=rationale,
        details=details,
    )


def _canonical(item: Any) -> Any:
    # An object matches only as a whole, so a right wo_id with a wrong
    # failure_code is one missing item plus one extra item.
    if isinstance(item, dict):
        return tuple(
            sorted((str(key).strip().lower(), _normalize(value)) for key, value in item.items())
        )
    return _normalize(item)


def _distinct(items: list[Any] | tuple[Any, ...]) -> dict[Any, Any]:
    """Map each canonical item to its first original form, in answer order."""
    out: dict[Any, Any] = {}
    for item in items:
        out.setdefault(_canonical(item), item)
    return out


def _score_set_f1(
    scenario: Scenario, answer: str, meta: dict[str, Any]
) -> ScorerResult:
    threshold = _setting(meta, "f1_pass_threshold")
    if threshold is None or not 0.0 <= threshold <= 1.0:
        return _failed(
            "scenario_meta.json needs an f1_pass_threshold in [0, 1] for "
            f"match 'set_f1'; got {meta.get('f1_pass_threshold')!r}.",
            error="invalid_scenario_meta",
        )

    gold = _parse(scenario.expected_answer)
    if not isinstance(gold, (list, tuple)):
        return _failed("FCC ground truth is not a list.", error="invalid_ground_truth")

    gold_items = _distinct(gold)
    model = _parse(answer)
    is_list = isinstance(model, (list, tuple))
    model_items = _distinct(model) if is_list else {}
    matched = gold_items.keys() & model_items.keys()
    if not is_list:
        precision = recall = f1 = 0.0
    else:
        precision = len(matched) / len(model_items) if model_items else float(not gold_items)
        recall = len(matched) / len(gold_items) if gold_items else 1.0
        total = len(gold_items) + len(model_items)
        f1 = 2.0 * len(matched) / total if total else 1.0
    missing = [item for key, item in gold_items.items() if key not in model_items]
    extra = [item for key, item in model_items.items() if key not in gold_items]

    # Compare at the reported precision, as the fmea scorer does.
    score = round(f1, 6)
    passed = score >= threshold
    details: dict[str, Any] = {
        "match": "set_f1",
        "answer_is_list": is_list,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_pass_threshold": threshold,
        "f1_threshold_operator": ">=",
        "gold_count": len(gold_items),
        "model_count": len(model_items),
        "matched_count": len(matched),
        "missing_items": missing,
        "extra_items": extra,
    }
    if not is_list:
        rationale = f"Answer is not a list; F1=0.000 (< {threshold:g})."
    else:
        rationale = (
            f"Set F1={f1:.3f} ({'>=' if passed else '<'} {threshold:g}); "
            f"precision={precision:.3f}, recall={recall:.3f}, "
            f"{len(missing)} missing, {len(extra)} extra."
        )
    return ScorerResult(
        scorer=_NAME, passed=passed, score=score, rationale=rationale, details=details
    )


_MATCHERS = {"numeric": _score_numeric, "set_f1": _score_set_f1}


def fcc_score(scenario: Scenario, answer: str, trajectory_text: str) -> ScorerResult:
    """Score an FCC answer by numeric tolerance or set F1, per scenario_meta.json."""
    meta = _metadata(scenario, "scenario_meta")
    match = meta.get("match")
    matcher = _MATCHERS.get(match) if isinstance(match, str) else None
    if matcher is None:
        return _failed(
            f"scenario_meta.json match must be one of {sorted(_MATCHERS)}; "
            f"got {match!r}.",
            error="invalid_scenario_meta",
        )
    return matcher(scenario, answer, meta)


def install(name: str = _NAME) -> None:
    """Register the deterministic FCC scorer."""
    register(name, fcc_score)
