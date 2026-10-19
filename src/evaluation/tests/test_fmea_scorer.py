import json

import pytest

from evaluation.models import Scenario
from evaluation.scorers.fmea import fmea_score


def _scenario(gold, *, tag="asset_modes", weights=None):
    dimensions = weights or [
        {"name": "coverage", "weight": 0.35},
        {"name": "precision", "weight": 0.35},
        {"name": "instruction_following", "weight": 0.2},
        {"name": "grounding", "weight": 0.1},
    ]
    return Scenario(
        id="9001",
        text="",
        expected_answer=json.dumps(gold),
        evaluation_metadata={
            "scenario_meta": {"tag": tag},
            "rubric": {"dimensions": dimensions},
        },
    )


def test_asset_modes_awards_set_precision_and_recall():
    scenario = _scenario({"asset": "Electric Motor", "failure_modes": {"A": "", "B": ""}})
    answer = json.dumps({"asset": "electric motor", "failure_modes": {"A": "", "C": ""}})

    result = fmea_score(scenario, answer, "")

    assert result.passed is False
    assert result.details["dimensions"]["coverage"] == 0.5
    assert result.details["dimensions"]["precision"] == 0.5
    assert result.details["asset_match"] is True
    assert result.details["missing_modes"] == ["B"]
    assert result.details["extra_modes"] == ["C"]
    assert result.score == 0.5


def test_asset_modes_perfect_answer_passes_case_insensitive_asset():
    gold = {"asset": "Electric Motor", "failure_modes": {"A": ""}}
    result = fmea_score(
        _scenario(gold), '{"asset":"electric motor","failure_modes":{"A":""}}', ""
    )
    assert result.passed is True
    assert result.score == 1.0


def test_markdown_fence_gets_content_credit_but_fails_contract():
    gold = {"asset": "Electric Motor", "failure_modes": {"A": ""}}
    result = fmea_score(_scenario(gold), f"```json\n{json.dumps(gold)}\n```", "")
    assert result.passed is True
    assert result.details["semantic_exact"] is True
    assert result.details["strict_success"] is False
    assert result.details["json_only"] is False
    assert result.details["dimensions"]["instruction_following"] == 0.0


def test_analytics_scores_atomic_facts_and_ordered_arrays():
    scenario = _scenario(
        {
            "ranking": ["Seal", "Bearing", "Motor"],
            "counts": {"Seal": 16, "Bearing": 13, "Motor": 7},
        },
        tag="ranking_top3",
    )
    answer = json.dumps(
        {
            "ranking": ["Bearing", "Seal", "Motor"],
            "counts": {"Seal": 16, "Bearing": 13, "Motor": 7},
        }
    )
    result = fmea_score(scenario, answer, "")
    assert result.passed is False
    assert result.details["dimensions"]["coverage"] == pytest.approx(4 / 6)
    assert result.details["wrong_value_keys"] == ["answer.ranking[0]", "answer.ranking[1]"]


def test_duplicate_nested_mode_key_fails_json_contract():
    scenario = _scenario({"asset": "Volute", "failure_modes": {"Corrosion": ""}})
    answer = '{"asset":"Volute","failure_modes":{"Corrosion":"","Corrosion":""}}'
    result = fmea_score(scenario, answer, "")
    assert result.passed is True
    assert result.details["semantic_exact"] is True
    assert result.details["strict_success"] is False
    assert result.details["duplicate_keys"] == ["Corrosion"]


def test_trajectory_reports_catalog_and_out_of_scope_tool_use():
    scenario = _scenario({"asset": "Electric Motor", "failure_modes": {"A": ""}})
    trajectory = json.dumps(
        {"turns": [{"tool_calls": [
            {"name": "utilities__get_asset_catalog"},
            {"name": "utilities__get_failure_mode_catalog"},
            {"name": "fmsr__get_failure_modes"},
        ]}]}
    )
    result = fmea_score(scenario, scenario.expected_answer, trajectory)
    assert result.details["missing_catalog_tools"] == []
    assert result.details["out_of_scope_domain_tools"] == ["fmsr__get_failure_modes"]
    assert result.details["catalog_process_compliant"] is False


def test_pass_requires_at_least_point_eight_f1():
    scenario = _scenario(
        {"asset": "Electric Motor", "failure_modes": {str(i): "" for i in range(7)}}
    )
    below = {
        "asset": "Electric Motor",
        "failure_modes": {
            **{str(i): "" for i in range(4)},
            **{f"extra-{i}": "" for i in range(3)},
        },
    }
    result = fmea_score(scenario, json.dumps(below), "")
    assert result.score == pytest.approx(4 / 7, abs=1e-6)
    assert result.passed is False
    assert result.details["f1_pass_threshold"] == 0.8


def test_f1_exactly_at_point_eight_passes_despite_float_error():
    scenario = _scenario(
        {"asset": "Electric Motor", "failure_modes": {str(i): "" for i in range(7)}}
    )
    at_threshold = {
        "asset": "Electric Motor",
        "failure_modes": {
            **{str(i): "" for i in range(6)},
            **{f"extra-{i}": "" for i in range(2)},
        },
    }
    result = fmea_score(scenario, json.dumps(at_threshold), "")
    assert result.details["semantic_f1"] < 0.8
    assert result.score == 0.8
    assert result.passed is True


def test_f1_above_point_eight_passes_without_requiring_exact_match():
    scenario = _scenario(
        {"asset": "Electric Motor", "failure_modes": {str(i): "" for i in range(8)}}
    )
    partial = {
        "asset": "Electric Motor",
        "failure_modes": {str(i): "" for i in range(6)},
    }
    result = fmea_score(scenario, json.dumps(partial), "")
    assert result.score == pytest.approx(6 / 7, abs=1e-6)
    assert result.passed is True
    assert result.details["strict_success"] is False


def _six_of_eight(threshold):
    """An answer with F1 6/7 (0.857) against a scenario carrying ``threshold``."""
    scenario = _scenario(
        {"asset": "Electric Motor", "failure_modes": {str(i): "" for i in range(8)}}
    )
    if threshold != "unset":
        scenario.evaluation_metadata["scenario_meta"]["f1_pass_threshold"] = threshold
    answer = {"asset": "Electric Motor", "failure_modes": {str(i): "" for i in range(6)}}
    return fmea_score(scenario, json.dumps(answer), "")


def test_scenario_threshold_overrides_the_default():
    result = _six_of_eight(0.9)
    assert result.details["f1_pass_threshold"] == 0.9
    assert result.passed is False


@pytest.mark.parametrize("threshold", ["unset", None])
def test_unset_threshold_uses_the_default(threshold):
    result = _six_of_eight(threshold)
    assert result.details["f1_pass_threshold"] == 0.8
    assert result.passed is True


@pytest.mark.parametrize("threshold", [True, -0.1, 1.5, 80, "0.8", float("nan")])
def test_invalid_threshold_fails_the_scenario_with_a_rationale(threshold):
    result = _six_of_eight(threshold)
    assert result.passed is False
    assert result.score == 0.0
    assert result.details["error"] == "invalid_scenario_meta"
    assert "f1_pass_threshold" in result.rationale
