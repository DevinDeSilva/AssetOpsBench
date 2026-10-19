import json

import pytest

from evaluation import scorers as registry
from evaluation.models import Scenario
from evaluation.scorers.fcc import fcc_score


def _scenario(gold, meta):
    return Scenario(
        id="9101",
        text="",
        expected_answer=gold if isinstance(gold, str) else json.dumps(gold),
        evaluation_metadata={"scenario_meta": {"scoring_method": "fcc", **meta}},
    )


def _numeric(gold, tolerance):
    return _scenario(gold, {"match": "numeric", "numeric_tolerance": tolerance})


def _set_f1(gold, threshold):
    return _scenario(gold, {"match": "set_f1", "f1_pass_threshold": threshold})


def _pairs(codes):
    return [
        {"wo_id": f"WO-{index}", "failure_code": code}
        for index, code in enumerate(codes, start=1)
    ]


def test_registered_as_fcc():
    assert registry.get("fcc") is fcc_score


@pytest.mark.parametrize(("answer", "passed"), [("11", True), ("13", True), ("10", False)])
def test_count_passes_within_tolerance_inclusive(answer, passed):
    result = fcc_score(_numeric("12", 1), answer, "")
    assert result.passed is passed
    assert result.score == (1.0 if passed else 0.0)
    assert result.details["gold"] == 12
    assert result.details["model"] == float(answer)
    assert result.details["deltas"] == [abs(float(answer) - 12)]
    assert result.details["numeric_tolerance"] == 1
    assert result.details["tolerance_operator"] == "<="


def test_prose_with_final_bare_number_passes():
    answer = "After filling the blank records I counted them.\n12"
    result = fcc_score(_numeric("12", 1), answer, "")
    assert result.passed is True
    assert result.details["model"] == 12
    assert result.details["model_source"] == "answer"


def test_bare_first_line_wins_over_numbers_in_a_flag_line():
    # Parsing the whole answer would pick up the parenthesised "(3)".
    answer = "12\nFlag: A (3) + B (9) = 12 is above C (10)"
    result = fcc_score(_numeric("12", 1), answer, "")
    assert result.passed is True
    assert result.details["model"] == 12
    assert result.details["model_source"] == "first_line"


def test_first_line_fallback_when_the_whole_answer_is_ambiguous():
    answer = "12 records\nThat is above the 10 and 11 seen last time"
    result = fcc_score(_numeric("12", 1), answer, "")
    assert result.passed is True
    assert result.details["model"] == 12
    assert result.details["model_source"] == "first_line"


def test_boolean_is_never_a_number():
    result = fcc_score(_numeric("1", 1), "true", "")
    assert result.passed is False
    assert result.details["model"] is None


@pytest.mark.parametrize(
    ("answer", "passed"),
    [
        ("43.2%", True),
        ("43.3%", True),  # 43.3 - 40.0 is 3.299999999999997 in floats
        ("36.7%", True),
        ("44.0%", False),
        ("40.0%\nFlag: the descriptions are too thin to trust", True),
    ],
)
def test_percentage_within_tolerance(answer, passed):
    result = fcc_score(_numeric("40.0%", 3.3), answer, "")
    assert result.passed is passed
    assert result.details["gold"] == 40.0


@pytest.mark.parametrize(
    ("answer", "passed"),
    [
        ("[52.0, 48.0]", True),
        ("[51.7, 48.3]", True),
        ('["55.0%", "45.0%"]', True),
        ("[51.0, 49.0]", False),
        ("[55.0]", False),
        ("[55.0, 45.0, 0.0]", False),
        ("55.0", False),
    ],
)
def test_number_list_compares_element_wise(answer, passed):
    result = fcc_score(_numeric([55.0, 45.0], 3.3), answer, "")
    assert result.passed is passed


def test_number_list_reports_each_delta():
    result = fcc_score(_numeric([55.0, 45.0], 3.3), "[51.0, 49.0]", "")
    assert result.details["deltas"] == [4.0, 4.0]
    assert result.details["model"] == [51.0, 49.0]


def test_set_f1_exactly_at_threshold_passes():
    result = fcc_score(_set_f1(["CODE_A", "CODE_B", "CODE_C"], 0.8), '["CODE_A", "CODE_B"]', "")
    assert result.score == 0.8
    assert result.passed is True
    assert result.details["precision"] == 1.0
    assert result.details["recall"] == pytest.approx(2 / 3)
    assert result.details["f1_pass_threshold"] == 0.8
    assert result.details["f1_threshold_operator"] == ">="
    assert result.details["missing_items"] == ["CODE_C"]
    assert result.details["extra_items"] == []


def test_set_f1_below_threshold_fails():
    result = fcc_score(_set_f1(["CODE_A", "CODE_B", "CODE_C"], 0.8), '["CODE_A"]', "")
    assert result.score == 0.5
    assert result.passed is False


def test_set_f1_ignores_order_case_and_duplicates():
    scenario = _set_f1(["CODE_A", "CODE_B", "CODE_C"], 0.8)
    result = fcc_score(scenario, '["code_c", "CODE_B", "Code_A", "CODE_B"]', "")
    assert result.score == 1.0
    assert result.details["model_count"] == 3


def test_set_f1_non_list_answer_fails():
    result = fcc_score(_set_f1(["CODE_A", "CODE_B"], 0.8), "CODE_A, CODE_B", "")
    assert result.passed is False
    assert result.score == 0.0
    assert result.details["answer_is_list"] is False


def test_empty_gold_and_empty_answer_is_a_perfect_match():
    result = fcc_score(_set_f1([], 0.8), "[]", "")
    assert result.passed is True
    assert result.score == 1.0


def test_object_pairs_eight_of_ten_passes_at_point_eight():
    gold = _pairs(["CODE_A", "CODE_B", "CODE_C"] * 3 + ["CODE_A"])
    model = [dict(item) for item in gold]
    model[0]["failure_code"] = "CODE_C"
    model[1]["failure_code"] = "CODE_C"
    model[2] = {"WO_ID": " wo-3 ", "failure_code": "code_c"}
    result = fcc_score(_set_f1(gold, 0.8), json.dumps(model), "")
    assert result.details["matched_count"] == 8
    assert result.score == 0.8
    assert result.passed is True


def test_object_with_right_wo_id_and_wrong_code_is_a_miss_and_an_extra():
    gold = _pairs(["CODE_A", "CODE_B"])
    model = _pairs(["CODE_A", "CODE_C"])
    result = fcc_score(_set_f1(gold, 0.8), json.dumps(model), "")
    assert result.details["missing_items"] == [gold[1]]
    assert result.details["extra_items"] == [model[1]]
    assert result.score == 0.5
    assert result.passed is False


@pytest.mark.parametrize(
    ("gold", "meta"),
    [
        ("12", {}),
        ("12", {"match": "fuzzy"}),
        ("12", {"match": "numeric"}),
        ("12", {"match": "numeric", "numeric_tolerance": -1}),
        ("12", {"match": "numeric", "numeric_tolerance": True}),
        ("12", {"match": "numeric", "numeric_tolerance": "1"}),
        ("not a number", {"match": "numeric", "numeric_tolerance": 1}),
        ("[]", {"match": "numeric", "numeric_tolerance": 1}),
        ('["CODE_A"]', {"match": "set_f1"}),
        ('["CODE_A"]', {"match": "set_f1", "f1_pass_threshold": 1.5}),
        ('["CODE_A"]', {"match": "set_f1", "f1_pass_threshold": "0.8"}),
        ("CODE_A", {"match": "set_f1", "f1_pass_threshold": 0.8}),
    ],
)
def test_invalid_meta_or_gold_fails_without_raising(gold, meta):
    result = fcc_score(_scenario(gold, meta), "12", "")
    assert result.passed is False
    assert result.score == 0.0
    assert result.details["error"] in {"invalid_scenario_meta", "invalid_ground_truth"}
    assert result.rationale


def test_missing_scenario_meta_fails_without_raising():
    scenario = Scenario(id="9101", text="", expected_answer="12")
    result = fcc_score(scenario, "12", "")
    assert result.passed is False
    assert result.details["error"] == "invalid_scenario_meta"


@pytest.mark.parametrize("answer", ["{[1]: 2}", '[{"wo_id": "{[1]: 2}"}]', ""])
def test_unparseable_answer_fails_without_raising(answer):
    assert fcc_score(_numeric("12", 1), answer, "").passed is False
    assert fcc_score(_set_f1(["CODE_A"], 0.8), answer, "").passed is False
