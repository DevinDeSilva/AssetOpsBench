import json

import pytest

from evaluation.scorers.static_json import (
    evaluate_static_json,
    evaluate_static_json_batch,
    flatten_answer,
    normalize_value,
    parse_structured_answer,
)


def test_parse_json_object_from_noisy_markdown_answer():
    raw = 'Answer:\n```json\n{"energy": 3, "material": 12}\n```'

    assert parse_structured_answer(raw) == {"energy": 3, "material": 12}


def test_parse_fenced_json_response_key_from_noisy_answer():
    raw = (
        "Based on my analysis, here are the findings.\n\n"
        "```json\n"
        '{"response": "PMP42144 has 4 recurring air conditioner issues."}\n'
        "```"
    )

    assert parse_structured_answer(raw) == {
        "response": "PMP42144 has 4 recurring air conditioner issues."
    }


def test_parse_json_after_parenthetical_prose():
    raw = (
        'The registry has generic descriptions (all descriptions are "Asset PMPxxxxx").\n\n'
        '{"clarification": "Which pump do you mean by the big pump out the back?"}'
    )

    assert parse_structured_answer(raw) == {
        "clarification": "Which pump do you mean by the big pump out the back?"
    }


def test_parse_json_after_parenthetical_with_number():
    raw = (
        'Based on the repair history (asset PMP42144 has 17 work orders), '
        'the answer is:\n'
        '{"clarification": "Which asset do you mean by the main unit?"}'
    )

    assert parse_structured_answer(raw) == {
        "clarification": "Which asset do you mean by the main unit?"
    }


def test_parse_python_style_dict():
    raw = "{'energy': 14, 'material': 48}"

    assert parse_structured_answer(raw) == {"energy": 14, "material": 48}


def test_parse_python_style_list_of_tuples():
    raw = '[("Engines & motors", 5), ("Lines & drives", 2)]'

    assert parse_structured_answer(raw) == [
        ("Engines & motors", 5),
        ("Lines & drives", 2),
    ]


def test_parse_count_only_answer():
    assert parse_structured_answer("34") == 34


def test_parse_noisy_count_answer():
    assert parse_structured_answer("The answer is 34.") == 34


def test_parse_noisy_count_answer_prefers_final_standalone_number():
    raw = (
        "I checked the work orders and found no normal-operation jobs "
        "(such as tramming or normal machine movement) that should be counted.\n\n"
        "0"
    )

    assert parse_structured_answer(raw) == 0


def test_count_answer_compares_final_number_not_parenthetical_text():
    score = evaluate_static_json(
        "0",
        (
            "I checked the work orders and found no normal-operation jobs "
            "(such as tramming or normal machine movement) that should be counted.\n\n"
            "0"
        ),
    )

    assert score.strict_exact_match_accuracy == 1.0
    assert score.f1 == 1.0
    assert score.details[0].model_value == "0"


def test_count_answer_with_wrong_final_number_fails_against_final_number():
    score = evaluate_static_json(
        "52",
        (
            'One work order mentions both "engine" and "motor"; this concerns '
            "an engine (the engine fan), so it counts as an engine job.\n\n"
            "55"
        ),
    )

    assert score.strict_exact_match_accuracy == 0.0
    assert score.details[0].model_value == "55"


def test_parse_fault_code_as_categorical_string():
    assert parse_structured_answer("FC101") == "FC101"


def test_parse_json_object_after_parenthetical_text():
    raw = 'The dataset is ambiguous (no asset is specified).\n\n{"clarification": "Which asset?"}'

    assert parse_structured_answer(raw) == {"clarification": "Which asset?"}


def test_choice_answer_accepts_final_standalone_letter_after_explanation():
    score = evaluate_static_json(
        "C",
        (
            "I compared the excavator repair costs and C has the highest total.\n\n"
            "C"
        ),
    )

    assert score.strict_exact_match_accuracy == 1.0
    assert score.partial_exact_match_accuracy == 1.0
    assert score.details[0].model_value == "c"


def test_choice_answer_accepts_final_answer_label():
    score = evaluate_static_json(
        "C",
        "The data points to excavator C.\n\nFinal Answer: C",
    )

    assert score.strict_exact_match_accuracy == 1.0
    assert score.f1 == 1.0


def test_choice_answer_accepts_trailing_option_phrase():
    score = evaluate_static_json(
        "C",
        "After reviewing the options, the correct option is C.",
    )

    assert score.strict_exact_match_accuracy == 1.0


def test_flatten_nested_json():
    answer = {"a": {"b": 2}, "c": [3, {"d": 4}]}

    assert flatten_answer(answer) == {
        "answer.a.b": "2",
        "answer.c[0]": "3",
        "answer.c[1].d": "4",
    }


def test_flatten_tuple_list_answer():
    answer = '[("Engines & motors", 5), ("Lines & drives", 2)]'

    assert flatten_answer(answer) == {
        "answer[0][0]": "engines & motors",
        "answer[0][1]": "5",
        "answer[1][0]": "lines & drives",
        "answer[1][1]": "2",
    }


def test_exact_match_json_object_with_prefix():
    score = evaluate_static_json(
        {"energy": 3, "material": 12},
        'Final Answer: {"energy": 3, "material": 12}',
    )

    assert score.strict_exact_match_accuracy == 1.0
    assert score.partial_exact_match_accuracy == 1.0
    assert score.partial_similarity_score == 1.0
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.f1 == 1.0
    assert score.missing_keys == []
    assert score.extra_keys == []


def test_missing_and_extra_keys_are_reported():
    score = evaluate_static_json(
        {"energy": 3, "material": 12},
        {"energy": 3, "other": 99},
    )

    assert score.strict_exact_match_accuracy == 0.0
    assert score.exact_value_matches == 1
    assert score.missing_keys == ["answer.material"]
    assert score.extra_keys == ["answer.other"]


def test_wrong_value_is_not_strict_match():
    score = evaluate_static_json(
        {"energy": 14, "material": 48},
        {"energy": 14, "material": 27},
    )

    assert score.strict_exact_match_accuracy == 0.0
    assert score.partial_exact_match_accuracy == 0.5
    assert score.exact_value_matches == 1
    assert score.missing_keys == []
    assert score.extra_keys == []


def test_numeric_partial_similarity():
    score = evaluate_static_json({"count": 100}, {"count": 104})

    assert score.strict_exact_match_accuracy == 0.0
    assert score.partial_similarity_score == 0.7


def test_anomaly_segment_scores_exact_categorical_and_numeric_delta():
    gold = {
        "condition": "faulty",
        "start_point": "240",
        "end_point": "511",
        "fault_type": "FC101",
    }
    model = {
        "condition": "faulty",
        "start_point": "241",
        "end_point": "512",
        "fault_type": "FC101",
    }

    score = evaluate_static_json(gold, model)
    details = {item.key: item for item in score.details}

    assert score.strict_exact_match_accuracy == 0.0
    assert score.partial_exact_match_accuracy == 0.5
    assert score.partial_match_accuracy == 1.0
    assert score.partial_numeric_match_accuracy == 1.0
    assert score.range_match_accuracy == 0.5
    assert score.delta_1_match_accuracy == 1.0
    assert details["answer.condition"].match_type == "exact"
    assert details["answer.fault_type"].match_type == "exact"
    assert details["answer.start_point"].match_type == "partial_delta_1"
    assert details["answer.start_point"].range_match is True
    assert details["answer.start_point"].delta_1_match is True
    assert details["answer.end_point"].match_type == "partial_delta_1"
    assert details["answer.end_point"].range_match is False
    assert details["answer.end_point"].delta_1_match is True


def test_anomaly_segment_scores_numeric_range_match():
    gold = {
        "condition": "faulty",
        "start_point": "240",
        "end_point": "511",
        "fault_type": "FC101",
    }
    model = {
        "condition": "faulty",
        "start_point": "300",
        "end_point": "500",
        "fault_type": "FC101",
    }

    score = evaluate_static_json(gold, model)
    details = {item.key: item for item in score.details}

    assert score.strict_exact_match_accuracy == 0.0
    assert score.partial_exact_match_accuracy == 0.5
    assert score.partial_match_accuracy == 1.0
    assert score.range_match_accuracy == 1.0
    assert score.delta_1_match_accuracy == 0.0
    assert details["answer.start_point"].match_type == "partial_range"
    assert details["answer.end_point"].match_type == "partial_range"


def test_count_only_delta_one_is_numeric_partial_match():
    score = evaluate_static_json("34", "35")

    assert score.strict_exact_match_accuracy == 0.0
    assert score.partial_exact_match_accuracy == 0.0
    assert score.partial_match_accuracy == 1.0
    assert score.partial_numeric_match_accuracy == 1.0
    assert score.delta_1_match_accuracy == 1.0
    assert score.details[0].match_type == "partial_delta_1"


def test_numeric_answer_can_match_explicit_ground_truth_range():
    score = evaluate_static_json({"count": "10-12"}, {"count": 11})

    assert score.strict_exact_match_accuracy == 0.0
    assert score.partial_match_accuracy == 1.0
    assert score.partial_numeric_match_accuracy == 1.0
    assert score.range_match_accuracy == 1.0
    assert score.details[0].match_type == "partial_range"


def test_mode_clarification_accepts_equivalent_question_with_required_phrase():
    score = evaluate_static_json(
        {"clarification": "Which asset do you mean by 'the main unit'?"},
        {"clarification": "Can you clarify which asset you mean by the main unit?"},
    )

    assert score.strict_exact_match_accuracy == 1.0
    assert score.mode_key_match == 1.0
    assert score.mode_exactly_one_key == 1.0
    assert score.mode_required_terms == ["main unit"]
    assert score.mode_matched_terms == ["main unit"]
    assert score.mode_term_coverage == 1.0


def test_mode_clarification_fails_when_mode_key_is_wrong():
    score = evaluate_static_json(
        {"clarification": "Which asset do you mean by 'the usual suspect'?"},
        {"response": "The usual suspect is pump PMP42144."},
    )

    assert score.strict_exact_match_accuracy == 0.0
    assert score.mode_key_match == 0.0
    assert score.missing_keys == ["answer.clarification"]
    assert score.extra_keys == ["answer.response"]


def test_mode_abstain_requires_lately_dataset_date_time_terms():
    score = evaluate_static_json(
        {
            "abstain": (
                "Cannot determine what has been giving trouble 'lately' because "
                "the dataset does not contain date or time information."
            )
        },
        {
            "abstain": (
                "Cannot determine the lately-troublesome asset because the "
                "dataset has no date or time information."
            )
        },
    )

    assert score.strict_exact_match_accuracy == 1.0
    assert score.mode_required_terms == [
        "lately",
        "cannot determine",
        "date",
        "dataset",
        "time",
    ]
    assert score.mode_term_coverage == 1.0


def test_mode_response_checks_asset_id_and_domain_terms():
    score = evaluate_static_json(
        {
            "response": (
                "No - the 'PMP' tag is unreliable. Tag PMP42144's work orders "
                "describe an air conditioner, steering/stick cylinders, a "
                "cracked handrail, tyres, a pressure vessel and an exhaust leak, "
                "none of which is pump work, so the tag does not indicate a pump."
            )
        },
        {
            "response": (
                "No. PMP42144 should not be treated as a pump; the tag is "
                "unreliable because the history points to air conditioner, "
                "steering, handrail, pressure vessel, and exhaust leak work."
            )
        },
    )

    assert score.strict_exact_match_accuracy == 1.0
    assert "pmp42144" in score.mode_required_terms
    assert "pump" in score.mode_required_terms
    assert "unreliable" in score.mode_required_terms
    assert score.mode_term_coverage == 1.0


def test_mode_requires_exactly_one_top_level_key():
    score = evaluate_static_json(
        {"clarification": "Which asset do you mean by 'the main unit'?"},
        {
            "clarification": "Which asset do you mean by the main unit?",
            "response": "I can also answer once you clarify.",
        },
    )

    assert score.strict_exact_match_accuracy == 0.0
    assert score.mode_key_match == 0.0
    assert score.mode_exactly_one_key == 0.0
    assert score.extra_keys == ["answer.clarification", "answer.response"]


def test_mode_fails_when_a_required_term_is_missing():
    score = evaluate_static_json(
        {"clarification": "Which asset do you mean by 'the main unit'?"},
        {"clarification": "Which asset should I analyze?"},
    )

    assert score.strict_exact_match_accuracy == 0.0
    assert score.mode_key_match == 1.0
    assert score.mode_matched_terms == []
    assert score.mode_term_coverage == 0.0
    assert score.f1 == 0.5


def test_mode_needs_every_required_term_to_pass():
    score = evaluate_static_json(
        {
            "abstain": (
                "Cannot determine what has been giving trouble 'lately' because "
                "the dataset does not contain date or time information."
            )
        },
        {"abstain": "Cannot determine this because the dataset has no date information."},
    )

    assert score.strict_exact_match_accuracy == 0.0
    assert score.mode_key_match == 1.0
    assert score.mode_matched_terms == ["cannot determine", "date", "dataset"]
    assert score.mode_term_coverage == 0.6
    assert score.f1 == pytest.approx(2 / 3)


def test_mode_wrong_key_still_earns_partial_f1_for_terms():
    score = evaluate_static_json(
        {"clarification": "Which asset do you mean by 'the main unit'?"},
        {"response": "The main unit appears to be PMP42144."},
    )

    assert score.strict_exact_match_accuracy == 0.0
    assert score.mode_key_match == 0.0
    assert score.mode_term_coverage == 1.0
    assert score.f1 == pytest.approx(0.4)


def test_mode_key_is_case_insensitive():
    score = evaluate_static_json(
        {"abstain": "Not enough information."},
        '{"Abstain": "Not enough information."}',
    )

    assert score.strict_exact_match_accuracy == 1.0
    assert score.mode_key_match == 1.0


def test_mode_fails_when_answer_is_not_an_object():
    score = evaluate_static_json(
        {"abstain": "Cannot determine because date or time is missing."},
        "I cannot answer this question.",
    )

    assert score.strict_exact_match_accuracy == 0.0
    assert score.f1 == 0.0
    assert score.mode_exactly_one_key == 0.0
    assert score.extra_keys == ["answer"]


def test_count_only_exact_match():
    score = evaluate_static_json("34", "The answer is 34.")

    assert score.strict_exact_match_accuracy == 1.0
    assert score.f1 == 1.0


def test_batch_evaluation():
    result = evaluate_static_json_batch(
        [
            ({"energy": 3}, {"energy": 3}),
            ({"material": 10}, {"material": 9}),
        ]
    )

    assert result["num_examples"] == 2
    assert result["strict_exact_match_accuracy"] == 0.5



from evaluation.models import Scenario
from evaluation.scorers.static_json import StaticJsonScorer


def test_static_json_scorer_wrapper_exact_match():
    scenario = Scenario.from_raw(
        {
            "id": "11",
            "text": "Count storage jobs.",
            "expected_answer": "{'energy': 14, 'material': 48}",
            "scoring_method": "static_json",
        }
    )

    scorer = StaticJsonScorer()
    result = scorer(
        scenario,
        '{"energy": 14, "material": 48}',
        "",
    )

    assert result.scorer == "static_json"
    assert result.passed is True
    assert result.score == 1.0
    assert result.details["strict_exact_match_accuracy"] == 1.0


def test_static_json_scorer_scores_car_on_mode_key_and_required_terms():
    scenario = Scenario.from_raw(
        {
            "id": "151",
            "text": "Clarify main unit.",
            "expected_answer": '{"clarification": "Which asset do you mean by the main unit?"}',
            "scoring_method": "static_json",
        }
    )

    scorer = StaticJsonScorer()
    full = scorer(scenario, '{"clarification": "Which asset is the main unit?"}', "")
    term_missing = scorer(scenario, '{"clarification": "Which pump?"}', "")
    wrong_key = scorer(scenario, '{"response": "The main unit is PMP42144."}', "")

    assert full.passed is True
    assert full.score == 1.0
    assert term_missing.passed is False
    assert term_missing.score == 0.5
    assert wrong_key.passed is False
    assert wrong_key.score == 0.4


_CAR_GOLD = {"clarification": "Which asset and which time window do you mean?"}
_CAR_METADATA = {
    "mode": "clarification",
    "required_terms": [
        {"concept": "asset", "any_of": ["asset", "equipment"]},
        {"concept": "time window", "any_of": ["time window", "date range"]},
    ],
}


def test_car_concepts_give_zero_recall_on_a_wrong_key():
    score = evaluate_static_json(
        _CAR_GOLD,
        {"abstain": "Which asset and which time window?"},
        eval_metadata=_CAR_METADATA,
    )

    assert score.mode_key_match == 0.0
    assert score.mode_term_coverage == 1.0
    assert score.recall == 0.0


def test_car_concepts_give_zero_recall_to_a_multi_key_answer():
    score = evaluate_static_json(
        _CAR_GOLD,
        {"clarification": "Which asset, which time window?", "abstain": "n/a"},
        eval_metadata=_CAR_METADATA,
    )

    assert score.mode_exactly_one_key == 0.0
    assert score.recall == 0.0


def test_car_concepts_recall_is_the_covered_fraction():
    half = evaluate_static_json(
        _CAR_GOLD,
        {"clarification": "Which equipment do you mean?"},
        eval_metadata=_CAR_METADATA,
    )
    none = evaluate_static_json(
        _CAR_GOLD,
        {"clarification": "Can you say more?"},
        eval_metadata=_CAR_METADATA,
    )

    assert half.mode_required_terms == ["asset", "time window"]
    assert half.mode_matched_terms == ["asset"]
    assert half.recall == 0.5
    assert half.precision is None
    assert half.f1 is None
    assert none.recall == 0.0


def test_car_concepts_match_any_form_on_word_boundaries():
    score = evaluate_static_json(
        _CAR_GOLD,
        {"clarification": "Which equipment, and over what date range?"},
        eval_metadata=_CAR_METADATA,
    )
    partial_word = evaluate_static_json(
        _CAR_GOLD,
        {"clarification": "Which assetid?"},
        eval_metadata=_CAR_METADATA,
    )

    assert score.recall == 1.0
    assert score.details[1].model_value == "equipment"
    assert score.details[2].model_value == "date range"
    assert partial_word.recall == 0.0


@pytest.mark.parametrize(
    ("form", "text"),
    [
        ("asset", "Which assets?"),
        ("time window", "Which time windows?"),
        ("switch", "Which switches?"),
        ("battery", "Which batteries?"),
        ("relay", "Which relays?"),
    ],
)
def test_car_concepts_match_the_plural_of_a_form(form, text):
    score = evaluate_static_json(
        _CAR_GOLD,
        {"clarification": text},
        eval_metadata={"required_terms": [{"concept": form, "any_of": [form]}]},
    )

    assert score.recall == 1.0
    assert score.details[1].model_value == form


def test_car_extracted_terms_do_not_match_plurals():
    score = evaluate_static_json(
        {"clarification": "Which asset do you mean by 'the main unit'?"},
        {"clarification": "Which of the main units?"},
    )

    assert score.mode_matched_terms == []


def test_car_concepts_read_a_structured_value_case_insensitively():
    score = evaluate_static_json(
        _CAR_GOLD,
        {"clarification": {"question": "Which Asset, over which Time Window?"}},
        eval_metadata=_CAR_METADATA,
    )

    assert score.recall == 1.0


def test_car_concepts_read_a_structured_value_across_newlines():
    score = evaluate_static_json(
        _CAR_GOLD,
        {"clarification": {"questions": ["Which asset?\ntime window?"]}},
        eval_metadata=_CAR_METADATA,
    )

    assert score.recall == 1.0


def test_car_concepts_accept_bare_string_terms():
    score = evaluate_static_json(
        _CAR_GOLD,
        {"clarification": "Which asset?"},
        eval_metadata={"required_terms": ["asset", "time window"]},
    )

    assert score.mode_required_terms == ["asset", "time window"]
    assert score.recall == 0.5


def test_car_without_concepts_keeps_extracted_term_scoring():
    score = evaluate_static_json(
        {"clarification": "Which asset do you mean by 'the main unit'?"},
        {"clarification": "Which pump?"},
        eval_metadata={"optional_terms": []},
    )

    assert score.mode_required_terms == ["main unit"]
    assert score.precision == 0.5
    assert score.f1 == 0.5


_CAR_GOLD_4 = {
    "clarification": "Which asset, which site, which sensor and what time window?"
}
_CAR_METADATA_4 = {
    "mode": "clarification",
    "required_terms": [
        {"concept": "asset", "any_of": ["asset", "equipment"]},
        {"concept": "site", "any_of": ["site", "plant"]},
        {"concept": "sensor", "any_of": ["sensor", "tag"]},
        {"concept": "time window", "any_of": ["time window", "date range"]},
    ],
}


def _car_scenario(gold, metadata, scenario_id="1"):
    return Scenario.from_raw(
        {
            "id": scenario_id,
            "text": "Synthetic CAR scenario.",
            "expected_answer": json.dumps(gold),
            "evaluation_metadata": metadata,
            "scoring_method": "static_json",
        }
    )


def test_static_json_scorer_fails_car_at_half_recall():
    """Half coverage no longer passes: the threshold is 0.75."""
    scenario = _car_scenario(_CAR_GOLD, _CAR_METADATA)

    scorer = StaticJsonScorer()
    half = scorer(scenario, '{"clarification": "Which asset?"}', "")
    full = scorer(scenario, '{"clarification": "Which equipment, what date range?"}', "")
    none = scorer(scenario, '{"clarification": "Can you say more?"}', "")
    wrong_key = scorer(scenario, '{"abstain": "Which asset and time window?"}', "")

    assert half.passed is False
    assert half.score == 0.5
    assert half.details["recall"] == 0.5
    assert half.details["f1"] is None
    assert half.rationale == "CAR recall 0.500 < 0.75"
    assert full.passed is True
    assert full.score == 1.0
    assert none.passed is False
    assert none.score == 0.0
    assert wrong_key.passed is False
    assert wrong_key.score == 0.0


def test_static_json_scorer_passes_car_at_three_quarter_recall():
    """Four concepts make the 0.75 boundary expressible: 3 of 4 passes, 2 of 4 does not."""
    scenario = _car_scenario(_CAR_GOLD_4, _CAR_METADATA_4, scenario_id="2")

    scorer = StaticJsonScorer()
    three = scorer(
        scenario, '{"clarification": "Which asset, which site, which sensor?"}', ""
    )
    two = scorer(scenario, '{"clarification": "Which asset and which site?"}', "")

    assert three.passed is True
    assert three.details["recall"] == 0.75
    assert three.rationale == "CAR recall 0.750 >= 0.75"
    assert two.passed is False
    assert two.details["recall"] == 0.5
    assert two.rationale == "CAR recall 0.500 < 0.75"


def test_identifier_with_a_separator_is_not_read_as_a_count():
    """An asset id is a label, not a number.

    ``FAN-01`` used to reach ``_extract_count_from_text``, whose lookbehind
    excluded letters and digits but not the hyphen, so the whole id parsed as
    the integer 1. Every ``XXX-01`` asset then normalised to the same value and
    any list of them matched any other.
    """
    assert normalize_value("FAN-01") == "fan-01"
    assert normalize_value("MOT-01") == "mot-01"
    assert normalize_value("RC-1") == "rc-1"
    assert normalize_value("GEN-01") == "gen-01"


def test_distinct_asset_lists_do_not_match():
    gold = '["FAN-01", "MOT-01", "PMP-01"]'
    model = '["CMP-01", "FAN-01", "PMP-01"]'

    score = evaluate_static_json(gold, model)

    assert score.strict_exact_match_accuracy == 0.0
    assert score.exact_value_matches == 1


def test_count_rescue_still_reads_a_number_out_of_prose():
    """The fix must not break the path it guards."""
    assert parse_structured_answer("The answer is 10.") == 10
    assert parse_structured_answer("10") == 10
    assert normalize_value("PMP42144") == "pmp42144"


_RATES_GOLD = [
    {"asset": "D", "failures_per_year": 125.2},
    {"asset": "B", "failures_per_year": 85.0},
    {"asset": "E", "failures_per_year": 75.6},
    {"asset": "C", "failures_per_year": 54.1},
    {"asset": "A", "failures_per_year": 13.2},
]


def _rates_scenario(meta):
    return Scenario.from_raw(
        {
            "id": "410",
            "text": "Breakdown rate per machine, highest to lowest.",
            "expected_answer": json.dumps(_RATES_GOLD),
            "evaluation_metadata": {"scenario_meta": meta},
            "scoring_method": "static_json",
        }
    )


def _rates(*values, assets="DBECA"):
    return json.dumps(
        [
            {"asset": asset, "failures_per_year": value}
            for asset, value in zip(assets, values)
        ]
    )


_TEN_PERCENT = {
    "scoring_method": "static_json",
    "match": "numeric",
    "relative_tolerance": 0.1,
}


def test_relative_tolerance_passes_rates_within_ten_percent():
    result = StaticJsonScorer()(
        _rates_scenario(_TEN_PERCENT), _rates(122.4, 83.8, 73.5, 53.0, 13.0), ""
    )

    assert result.passed is True
    assert result.score == 1.0
    assert result.details["relative_tolerance"] == 0.1
    assert result.details["tolerance_operator"] == "<="
    assert result.details["strict_exact_match_accuracy"] == 0.0
    assert result.rationale.startswith("10 of 10 values accepted")


@pytest.mark.parametrize(
    ("b_rate", "passed"), [(93.5, True), (76.5, True), (93.6, False)]
)
def test_relative_tolerance_boundary_is_inclusive(b_rate, passed):
    result = StaticJsonScorer()(
        _rates_scenario(_TEN_PERCENT), _rates(125.2, b_rate, 75.6, 54.1, 13.2), ""
    )

    assert result.passed is passed


def test_relative_tolerance_fails_counting_every_work_order():
    result = StaticJsonScorer()(
        _rates_scenario(_TEN_PERCENT), _rates(285.9, 243.1, 164.4, 104.9, 30.1), ""
    )

    assert result.passed is False
    assert result.score == 0.5
    rejected = [
        v["key"] for v in result.details["tolerance_values"] if not v["accepted"]
    ]
    assert len(rejected) == 5
    assert all(key.endswith(".failures_per_year") for key in rejected)


def test_relative_tolerance_keeps_list_order_and_asset_ids_exact():
    swapped = StaticJsonScorer()(
        _rates_scenario(_TEN_PERCENT),
        _rates(125.2, 75.6, 85.0, 54.1, 13.2, assets="DEBCA"),
        "",
    )
    renamed = StaticJsonScorer()(
        _rates_scenario(_TEN_PERCENT),
        _rates(125.2, 85.0, 75.6, 54.1, 13.2, assets="DBECX"),
        "",
    )

    assert swapped.passed is False
    assert renamed.passed is False


def test_relative_tolerance_fails_on_an_extra_key():
    answer = json.loads(_rates(125.2, 85.0, 75.6, 54.1, 13.2))
    answer[0]["note"] = "worst unit"

    result = StaticJsonScorer()(_rates_scenario(_TEN_PERCENT), json.dumps(answer), "")

    assert result.passed is False
    assert result.details["extra_keys"] == ["answer[0].note"]


def test_relative_tolerance_reads_a_fenced_answer():
    answer = "```json\n" + _rates(122.4, 83.8, 73.5, 53.0, 13.0) + "\n```"

    result = StaticJsonScorer()(_rates_scenario(_TEN_PERCENT), answer, "")

    assert result.passed is True


@pytest.mark.parametrize(
    "meta",
    [
        {"match": "numeric"},
        {"match": "numeric", "relative_tolerance": -0.1},
        {"match": "numeric", "relative_tolerance": True},
        {"match": "numeric", "relative_tolerance": "0.1"},
        {"match": "set_f1", "relative_tolerance": 0.1},
    ],
)
def test_relative_tolerance_rejects_invalid_scenario_meta(meta):
    result = StaticJsonScorer()(
        _rates_scenario({"scoring_method": "static_json", **meta}),
        json.dumps(_RATES_GOLD),
        "",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.details["error"] == "invalid_scenario_meta"


def test_scenario_meta_without_match_keeps_strict_matching():
    scenario = _rates_scenario({"scoring_method": "static_json"})

    exact = StaticJsonScorer()(scenario, json.dumps(_RATES_GOLD), "")
    near = StaticJsonScorer()(scenario, _rates(125.2, 85.0, 75.6, 54.1, 13.0), "")

    assert exact.passed is True
    assert near.passed is False
    assert near.rationale == "structured answer differs from ground truth"
