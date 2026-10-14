# Static JSON Evaluation

`static_json` is a deterministic scorer for structured AssetOpsBench answers. It is intended for scenarios whose expected answers are structured outputs, including JSON objects, JSON arrays, Python-style dictionaries, tuple lists, nested JSON, and count-only answers.

This scorer is adapted from the DeepSynth-style static evaluation design: it parses structured outputs, normalizes values, flattens nested structures, compares key-value pairs, computes exact and partial metrics, and reports missing/extra keys. In AssetOpsBench, it is integrated into the existing trajectory-based evaluation pipeline.

---

## Evaluation Workflow

`static_json` follows the same high-level workflow as the LLM-as-Judge evaluator:

```text
agent run
→ saved trajectory
→ extract final answer and scenario_id
→ load ground truth
→ run scorer
→ aggregate benchmark-level report
```

The scorer is invoked through the existing `evaluate` command:

```bash
uv run evaluate \
  --trajectories traces/trajectories \
  --scenarios /path/to/scenarios_data \
  --scorer-default static_json \
  --reports-dir reports/static_json_eval
```

---

## Expected Inputs

### Trajectories

The evaluator reads saved trajectory JSON files from the path passed to `--trajectories`.

Example trajectory:

```json
{
  "run_id": "11",
  "scenario_id": "11",
  "runner": "direct-llm-agent",
  "model": "litellm_proxy/Azure/gpt-5-mini-2025-08-07",
  "question": "Check the storage-related maintenance job records...",
  "answer": "{\"energy\":14,\"material\":27}",
  "trajectory": {
    "turns": []
  }
}
```

The model's final answer is read from the trajectory's `answer` field.

If `scenario_id` is missing or null, the evaluator can fall back to the trajectory filename stem. For example:

```text
traces/trajectories/34.json → scenario_id = "34"
```

For some generated trajectories, `scenario_id` may contain a descriptive label while `run_id` contains the numeric scenario id. In that case, the loader can use `run_id` as a fallback join key.

---

## Ground Truth Layout

The evaluator can load ground truth from scenario folders.

Expected folder layout:

```text
scenarios_data/
  scenario_11/
    groundtruth.txt
    groundtruth_eval.json    # optional
  scenario_12/
    groundtruth.txt
  scenario_13/
    groundtruth.txt
```

Example `groundtruth.txt`:

```python
{'energy': 14, 'material': 48}
```

The folder name is converted into the scenario id:

```text
scenario_11 → 11
```

The evaluator then joins trajectories and ground truth by scenario id.

### Optional evaluation metadata

Scenario folders may also include `groundtruth_eval.json`, which carries
scorer-specific metadata alongside the canonical answer in `groundtruth.txt`.
The loader exposes it as `Scenario.evaluation_metadata`. `static_json` reads
it only for clarification-abstain-response (CAR) scenarios, whose expected
mode is the single key of `groundtruth.txt`.

Example CAR `groundtruth.txt`:

```json
{
  "clarification": "Which asset and which time window do you mean?"
}
```

Example CAR `groundtruth_eval.json`:

```json
{
  "mode": "clarification",
  "required_terms": [
    {"concept": "asset", "any_of": ["asset", "equipment"]},
    {"concept": "time window", "any_of": ["time window", "date range"]}
  ],
  "optional_terms": []
}
```

Each required concept lists the surface forms (`any_of`) that count as
covering it. A bare string in `required_terms` is a concept with that one
form. `mode` and `optional_terms` are not scored; the mode comes from
`groundtruth.txt`.

---

## How Matching Works

The evaluator loads all trajectories and all ground-truth scenario folders, then joins them.

Example:

```text
trajectory: traces/trajectories/direct_llm/11.json
run_id: 11
scenario_id: 11
answer: {"energy":14,"material":27}

ground truth: scenarios_data/scenario_11/groundtruth.txt
expected answer: {'energy': 14, 'material': 48}
```

The evaluator matches the trajectory to the ground truth using the available join keys, then compares:

```text
model answer vs. ground-truth answer
```

Only matched trajectory/scenario pairs are evaluated. If a trajectory has no matching ground-truth scenario, it is skipped. If a ground-truth scenario has no matching trajectory, it is skipped.

---

## What `static_json` Handles

The scorer can parse common structured model outputs, including:

### JSON object

```json
{"energy": 14, "material": 48}
```

### JSON array

```json
[
  {"equipment_group": "Pumps/fans/generators", "count": 21},
  {"equipment_group": "Lines & drives", "count": 15}
]
```

### Python-style dictionary

```python
{'energy': 14, 'material': 48}
```

### Python-style list of tuples

```python
[("Engines & motors", 5), ("Lines & drives", 2)]
```

### Markdown fenced answer

````markdown
```json
{"repair": 13, "replace": 0}
```
````

### Answer-prefixed output

```text
Final Answer: {"repair": 13, "replace": 0}
```

### Count-only answer

```text
34
```

The scorer also handles simple noisy count-only answers such as:

```text
The answer is 34.
```

For noisy structured answers, the parser prefers a final JSON/Python structure
over earlier explanatory parenthetical text. This avoids evaluating prose such
as `(no asset is specified)` when the model later returns the actual final JSON
object.

---

## Metrics

`static_json` reports both aggregate metrics and key-level details.

| Metric                         | Meaning                                                                               |
| ------------------------------ | ------------------------------------------------------------------------------------- |
| `strict_exact_match_accuracy`  | 1.0 only if the full flattened gold answer exactly matches the flattened model answer |
| `partial_exact_match_accuracy` | Fraction of gold keys whose values exactly match                                      |
| `partial_similarity_score`     | Average similarity score across gold keys, including numeric closeness                |
| `precision`                    | Exact matches divided by total model keys                                             |
| `recall`                       | Exact matches divided by total gold keys                                              |
| `f1`                           | Harmonic mean of precision and recall                                                 |
| `missing_keys`                 | Keys present in gold but missing from model output                                    |
| `extra_keys`                   | Keys present in model output but not in gold                                          |
| `details`                      | Per-key comparison records                                                            |

The scorer marks a scenario as passed only when the structured answer is a strict exact match, and reports `f1` as `score`. Partial correctness is still available through `score`, `f1`, `partial_exact_match_accuracy`, and detailed key-level results. CAR scenarios with `groundtruth_eval.json` are the exception: see below.

### CAR metrics and pass criteria

For CAR scenarios, the scorer evaluates the answer as a mode-selection task plus
required-term coverage. The model answer should be a single JSON object using
exactly one of these top-level keys:

```text
response | clarification | abstain
```

Matching is case-insensitive on normalised text and on word boundaries, so
`asset` does not match inside `assetid`.

#### With `groundtruth_eval.json`

When the scenario's `groundtruth_eval.json` has `required_terms`, the score is
`recall`, in [0, 1]:

1. If the answer is not a JSON object with exactly one top-level key, or that
   key is not the expected mode, recall is 0.
2. Otherwise recall is the fraction of required concepts covered. A
   concept is covered when any one of its `any_of` forms, or that form's
   regular plural (`asset` → `assets`, `switch` → `switches`, `battery` →
   `batteries`), appears in the value under the key. Irregular forms such as
   `criteria` / `criterion` still need listing.

The scenario passes when recall is at least 0.5 (`CAR_PASS_THRESHOLD`).
`precision` and `f1` are `null`: precision needs a count of what the model
asserted, and a free-text value gives none, so it could only repeat recall.
For the example above (two concepts):

| Model answer                                                  | Passed | Score (recall) |
| ------------------------------------------------------------- | ------ | -------------- |
| `{"clarification": "Which equipment, over what date range?"}` | yes    | 1.0            |
| `{"clarification": "Which assets?"}`                          | yes    | 0.5            |
| `{"clarification": "Can you say more?"}`                      | no     | 0.0            |
| `{"abstain": "Which asset and time window?"}`                 | no     | 0.0            |

#### Without `groundtruth_eval.json`

The required terms are extracted from the gold value: quoted phrases, asset or
fault identifiers (e.g. `PMP42144`), a leading `yes` / `no` stance, and a small
allowlist of domain terms. The scenario passes when the answer has exactly one
top-level key, that key is the expected mode, and every required term appears
in the value under it. The score is the F1 over the mode key plus each required
term. Gold has `1 + len(required_terms)` entries; the model side has the same
plus any extra top-level keys. For gold `{"clarification": "Which asset do you
mean by 'the main unit'?"}` (one required term, `main unit`):

| Model answer                                         | Passed | Score |
| ---------------------------------------------------- | ------ | ----- |
| `{"clarification": "Which asset is the main unit?"}` | yes    | 1.0   |
| `{"clarification": "Which pump?"}`                   | no     | 0.5   |
| `{"response": "The main unit is PMP42144."}`         | no     | 0.4   |

Additional CAR report fields:

| Metric                 | Meaning                                                              |
| ---------------------- | -------------------------------------------------------------------- |
| `mode_key_match`       | 1.0 when the returned mode key matches the expected mode             |
| `mode_exactly_one_key` | 1.0 when the answer has exactly one top-level key                    |
| `mode_required_terms`  | Required concepts from `groundtruth_eval.json`, or extracted terms   |
| `mode_matched_terms`   | Required concepts or terms found in the model answer value           |
| `mode_term_coverage`   | Fraction of required concepts or terms found, regardless of the key  |

---

## Example

Ground truth:

```python
{'energy': 14, 'material': 48}
```

Model answer:

```json
{"energy":14,"material":27}
```

The scorer flattens both answers:

```text
gold:
answer.energy = 14
answer.material = 48

model:
answer.energy = 14
answer.material = 27
```

Then it compares each key:

```text
answer.energy: expected 14, got 14 → exact
answer.material: expected 48, got 27 → mismatch
```

Example score:

```json
{
  "passed": false,
  "score": 0.5,
  "details": {
    "partial_exact_match_accuracy": 0.5,
    "strict_exact_match_accuracy": 0.0,
    "partial_similarity_score": 0.5,
    "precision": 0.5,
    "recall": 0.5,
    "f1": 0.5,
    "total_gold_keys": 2,
    "total_model_keys": 2,
    "matched_keys": 2,
    "exact_value_matches": 1,
    "missing_keys": [],
    "extra_keys": []
  }
}
```

---

## Reports

The evaluator writes per-run reports and an aggregate report to the directory passed with `--reports-dir`.

Example:

```bash
uv run evaluate \
  --trajectories traces/trajectories/direct_llm \
  --scenarios /path/to/scenarios_data \
  --scorer-default static_json \
  --reports-dir reports/static_json_direct_llm
```

Output:

```text
reports/static_json_direct_llm/
  11.json
  12.json
  ...
  _aggregate.json
```

Each per-run report contains the scenario id, run id, model answer, score, key-level comparison details, and operational metrics.

The aggregate report summarizes the number of scored scenarios, pass rate, runner/model names, and operational metrics.

---

## Recommended Usage

Use `llm_judge` for:

* natural-language reasoning answers
* explanations
* open-ended diagnostic responses
* answers with many acceptable phrasings

Use `static_json` for:

* JSON objects
* JSON arrays
* Python-style dictionaries
* tuple lists
* nested structured outputs
* count-only answers
* benchmark scenarios where key/value correctness matters

---

## Running Tests

Run the static JSON scorer tests:

```bash
uv run pytest src/evaluation/tests/test_static_json_scorer.py -v
```

Run loader tests for trajectory/scenario matching:

```bash
uv run pytest src/evaluation/tests/test_loader.py -v
```

Run evaluation tests:

```bash
uv run pytest src/evaluation/tests -v
```

---

## Notes for Developers

The core scorer lives in:

```text
src/evaluation/scorers/static_json.py
```

It is registered as:

```text
static_json
```

and can be selected through:

```bash
--scorer-default static_json
```

The scorer itself only needs:

```text
gold_answer
model_answer
eval_metadata   # groundtruth_eval.json, CAR only
```

The evaluation pipeline is responsible for:

```text
trajectory file → final answer + scenario_id/run_id
scenario folder → groundtruth.txt (+ groundtruth_eval.json)
join trajectory and scenario
run scorer
write report
```

---

## Acknowledgment
The static JSON scorer is inspired by DeepSynth-style structured-answer scoring. See [Acknowledgments](acknowledgments.md) for details.
