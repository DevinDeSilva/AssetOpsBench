# Dataset creation utilities

This directory contains the DiagnosticIQ dataset creation utilities and the
notebooks used to create its simple, complex, perturbed, and test variants.
The implementation and notebooks were transferred without logic changes.

## Contents

- `dataset_utils/`: question generation, configuration, serialization, and
  dataset reading utilities.
- `utils/`: file, tree, and grouping helpers used by `dataset_utils`.
- `simple_dataset_creation.ipynb`: creates the base multiple-choice dataset.
- `complex_dataset_creation.ipynb`: expands the base questions with additional
  options.
- `simple_perteval_simple_dataset_creation.ipynb`: creates perturbed variants
  for robustness evaluation.
- `simple_test_dataset_creation.ipynb`: creates a small test variant and checks
  for duplicate questions.

## Setup

Install the notebook dependencies from the repository root:

```bash
uv sync --group dataset-creation
```

The notebooks retain their original relative imports and paths. Start Jupyter
with this directory as the working directory:

```bash
cd dataset
uv run --group dataset-creation jupyter lab
```

For direct use from this directory:

```python
from dataset_utils import config, creation

dataset_config = config.DatasetCreationConfig(
    {
        "num_options": 4,
        "question_template_mode": config.QuestionTemplateMode.random,
        "question_template_random_count": 1,
    }
)

dataset = creation.create_dataset(
    rule_information,
    observation_mapping,
    asset_descriptions,
    rule_id_similarity_map,
    dataset_config,
)
```

## Input data

Generated datasets and source artifacts are not duplicated here. Supply the
inputs referenced by a notebook before running it. In particular, the base
creation notebook expects:

- `extracted/TreeStruct.pkl`
- `extracted/observations.rules.json`
- `extracted/equipment_info/gre_desc.json`

The complex and perturbation notebooks consume an existing generated dataset,
and the test notebook consumes test rule artifacts. Their paths and expected
variant names are visible in the first setup cells of each notebook.
