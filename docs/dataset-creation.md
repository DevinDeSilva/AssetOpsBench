# DiagnosticIQ dataset creation

The `dataset_utils` package and the accompanying notebooks create the simple,
complex, perturbed, and test variants of the DiagnosticIQ dataset. The core
generation algorithms were transferred without behavioral changes.

## Repository layout

```text
src/dataset_utils/                       Installable dataset creation package
src/dataset_utils/utils/                 File, tree, and grouping helpers
src/dataset_utils/tests/                 Focused package tests
notebook/simple_dataset_creation.ipynb   Base multiple-choice dataset
notebook/complex_dataset_creation.ipynb  Expanded-option variant
notebook/simple_perteval_simple_dataset_creation.ipynb
                                         Perturbation-based variant
notebook/simple_test_dataset_creation.ipynb
                                         Small test variant and duplicate checks
```

## Setup

Install AssetOpsBench and the notebook dependencies from the repository root:

```bash
uv sync --group dataset-creation
```

The source artifacts and generated datasets are intentionally external to the
installed package. Set `ASSETOPSBENCH_DATASET_DIR` to their common parent:

```bash
export ASSETOPSBENCH_DATASET_DIR=/path/to/diagnosticiq-data
```

The expected workspace layout is:

```text
$ASSETOPSBENCH_DATASET_DIR/
├── extracted/
│   ├── TreeStruct.pkl
│   ├── observations.rules.json
│   └── equipment_info/
│       └── gre_desc.json
├── datasets/
│   └── ... generated dataset variants ...
└── test_datasets/
    ├── TEST_TreeStruct.pkl
    ├── TEST_rule_information.json
    └── datasets/
```

Start Jupyter from the repository root. The notebooks no longer depend on the
process working directory:

```bash
uv run --group dataset-creation jupyter lab notebook/
```

## Programmatic use

The package is installed with AssetOpsBench and can be imported from any
working directory:

```python
from dataset_utils import config, creation
from dataset_utils.paths import get_dataset_dir

dataset_dir = get_dataset_dir()
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
creation.save_dataset(dataset, "simpleV5", dataset_dir / "datasets")
```

## Existing pickle compatibility

Older `TreeStruct.pkl` files identify `Node` as either `utils.tree.Node` or
`dataset.utils.tree.Node`. `dataset_utils.utils.file_handle.load_pickle`
remaps those legacy module names to `dataset_utils.utils.tree.Node` while
loading. New pickle files use the installed package path directly.

As with any Python pickle, only load files from a trusted source.
