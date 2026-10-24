from dataset_utils.config import DatasetCreationConfig
from dataset_utils.creation import create_dataset, load_dataset, save_dataset
from dataset_utils.reader import ADIQDataset
from utils.tree import Node


def _rule(
    number: int, rule_id: str, asset: str, conditions: str, observations: list[str]
):
    return {
        "#n": number,
        "id": rule_id,
        "asset": asset,
        "display_text": {
            "date": "",
            "observations": observations,
            "rules": Node.generate_graph(conditions),
        },
    }


def test_create_save_and_load_dataset(tmp_path):
    rule_information = {
        "rule_set": [
            _rule(
                1,
                "RULE_1",
                "pump",
                "temperature is high OR pressure is low",
                ["bearing wear", "seal leak"],
            ),
            _rule(
                2,
                "RULE_2",
                "fan",
                "vibration is high OR airflow is low",
                ["blade imbalance", "filter blockage"],
            ),
        ]
    }
    observations = {
        observation: observation
        for observation in (
            "bearing wear",
            "seal leak",
            "blade imbalance",
            "filter blockage",
        )
    }
    similarities = {
        "RULE_1": ["RULE_2", "RULE_1"],
        "RULE_2": ["RULE_1", "RULE_2"],
    }
    config = DatasetCreationConfig(
        {
            "num_options": 2,
            "sel_random_sample": 1,
            "elem_random_sample": 1,
            "n_least_sim_rules": 2,
        }
    )

    dataset = create_dataset(
        rule_information,
        observations,
        {"pump": "A test pump", "fan": "A test fan"},
        similarities,
        config,
    )

    assert dataset["questions"]
    assert dataset["creation_config"]["num_options"] == 2
    assert all(
        len(question["options"])
        == len(question["option_ids"])
        == len(question["correct"])
        for question in dataset["questions"]
    )

    save_dataset(dataset, "smoke", str(tmp_path))
    loaded = load_dataset(str(tmp_path / "smoke"))
    reader = ADIQDataset(str(tmp_path / "smoke"))

    assert loaded["creation_config"] == dataset["creation_config"]
    assert len(reader.get_questions()) == len(dataset["questions"])
