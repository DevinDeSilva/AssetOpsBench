import os
import random
from typing import Any, Dict, List

from .creation import load_dataset
from .question import Question
from .utils.file_handle import save_json, save_jsonl


class ADIQDataset:
    questions: List[Question]
    unique_observations: Dict[str, str]
    asset_descriptions: Dict[str, Any]
    rule_info: Dict[str, Any]

    def __init__(self, loc: str) -> None:
        self.loc = loc
        self.__build_dataset__()

    def __build_dataset__(self) -> None:
        self.data = load_dataset(self.loc)
        for k, v in self.data.items():
            setattr(self, k, v)

        self.questions = [Question.from_dict(x) for x in self.data["questions"]]
        random.shuffle(self.questions)

    def get_questions(self) -> List[Question]:
        return self.questions

    def save(self, dataset_name: str, loc: str) -> None:
        dir = os.path.join(loc, dataset_name)

        if os.path.exists(dir):
            print("Overridding Dataset")

        os.makedirs(dir, exist_ok=True)

        save_jsonl(
            [x.to_dict() for x in self.questions], os.path.join(dir, "data.jsonl")
        )

        save_json(
            {k: v for k, v in self.__dict__.items() if k not in ["questions", "data"]},
            os.path.join(dir, "metadata.json"),
        )
