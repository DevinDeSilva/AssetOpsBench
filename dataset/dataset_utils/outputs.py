"""
Any conversion of a question object to any output format is considered here.
"""
from typing import Dict

from .question import QuestionDifficulty, Question

def to_basic_prompt(q:Question, template:str, asset_descriptions:Dict[str,str]) -> str:
        return template.format(
            asset_type = q.asset_type,
            asset_description = asset_descriptions.get(q.asset_type, "NONE"),
            conditions = "\n".join(list(map(lambda x:"- "+x, q.condition_description))),
            temporal_condition = q.temporal_condition[0] if len(q.temporal_condition)>0 else "NONE",
            question_prompt = q.question_prompt,
            options = "\n".join(["{}). {}".format(op_id, op) for op_id, op in zip(q.option_ids,q.options)])
        )