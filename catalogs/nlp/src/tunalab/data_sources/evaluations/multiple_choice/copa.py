"""
COPA (Choice of Plausible Alternatives) dataset for multiple-choice evaluation.

Reference: https://huggingface.co/datasets/super_glue (config: copa)

2-way causal reasoning: given a premise and a question type ("cause" or "effect"),
pick which of two alternatives is the more plausible cause or effect.

Context is formatted as:
  "{premise} What was the {cause/effect} of this?"

Random baseline is 50%. Small but extremely clean causal signal.
GPT-2 (124M) acc ≈ 58%; competitive models reach 95%+.
"""
import os
from enum import Enum
from typing import Optional

from torch.utils.data import Dataset

from tunalab.evaluations.multiple_choice import MultipleChoiceItem

_QUESTION_SUFFIX = {
    "cause":  "What was the cause of this?",
    "effect": "What happened as a result?",
}


class Split(Enum):
    TRAIN = "train"
    VAL   = "validation"
    TEST  = "test"


class COPADataset(Dataset):
    """COPA causal plausibility multiple-choice dataset (from SuperGLUE).

    Args:
        split: Dataset split.
        cache_dir: Directory to cache downloaded data.
        limit: Maximum number of examples to load.
    """

    def __init__(
        self,
        split: Split = Split.VAL,
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        from datasets import load_dataset

        if cache_dir is None:
            cache_dir = os.path.join("data", ".cache", "copa")

        raw = load_dataset(
            "super_glue",
            "copa",
            split=split.value,
            cache_dir=cache_dir,
        )

        items = []
        for ex in raw:
            question_type = ex["question"]   # "cause" or "effect"
            suffix        = _QUESTION_SUFFIX.get(question_type, "")
            context       = f"{ex['premise']} {suffix}"
            items.append(MultipleChoiceItem(
                context=context,
                choices=[ex["choice1"], ex["choice2"]],
                label=int(ex["label"]),
            ))

        if limit is not None:
            items = items[:limit]
        self.data = items

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> MultipleChoiceItem:
        return self.data[idx]
