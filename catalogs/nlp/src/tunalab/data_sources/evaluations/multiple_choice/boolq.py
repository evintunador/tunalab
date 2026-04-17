"""
BoolQ dataset for multiple-choice evaluation.

Reference: https://huggingface.co/datasets/google/boolq

Boolean yes/no questions paired with a Wikipedia passage. Each item presents
the passage + question as context; the two choices are "Yes" and "No".

2-way MC; random baseline is 50%.
GPT-2 (124M) acc ≈ 62%; competitive models reach 88%+.
"""
import os
from enum import Enum
from typing import Optional

from torch.utils.data import Dataset

from tunalab.evaluations.multiple_choice import MultipleChoiceItem


class Split(Enum):
    TRAIN = "train"
    VAL   = "validation"


class BoolQDataset(Dataset):
    """BoolQ boolean yes/no question answering dataset.

    Context is ``"{passage}\\n\\nQuestion: {question}"``.
    Choices are always ``["Yes", "No"]`` (label 0 = Yes, label 1 = No).

    Args:
        split: Dataset split (no test labels in BoolQ).
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
            cache_dir = os.path.join("data", ".cache", "boolq")

        raw = load_dataset(
            "google/boolq",
            split=split.value,
            cache_dir=cache_dir,
        )

        items = []
        for ex in raw:
            answer = ex["answer"]   # bool
            label  = 0 if answer else 1  # 0=Yes, 1=No
            context = f"{ex['passage']}\n\nQuestion: {ex['question']}"
            items.append(MultipleChoiceItem(
                context=context,
                choices=["Yes", "No"],
                label=label,
            ))

        if limit is not None:
            items = items[:limit]
        self.data = items

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> MultipleChoiceItem:
        return self.data[idx]
