"""
MMLU (Massive Multitask Language Understanding) dataset for multiple-choice evaluation.

Reference: https://huggingface.co/datasets/cais/mmlu

57 subjects spanning STEM, humanities, social sciences, and professional domains.
This adapter covers the STEM-relevant subjects; any valid MMLU subject name is accepted.

Each item is a 4-way multiple-choice question; the model picks the choice with minimum
NLL. Random baseline is 25%. GPT-2 (124M) sits near chance on most subjects.
"""
import os
from enum import Enum
from typing import Optional

from torch.utils.data import Dataset

from tunalab.evaluations.multiple_choice import MultipleChoiceItem


class Split(Enum):
    DEV   = "dev"
    VAL   = "validation"
    TEST  = "test"


class MMLUDataset(Dataset):
    """MMLU multiple-choice dataset for a single subject.

    Args:
        subject: MMLU subject name (e.g. ``'college_mathematics'``,
            ``'high_school_physics'``, ``'machine_learning'``).
            Any valid cais/mmlu subset name is accepted.
        split: Dataset split (default: test).
        cache_dir: Directory to cache downloaded data.
        limit: Maximum number of examples to load.
    """

    def __init__(
        self,
        subject: str = "college_mathematics",
        split: Split = Split.TEST,
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        from datasets import load_dataset

        if cache_dir is None:
            cache_dir = os.path.join("data", ".cache", "mmlu")

        raw = load_dataset(
            "cais/mmlu",
            subject,
            split=split.value,
            cache_dir=cache_dir,
        )

        items = []
        for ex in raw:
            items.append(MultipleChoiceItem(
                context=ex["question"],
                choices=ex["choices"],
                label=int(ex["answer"]),
            ))

        if limit is not None:
            items = items[:limit]
        self.data = items

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> MultipleChoiceItem:
        return self.data[idx]
