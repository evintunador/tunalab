"""
PIQA (Physical Intuition Question Answering) dataset for multiple-choice evaluation.

Reference: https://huggingface.co/datasets/baber/piqa

Physical intuition task: given a goal (e.g. "How do I soften butter quickly?"),
pick which of two solution strings achieves the goal correctly.

2-way MC; random baseline is 50%.
GPT-2 (124M) acc ≈ 63%; GPT-2-XL (1.5B) acc ≈ 70%.

Note: the canonical ``piqa`` HF dataset requires a loading script that is no
longer supported by the datasets library. This adapter uses the ``baber/piqa``
mirror which is stored as plain Parquet files.
"""
import os
from enum import Enum
from typing import Optional

from torch.utils.data import Dataset

from tunalab.evaluations.multiple_choice import MultipleChoiceItem


class Split(Enum):
    TRAIN = "train"
    VAL   = "validation"
    TEST  = "test"


class PIQADataset(Dataset):
    """PIQA physical-intuition multiple-choice dataset.

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
            cache_dir = os.path.join("data", ".cache", "piqa")

        raw = load_dataset(
            "baber/piqa",
            split=split.value,
            cache_dir=cache_dir,
        )

        items = []
        for ex in raw:
            label = int(ex["label"])
            if label not in (0, 1):
                continue
            items.append(MultipleChoiceItem(
                context=ex["goal"],
                choices=[ex["sol1"], ex["sol2"]],
                label=label,
            ))

        if limit is not None:
            items = items[:limit]
        self.data = items

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> MultipleChoiceItem:
        return self.data[idx]
