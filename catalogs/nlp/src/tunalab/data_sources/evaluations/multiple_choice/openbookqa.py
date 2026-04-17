"""
OpenBookQA dataset for multiple-choice evaluation.

Reference: https://huggingface.co/datasets/allenai/openbookqa

Elementary science multiple-choice questions designed to test comprehension of
core science facts plus broader common knowledge. 4-way MC (labelled A–D).

Random baseline is 25%. Complementary to ARC; tests similar domain but with
different question style and difficulty profile.
GPT-2 (124M) acc ≈ 31%; competitive models reach 80%+.
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


class OpenBookQADataset(Dataset):
    """OpenBookQA elementary-science multiple-choice dataset.

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
            cache_dir = os.path.join("data", ".cache", "openbookqa")

        raw = load_dataset(
            "allenai/openbookqa",
            "main",
            split=split.value,
            cache_dir=cache_dir,
        )

        items = []
        for ex in raw:
            choices_dict  = ex["choices"]   # {"text": [...], "label": ["A","B","C","D"]}
            choice_texts  = choices_dict["text"]
            choice_labels = choices_dict["label"]
            answer_key    = ex["answerKey"]

            try:
                label_idx = choice_labels.index(answer_key)
            except ValueError:
                continue

            items.append(MultipleChoiceItem(
                context=ex["question_stem"],
                choices=choice_texts,
                label=label_idx,
            ))

        if limit is not None:
            items = items[:limit]
        self.data = items

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> MultipleChoiceItem:
        return self.data[idx]
