"""
ARC (AI2 Reasoning Challenge) dataset for multiple-choice evaluation.

Reference: https://huggingface.co/datasets/allenai/ai2_arc

Two difficulty splits are available:
  ARC-Easy      — straightforward science questions (4-way MC)
  ARC-Challenge — harder questions requiring deeper reasoning (4-way MC)

Each item has a question stem and four answer choices labelled A/B/C/D (or
1/2/3/4 for a small subset). The label field maps to the index of the
correct answer in the choices list.
"""
import os
from enum import Enum
from typing import Optional

from torch.utils.data import Dataset

from tunalab.evaluations.multiple_choice import MultipleChoiceItem


class Config(Enum):
    EASY      = "ARC-Easy"
    CHALLENGE = "ARC-Challenge"


class Split(Enum):
    TRAIN = "train"
    VAL   = "validation"
    TEST  = "test"


class ARCDataset(Dataset):
    """ARC multiple-choice dataset (Easy or Challenge).

    Args:
        config: ARC sub-dataset — ``Config.EASY`` or ``Config.CHALLENGE``.
        split: Dataset split.
        cache_dir: Directory to cache downloaded data.
        limit: Maximum number of examples to load.
    """

    def __init__(
        self,
        config: Config = Config.CHALLENGE,
        split: Split = Split.TEST,
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        from datasets import load_dataset

        if cache_dir is None:
            cache_dir = os.path.join("data", ".cache", "arc")

        raw = load_dataset(
            "allenai/ai2_arc",
            config.value,
            split=split.value,
            cache_dir=cache_dir,
        )

        items = []
        for ex in raw:
            choices_dict = ex["choices"]   # {"text": [...], "label": [...]}
            choice_texts  = choices_dict["text"]
            choice_labels = choices_dict["label"]
            answer_key    = ex["answerKey"]  # e.g. "A", "B", "C", "D" or "1"–"4"

            try:
                label_idx = choice_labels.index(answer_key)
            except ValueError:
                continue

            items.append(MultipleChoiceItem(
                context=ex["question"],
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
