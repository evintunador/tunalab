"""
WinoGrande dataset for multiple-choice evaluation.

Reference: https://huggingface.co/datasets/winogrande

Commonsense pronoun resolution: each item is a sentence with a blank (_) and
two candidate noun phrases. The model picks the correct filler.

Multiple size configurations are available via the ``config`` parameter.
Random baseline is 50% (2-way MC).

GPT-2 (124M) acc ≈ 52%; GPT-2-XL (1.5B) acc ≈ 58%.
"""
import os
from enum import Enum
from typing import Optional

from torch.utils.data import Dataset

from tunalab.evaluations.multiple_choice import MultipleChoiceItem


class Config(Enum):
    XS  = "winogrande_xs"   # 640 train
    S   = "winogrande_s"    # 1280 train
    M   = "winogrande_m"    # 2558 train
    L   = "winogrande_l"    # 5120 train
    XL  = "winogrande_xl"   # 40398 train (default)


class Split(Enum):
    TRAIN = "train"
    VAL   = "validation"
    TEST  = "test"


class WinoGrandeDataset(Dataset):
    """WinoGrande commonsense pronoun resolution dataset.

    Each item is a sentence with a ``_`` placeholder and two candidate nouns.
    The model selects the candidate that correctly fills the blank.

    Args:
        config: Dataset size variant (default: XL, the largest).
        split: Dataset split.
        cache_dir: Directory to cache downloaded data.
        limit: Maximum number of examples to load.
    """

    def __init__(
        self,
        config: Config = Config.XL,
        split: Split = Split.VAL,
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        from datasets import load_dataset

        if cache_dir is None:
            cache_dir = os.path.join("data", ".cache", "winogrande")

        raw = load_dataset(
            "winogrande",
            config.value,
            split=split.value,
            cache_dir=cache_dir,
        )

        items = []
        for ex in raw:
            # sentence contains "_"; build two completions by substituting each option
            sentence = ex["sentence"]
            opt1     = ex["option1"]
            opt2     = ex["option2"]
            answer   = ex["answer"]   # "1" or "2"

            if answer not in ("1", "2"):
                continue

            # Context: sentence up to (but not including) the blank "_".
            # Completion: the filler + rest of sentence.
            blank_idx = sentence.find("_")
            if blank_idx == -1:
                continue
            prefix = sentence[:blank_idx]
            suffix = sentence[blank_idx + 1:]

            choices = [opt1 + suffix, opt2 + suffix]
            label   = int(answer) - 1   # 0-indexed

            items.append(MultipleChoiceItem(
                context=prefix,
                choices=choices,
                label=label,
            ))

        if limit is not None:
            items = items[:limit]
        self.data = items

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> MultipleChoiceItem:
        return self.data[idx]
