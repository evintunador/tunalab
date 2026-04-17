"""
SciQ dataset for multiple-choice evaluation.

Reference: https://huggingface.co/datasets/allenai/sciq

Science exam questions (physics, chemistry, biology) with 4-way MC. Each item
has one correct answer and three distractors. An optional support passage is
included in the dataset but not used here (NLL-only evaluation over bare text).

Random baseline is 25%.
GPT-2 (124M) acc ≈ 55%; competitive models reach 95%+.
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


class SciQDataset(Dataset):
    """SciQ science multiple-choice dataset.

    The correct answer is always ``correct_answer``; distractors are
    ``distractor1``, ``distractor2``, ``distractor3``. Choices are shuffled
    so the correct answer appears at a random position among the four choices.

    Args:
        split: Dataset split.
        cache_dir: Directory to cache downloaded data.
        limit: Maximum number of examples to load.
        seed: Random seed for shuffling choice order (default: 42).
    """

    def __init__(
        self,
        split: Split = Split.VAL,
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None,
        seed: int = 42,
    ):
        import random
        from datasets import load_dataset

        if cache_dir is None:
            cache_dir = os.path.join("data", ".cache", "sciq")

        raw = load_dataset(
            "allenai/sciq",
            split=split.value,
            cache_dir=cache_dir,
        )

        rng = random.Random(seed)
        items = []
        for ex in raw:
            correct     = ex["correct_answer"]
            distractors = [ex["distractor1"], ex["distractor2"], ex["distractor3"]]
            choices     = distractors + [correct]
            rng.shuffle(choices)
            label = choices.index(correct)

            items.append(MultipleChoiceItem(
                context=ex["question"],
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
