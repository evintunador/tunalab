"""
CommonsenseQA dataset for multiple-choice evaluation.

Reference: https://huggingface.co/datasets/tau/commonsense_qa

5-way multiple-choice commonsense reasoning questions. Each item presents a
question and five answer candidates (labelled A–E); the model picks the correct
one by minimum NLL.

Random baseline is 20%.
GPT-2 (124M) acc ≈ 32%; competitive models reach 75%+.
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


class CommonsenseQADataset(Dataset):
    """CommonsenseQA 5-way multiple-choice dataset.

    Args:
        split: Dataset split. Note: test labels are not publicly released;
            use VAL for scored evaluation.
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
            cache_dir = os.path.join("data", ".cache", "commonsense_qa")

        raw = load_dataset(
            "tau/commonsense_qa",
            split=split.value,
            cache_dir=cache_dir,
        )

        items = []
        for ex in raw:
            choices_dict = ex["choices"]          # {"label": ["A","B","C","D","E"], "text": [...]}
            choice_labels = choices_dict["label"]
            choice_texts  = choices_dict["text"]
            answer_key    = ex["answerKey"]       # "A"–"E" (empty string in test split)

            if not answer_key:
                continue  # test split has no labels

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
