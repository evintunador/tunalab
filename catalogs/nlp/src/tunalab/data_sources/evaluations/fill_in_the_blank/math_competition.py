"""
Hendrycks MATH competition dataset for fill-in-the-blank evaluation.

Reference: https://huggingface.co/datasets/EleutherAI/hendrycks_math

Competition-level mathematics (AMC/AIME difficulty) with dense LaTeX notation
throughout. Each item is a problem with a multi-step solution ending in
``\\boxed{answer}``. The adapter scores NLL over the full solution given the
problem — a perplexity measure over mathematical LaTeX.

This does NOT check answer correctness; results are not comparable to published
solve-rate numbers. Report as "perplexity on canonical solution".

Seven subjects are available (see ``Subject`` enum). Items with empty solutions
are skipped.
"""
import os
from enum import Enum
from typing import Optional

from torch.utils.data import Dataset

from tunalab.evaluations.fill_in_the_blank import FillInTheBlankItem


class Subject(Enum):
    ALGEBRA                = "algebra"
    COUNTING_PROBABILITY   = "counting_and_probability"
    GEOMETRY               = "geometry"
    INTERMEDIATE_ALGEBRA   = "intermediate_algebra"
    NUMBER_THEORY          = "number_theory"
    PREALGEBRA             = "prealgebra"
    PRECALCULUS            = "precalculus"


class Split(Enum):
    TRAIN = "train"
    TEST  = "test"


class MATHDataset(Dataset):
    """Hendrycks MATH fill-in-the-blank dataset.

    The prompt is the problem statement; the answer is the full solution
    (prefixed with a newline to match training context boundaries).

    Args:
        subject: Math subject (default: algebra).
        split: Dataset split.
        cache_dir: Directory to cache downloaded data.
        limit: Maximum number of examples to load.
    """

    def __init__(
        self,
        subject: Subject = Subject.ALGEBRA,
        split: Split = Split.TEST,
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        from datasets import load_dataset

        if cache_dir is None:
            cache_dir = os.path.join("data", ".cache", "math_competition")

        raw = load_dataset(
            "EleutherAI/hendrycks_math",
            subject.value,
            split=split.value,
            cache_dir=cache_dir,
        )

        items = []
        for ex in raw:
            if not ex["solution"].strip():
                continue
            items.append(FillInTheBlankItem(
                prompt=ex["problem"],
                answer="\n" + ex["solution"],
            ))

        if limit is not None:
            items = items[:limit]
        self.data = items

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> FillInTheBlankItem:
        return self.data[idx]
