"""
HumanEvalPack canonical-vs-buggy dataset for multiple-choice evaluation.

Reference: https://huggingface.co/datasets/bigcode/humanevalpack

164 programming problems per language (Python, C++, Go, Java, JavaScript, Rust).
Each item has a function signature + docstring as context, a canonical (correct)
solution, and a buggy solution with a single introduced error. The model picks
whichever solution has lower NLL — a 2-way contrastive task requiring no code
execution.

Random baseline is 50%. Label 0 is always canonical (correct by construction).

Primary metric: accuracy (fraction of items where the model correctly assigns
lower NLL to the canonical solution than the buggy one).
"""
import os
from enum import Enum
from typing import Optional

from torch.utils.data import Dataset

from tunalab.evaluations.multiple_choice import MultipleChoiceItem


class Language(Enum):
    PYTHON = "python"
    CPP    = "cpp"
    GO     = "go"
    JAVA   = "java"
    JS     = "js"
    RUST   = "rust"


class HumanEvalBuggyDataset(Dataset):
    """HumanEvalPack canonical-vs-buggy 2-way multiple-choice dataset.

    Items with empty canonical or buggy solutions are skipped.

    Args:
        language: Programming language (default: Python).
        cache_dir: Directory to cache downloaded data.
        limit: Maximum number of examples to load.
    """

    def __init__(
        self,
        language: Language = Language.PYTHON,
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        from datasets import load_dataset

        if cache_dir is None:
            cache_dir = os.path.join("data", ".cache", "humanevalpack")

        raw = load_dataset(
            "bigcode/humanevalpack",
            language.value,
            split="test",
            cache_dir=cache_dir,
        )

        items = []
        for ex in raw:
            canonical = ex["canonical_solution"]
            buggy     = ex.get("buggy_solution", "")
            if not canonical.strip() or not buggy.strip():
                continue
            items.append(MultipleChoiceItem(
                context=ex["prompt"],
                choices=[canonical, buggy],
                label=0,
            ))

        if limit is not None:
            items = items[:limit]
        self.data = items

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> MultipleChoiceItem:
        return self.data[idx]
