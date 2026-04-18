"""
MathQA dataset for multiple-choice evaluation.

Reference: https://math-qa.github.io/math-QA/
           (original paper: Amini et al., 2019)

2985 5-way multiple-choice math word problems. Options are encoded as a single
string like ``"a ) 24 , b ) 120 , c ) 625 , d ) 720 , e ) 5040"``; this adapter
parses them into a list. Random baseline is 20%.

The HuggingFace dataset (allenai/math_qa) uses a legacy loading script that is
no longer supported by datasets>=4.x. This adapter loads from raw JSON files
downloaded from the project website. Download instructions:

    wget https://math-qa.github.io/math-QA/data/MathQA.zip
    unzip MathQA.zip -d data/.cache/mathqa/

The resulting directory must contain ``test.json`` (and optionally ``train.json``,
``dev.json``).
"""
import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset

from tunalab.evaluations.multiple_choice import MultipleChoiceItem

_OPT_RE = re.compile(r"[a-e]\s*\)\s*(.+?)(?=\s*,\s*[a-e]\s*\)|\s*$)")
_LETTER_TO_IDX = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}


class Split(Enum):
    TRAIN = "train"
    VAL   = "dev"
    TEST  = "test"


class MathQADataset(Dataset):
    """MathQA 5-way multiple-choice math word problem dataset.

    Loads from raw JSON files extracted from the MathQA zip. Items with
    malformed options strings (not parseable into exactly 5 choices) are
    silently skipped.

    Args:
        split: Dataset split.
        data_dir: Directory containing the extracted JSON files. Defaults to
            ``data/.cache/mathqa/`` relative to the current working directory.
        limit: Maximum number of examples to load.
    """

    def __init__(
        self,
        split: Split = Split.TEST,
        data_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        if data_dir is None:
            data_dir = os.path.join("data", ".cache", "mathqa")

        json_path = Path(data_dir) / f"{split.value}.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"MathQA data file not found: {json_path}. "
                "Download from https://math-qa.github.io/math-QA/data/MathQA.zip "
                f"and extract to {data_dir}."
            )

        raw = json.loads(json_path.read_text())

        items = []
        for ex in raw:
            matches = _OPT_RE.findall(ex["options"])
            if len(matches) != 5:
                continue
            label = _LETTER_TO_IDX.get(ex["correct"].strip().lower())
            if label is None:
                continue
            items.append(MultipleChoiceItem(
                context=ex["Problem"],
                choices=[m.strip() for m in matches],
                label=label,
            ))

        if limit is not None:
            items = items[:limit]
        self.data = items

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> MultipleChoiceItem:
        return self.data[idx]
