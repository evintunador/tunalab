"""
RepoBench-C (cross-file next-line completion) dataset for fill-in-the-blank evaluation.

Reference: https://huggingface.co/datasets/tianyang/repobench_python_v1.1
           (ICLR 2024: Liu et al., "RepoBench: Benchmarking Repository-Level
           Code Auto-Completion Systems")

Each item provides cross-file context snippets (from imported modules in the
same repository) plus the current file prefix; the answer is the next line.
Three splits test different levels of cross-file dependency:

  cross_file_first  — first use of a cross-file import: highest expected benefit
                      from cross-document attention.
  cross_file_random — random line with a cross-file dependency.
  in_file           — random line with no cross-file dependency (control split).

Context construction: cross-file snippets are concatenated first (separated by
blank lines), then the current file prefix — mirroring how cross_doc_link models
see auxiliary documents before the active document.

Schema (tianyang/repobench_python_v1.1):
  context      — list of {'identifier': str, 'path': str, 'snippet': str}
  cropped_code — current file prefix up to the target line
  next_line    — ground truth next line to predict
"""
import os
from enum import Enum
from typing import Optional

from torch.utils.data import Dataset

from tunalab.evaluations.fill_in_the_blank import FillInTheBlankItem

#: HuggingFace repo ID for RepoBench Python v1.1 (moved from Leolty/ to tianyang/).
REPOBENCH_HF_REPO = "tianyang/repobench_python_v1.1"


class Split(Enum):
    CROSS_FILE_FIRST  = "cross_file_first"
    CROSS_FILE_RANDOM = "cross_file_random"
    IN_FILE           = "in_file"


class RepoBenchDataset(Dataset):
    """RepoBench-C Python cross-file next-line completion dataset.

    Items with an empty next_line are skipped.

    Args:
        split: Dataset split (default: cross_file_first).
        cache_dir: Directory to cache downloaded data.
        limit: Maximum number of examples to load.
    """

    def __init__(
        self,
        split: Split = Split.CROSS_FILE_FIRST,
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        from datasets import load_dataset

        if cache_dir is None:
            cache_dir = os.path.join("data", ".cache", "repobench")

        raw = load_dataset(
            REPOBENCH_HF_REPO,
            split=split.value,
            cache_dir=cache_dir,
        )

        items = []
        for ex in raw:
            next_line = ex.get("next_line", "")
            if not next_line.strip():
                continue

            # context is a list of {'identifier', 'path', 'snippet'} dicts.
            cross_snippets = [c["snippet"] for c in ex.get("context", []) if c.get("snippet")]
            current_prefix = ex.get("cropped_code", "")

            if cross_snippets:
                context = "\n\n".join(cross_snippets) + "\n\n" + current_prefix
            else:
                context = current_prefix

            items.append(FillInTheBlankItem(
                prompt=context,
                answer="\n" + next_line,
            ))

        if limit is not None:
            items = items[:limit]
        self.data = items

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> FillInTheBlankItem:
        return self.data[idx]
