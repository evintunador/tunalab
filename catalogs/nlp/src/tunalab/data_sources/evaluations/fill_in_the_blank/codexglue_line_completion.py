"""
CodeXGLUE line-level code completion dataset for fill-in-the-blank evaluation.

Reference: https://huggingface.co/datasets/google/code_x_glue_cc_code_completion_token

Each example is a tokenized Python (or Java) source file represented as a list
of space-separated tokens with special markers:
  <s>       — start of file
  </s>      — end of file
  <EOL>     — end of line (newline)

This adapter reconstructs plain-text code from the token list and creates
fill-in-the-blank items by masking the last complete line of each file. The
prompt is all content up to (but not including) the final non-empty line; the
answer is that final line.

The ``gt`` field in the ``code_completion_line`` variant is always empty (not
released); this adapter instead uses ``code_completion_token`` (100k Python /
50k test) which contains full files.

Primary metric: NLL over the last line's tokens (lower = better code
understanding). Exact-match accuracy is typically ~0 for NLL-only adapters.
"""
import os
from enum import Enum
from typing import List, Optional

from torch.utils.data import Dataset

from tunalab.evaluations.fill_in_the_blank import FillInTheBlankItem

_SPECIAL = {"<s>", "</s>"}


def _tokens_to_text(tokens: List[str]) -> str:
    """Reconstruct plain Python source from the CodeXGLUE token list."""
    parts = []
    for tok in tokens:
        if tok in _SPECIAL:
            continue
        if tok == "<EOL>":
            parts.append("\n")
        else:
            parts.append(tok)
    # Tokens are space-separated in the original; join them back
    return " ".join(parts).replace(" \n ", "\n").replace(" \n", "\n").replace("\n ", "\n")


class Language(Enum):
    PYTHON = "python"
    JAVA   = "java"


class Split(Enum):
    TRAIN = "train"
    TEST  = "test"


class CodeXGLUELineCompletionDataset(Dataset):
    """CodeXGLUE next-line fill-in-the-blank dataset.

    For each source file, the prompt is everything up to the last non-empty
    line; the answer is that final line.

    Args:
        language: Programming language (default: Python).
        split: Dataset split (train / test — no validation split available).
        cache_dir: Directory to cache downloaded data.
        limit: Maximum number of examples to load.
        min_answer_chars: Skip files whose last line is shorter than this
            (e.g. lone closing braces). Default: 10.
    """

    def __init__(
        self,
        language: Language = Language.PYTHON,
        split: Split = Split.TEST,
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None,
        min_answer_chars: int = 10,
    ):
        from datasets import load_dataset

        if cache_dir is None:
            cache_dir = os.path.join("data", ".cache", "codexglue_line_completion")

        raw = load_dataset(
            "google/code_x_glue_cc_code_completion_token",
            language.value,
            split=split.value,
            cache_dir=cache_dir,
        )

        items = []
        for ex in raw:
            text  = _tokens_to_text(ex["code"])
            lines = [l for l in text.splitlines() if l.strip()]
            if len(lines) < 2:
                continue

            answer = lines[-1]
            if len(answer) < min_answer_chars:
                continue

            prompt = "\n".join(lines[:-1])
            items.append(FillInTheBlankItem(prompt=prompt, answer=answer))

        if limit is not None:
            items = items[:limit]
        self.data = items

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> FillInTheBlankItem:
        return self.data[idx]
