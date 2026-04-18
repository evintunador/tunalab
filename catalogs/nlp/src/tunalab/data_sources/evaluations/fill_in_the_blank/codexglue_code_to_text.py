"""
CodeXGLUE code-to-text (docstring generation) dataset for fill-in-the-blank evaluation.

Reference: https://huggingface.co/datasets/google/code_x_glue_ct_code_to_text

Each item is a function body paired with its docstring. The model scores the NLL
of the docstring given the code — testing whether the model has learned the
association between code structure and natural-language summaries.

Particularly relevant for Stack-trained models: The Stack corpus contains millions
of Python functions with docstrings, and this benchmark directly exercises that
learned association.

Available languages: Python (14,918 test items), Go, Java, JavaScript, PHP, Ruby.
"""
import os
from enum import Enum
from typing import Optional

from torch.utils.data import Dataset

from tunalab.evaluations.fill_in_the_blank import FillInTheBlankItem


class Language(Enum):
    PYTHON     = "python"
    GO         = "go"
    JAVA       = "java"
    JAVASCRIPT = "javascript"
    PHP        = "php"
    RUBY       = "ruby"


class Split(Enum):
    TRAIN = "train"
    VAL   = "validation"
    TEST  = "test"


class CodeXGLUECodeToTextDataset(Dataset):
    """CodeXGLUE code-to-text fill-in-the-blank dataset.

    The prompt is the function code; the answer is the docstring (prefixed with
    a newline). Items with empty code or docstring are skipped.

    Args:
        language: Programming language (default: Python).
        split: Dataset split.
        cache_dir: Directory to cache downloaded data.
        limit: Maximum number of examples to load.
    """

    def __init__(
        self,
        language: Language = Language.PYTHON,
        split: Split = Split.TEST,
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        from datasets import load_dataset

        if cache_dir is None:
            cache_dir = os.path.join("data", ".cache", "codexglue_code_to_text")

        raw = load_dataset(
            "google/code_x_glue_ct_code_to_text",
            language.value,
            split=split.value,
            cache_dir=cache_dir,
        )

        items = []
        for ex in raw:
            code      = ex["code"]
            docstring = ex["docstring"]
            if not code.strip() or not docstring.strip():
                continue
            items.append(FillInTheBlankItem(
                prompt=code,
                answer="\n" + docstring,
            ))

        if limit is not None:
            items = items[:limit]
        self.data = items

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> FillInTheBlankItem:
        return self.data[idx]
