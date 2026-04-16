"""
LAMBADA dataset for fill-in-the-blank evaluation.

Reference: https://huggingface.co/datasets/EleutherAI/lambada_openai

Each item is a short passage where the model must predict the last word.
The prompt is all text up to (but not including) the final whitespace-delimited
word; the answer is that final word.

Standard metric: perplexity on the last word (NLL-based). Exact-match accuracy
is reported but typically zero for NLL-only adapters that return ("", nll).
"""
import os
from typing import Optional

from torch.utils.data import Dataset

from tunalab.evaluations.fill_in_the_blank import FillInTheBlankItem


class LambadaDataset(Dataset):
    """LAMBADA OpenAI test set for last-word prediction.

    Uses EleutherAI/lambada_openai which is the standard version used in
    LLM evaluations. Only a test split is available.

    Args:
        cache_dir: Directory to cache the downloaded data.
        limit: Maximum number of examples to load.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        from datasets import load_dataset

        if cache_dir is None:
            cache_dir = os.path.join("data", ".cache", "lambada")

        raw = load_dataset(
            "EleutherAI/lambada_openai",
            split="test",
            cache_dir=cache_dir,
        )

        items = []
        for ex in raw:
            text = ex["text"].rstrip()
            idx = text.rfind(" ")
            if idx == -1:
                continue
            prompt = text[:idx]
            answer = text[idx + 1:]
            if not answer:
                continue
            items.append(FillInTheBlankItem(prompt=prompt, answer=answer))

        if limit is not None:
            items = items[:limit]
        self.data = items

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> FillInTheBlankItem:
        return self.data[idx]
