from enum import Enum
from typing import Optional, Union, Callable, Iterable, List
from pathlib import Path
import os
import multiprocessing as mp
from functools import partial
import logging
import requests

import torch
import numpy as np
from tqdm import tqdm

from .shard_io import BinaryShardIO


logger = logging.getLogger(__name__)


class SequentialPretokenizedDatasetMixin:
    """
    Utility for building and reading token cache shards:
    - Shards are .bin files with 256*int32 header + tokens of uint{8,16,32,64}
    - First shard is 'val'; others are 'train'
    - Also provides indexing and slicing across shards given a seq_len
    """

    def __init__(
        self,
        save_dir: Union[str, Path],
        shard_size: int = 2**27,
        max_num_shards: Optional[int] = None,
        cache_filename_prefix: str = "dataset",
        num_workers: Optional[int] = None,
    ):
        self._cache_dir = Path(save_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._shard_size = int(shard_size)
        self._max_num_shards = max_num_shards
        self._cache_filename_prefix = cache_filename_prefix
        self._num_workers = num_workers or max(1, (os.cpu_count() or 2) - 2)
        
        logger.info(f"Initialized PrecachedDatasetMixin with cache_dir={self._cache_dir}, shard_size={self._shard_size:,}, num_workers={self._num_workers}")

        # indexing-related state (populated by setup_cache_index)
        self._files: List[Path] = []
        self._memmaps: List[np.memmap] = []
        self._shard_sizes: Optional[np.ndarray] = None
        self._cumsum: Optional[np.ndarray] = None
        self._seq_len: Optional[int] = None
        self._num_items: int = 0
        self._split: Optional[str] = None

    @staticmethod
    def _tokenize_worker(
        doc: str,
        encode: Callable[[str], List[int]] = None,
        doc_separator: Optional[int] = None,
        dtype: np.dtype = np.uint16
    ) -> np.ndarray:
        toks = encode(doc)
        if doc_separator:
            toks.append(doc_separator)
        return np.asarray(toks, dtype=dtype)

    def _cache_glob(self, split: str) -> List[Path]:
        return sorted(self._cache_dir.glob(f"{self._cache_filename_prefix}_{split}_*.bin"))

    def has_cache(self) -> bool:
        has_val = bool(self._cache_glob("val"))
        has_train = bool(self._cache_glob("train"))
        has_both = has_val and has_train
        if has_both:
            logger.info("Found existing cache for both train and val splits")
        return has_both

    def build_cache(
        self,
        doc_iter: Iterable[str],
        tokenizer_encode_fn: Callable[[str], List[int]],
        doc_separator: Optional[int] = None,
        token_dtype: np.dtype = np.uint16,
    ) -> None:
        if self.has_cache():
            logger.info("Cache already exists, skipping build")
            return

        logger.info(f"Building cache with {self._num_workers} workers, dtype={token_dtype}")

        shard_index = 0
        token_count = 0
        all_tokens_np = np.empty((self._shard_size,), dtype=token_dtype)
        pbar = None

        use_workers = self._num_workers > 1
        worker = SequentialPretokenizedDatasetMixin._tokenize_worker

        if use_workers:
            with mp.Pool(self._num_workers) as pool:
                imap_iter = pool.imap(
                    partial(worker, encode=tokenizer_encode_fn, doc_separator=doc_separator),
                    doc_iter,
                    chunksize=16,
                )
                for tokens in imap_iter:
                    if token_count + len(tokens) < self._shard_size:
                        all_tokens_np[token_count : token_count + len(tokens)] = tokens
                        token_count += len(tokens)
                        if pbar is None:
                            pbar = tqdm(total=self._shard_size, unit="tokens", desc=f"Shard {shard_index}")
                        pbar.update(len(tokens))
                    else:
                        split = "val" if shard_index == 0 else "train"
                        filename = self._cache_dir / f"{self._cache_filename_prefix}_{split}_{shard_index:06d}.bin"
                        remainder = self._shard_size - token_count
                        if pbar is None:
                            pbar = tqdm(total=self._shard_size, unit="tokens", desc=f"Shard {shard_index}")
                        pbar.update(remainder)
                        all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
                        BinaryShardIO.write_datafile(str(filename), all_tokens_np)
                        shard_index += 1

                        if self._max_num_shards is not None and shard_index >= self._max_num_shards + 1:
                            break

                        pbar = None
                        leftover = len(tokens) - remainder
                        all_tokens_np[0:leftover] = tokens[remainder:]
                        token_count = leftover
        else:
            for doc in doc_iter:
                tokens = worker(doc, encode=tokenizer_encode_fn, doc_separator=doc_separator)
                if token_count + len(tokens) < self._shard_size:
                    all_tokens_np[token_count : token_count + len(tokens)] = tokens
                    token_count += len(tokens)
                    if pbar is None:
                        pbar = tqdm(total=self._shard_size, unit="tokens", desc=f"Shard {shard_index}")
                    pbar.update(len(tokens))
                else:
                    split = "val" if shard_index == 0 else "train"
                    filename = self._cache_dir / f"{self._cache_filename_prefix}_{split}_{shard_index:06d}.bin"
                    remainder = self._shard_size - token_count
                    if pbar is None:
                        pbar = tqdm(total=self._shard_size, unit="tokens", desc=f"Shard {shard_index}")
                    pbar.update(remainder)
                    all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
                    BinaryShardIO.write_datafile(str(filename), all_tokens_np)
                    shard_index += 1

                    if self._max_num_shards is not None and shard_index >= self._max_num_shards + 1:
                        break

                    pbar = None
                    leftover = len(tokens) - remainder
                    all_tokens_np[0:leftover] = tokens[remainder:]
                    token_count = leftover

        if token_count != 0 and (self._max_num_shards is None or shard_index < self._max_num_shards + 1):
            split = "val" if shard_index == 0 else "train"
            filename = self._cache_dir / f"{self._cache_filename_prefix}_{split}_{shard_index:06d}.bin"
            BinaryShardIO.write_datafile(str(filename), all_tokens_np[:token_count])

    def _split_to_str(self, split: Union[str, "Enum"]) -> str:
        return split.value if hasattr(split, "value") else str(split)

    def setup_cache_index(self, split: Union[str, "Enum"], seq_len: int) -> None:
        """
        Open memmaps and prepare sharded indexing for the given split and seq_len
        """
        split_str = self._split_to_str(split)
        self._split = split_str
        self._seq_len = int(seq_len)

        logger.info(f"Setting up cache index for split={split_str}, seq_len={seq_len}")

        self._files = self._cache_glob(split_str)
        if not self._files:
            logger.error(f"No cached shards found for split='{split_str}' in {self._cache_dir}")
            raise RuntimeError(f"No cached shards found for split='{split_str}' in {self._cache_dir}")

        logger.debug(f"Found {len(self._files)} shard files for split={split_str}")
        self._memmaps = [BinaryShardIO.read_datafile_tokens_memmap(p) for p in self._files]
        self._shard_sizes = np.array(
            [BinaryShardIO.read_datafile_token_count(p) for p in self._files], dtype=np.int64
        )
        self._cumsum = np.cumsum(np.concatenate([[0], self._shard_sizes]))
        total_tokens = int(self._shard_sizes.sum())
        self._num_items = total_tokens // self._seq_len

        logger.info(f"Cache index ready: {total_tokens:,} tokens, {self._num_items:,} sequences of length {seq_len}")

    def __len__(self) -> int:
        if self._seq_len is None:
            raise RuntimeError("Cache index not initialized. Call setup_cache_index(split, seq_len) first.")
        return self._num_items

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._seq_len is None:
            raise RuntimeError("Cache index not initialized. Call setup_cache_index(split, seq_len) first.")
        if idx < 0 or idx >= self._num_items:
            raise IndexError(idx)
        start = idx * self._seq_len
        end = start + self._seq_len
        return self._slice_tokens(start, end).to(torch.long)

    def _slice_tokens(self, start: int, end: int) -> torch.Tensor:
        out = np.empty((end - start,), dtype=self._memmaps[0].dtype)
        write_pos = 0
        cur = start
        while cur < end:
            k = int(np.searchsorted(self._cumsum, cur, side="right")) - 1
            shard_offset = cur - int(self._cumsum[k])
            shard_available = int(self._shard_sizes[k] - shard_offset)
            need = end - cur
            take = min(shard_available, need)
            mm = self._memmaps[k]
            out[write_pos : write_pos + take] = mm[shard_offset : shard_offset + take]
            write_pos += take
            cur += take
        return torch.from_numpy(out)