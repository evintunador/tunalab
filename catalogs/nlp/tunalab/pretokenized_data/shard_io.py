from typing import Optional, Union
from pathlib import Path
import os
import logging
import requests

import torch
import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


class BinaryShardIO:
    @staticmethod
    def pick_token_dtype(vocab_size: int) -> np.dtype:
        """Selects the smallest uint dtype that can hold the vocabulary."""
        if vocab_size < 2**8:
            return np.uint8
        elif vocab_size < 2**16:
            return np.uint16
        elif vocab_size < 2**32:
            return np.uint32
        else:
            return np.uint64

    @staticmethod
    def write_datafile(filename, toks: np.ndarray):
        """
        Saves token data as a .bin file, for reading in C.
        - First comes a header with 256 int32s
        - The tokens follow
        """
        assert len(toks) < 2**31, "token count too large"  # ~2.1B tokens
        dtype_size = toks.dtype.itemsize
        header = np.zeros(256, dtype=np.int32)
        header[0] = 11041999  # magic number for file format identification/validation
        header[1] = 1  # version
        header[2] = len(toks)  # number of tokens after the header
        header[3] = dtype_size  # dtype of tokens after the header

        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        logger.info(f"writing {len(toks):,} tokens to {filename}")
        with open(filename, "wb") as f:
            f.write(header.tobytes())
            f.write(toks.tobytes())

    @staticmethod
    def read_datafile_tokens_memmap(path: Union[str, Path], dtype: Optional[np.dtype] = None) -> np.memmap:
        """Memory-map the token payload after the 256*4 byte header as uint16."""
        path = os.fspath(path)
        header_bytes = 256 * 4
        nbytes = BinaryShardIO.read_datafile_token_dtype(path)
        if dtype is None:
            if nbytes == 1:
                dtype = np.uint8
            elif nbytes == 2:
                dtype = np.uint16
            elif nbytes == 4:
                dtype = np.uint32
            elif nbytes == 8:
                dtype = np.uint64
            else:
                raise ValueError(f"Unsupported token dtype size in header: {nbytes}")
        else:
            assert (
                np.dtype(dtype).itemsize == nbytes
            ), f"Intended dataset read size {dtype} does not match data storage type {nbytes} bytes"
        return np.memmap(path, mode="r", dtype=dtype, offset=header_bytes)

    @staticmethod
    def read_datafile_token_count(path: Union[str, Path]) -> int:
        """Read header[2] which stores token count."""
        with open(path, "rb") as f:
            header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        return int(header[2])

    @staticmethod
    def read_datafile_token_dtype(path: Union[str, Path]) -> int:
        """Read header[3] which stores token dtype"""
        with open(path, "rb") as f:
            header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        return int(header[3])