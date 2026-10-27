"""Dataset loading, document boundary extraction, and context-window chunking.

This module provides functions and classes for:
1. Loading official splits of the WikiText-2 raw dataset without cross-split leakage.
2. Grouping raw lines into discrete documents using a top-level Wikipedia header heuristic.
3. Appending the special <|endoftext|> (EOT) token exactly once per document.
4. Constructing causal next-token prediction sample pairs (x, y) of exact length T
   from (T + 1)-token contiguous slices within each document.
5. Providing a standard PyTorch Dataset and DataLoader interface.

CRITICAL INVARIANTS:
- Every training sample consists of x = chunk[0:T] and y = chunk[1:T+1].
- Both x and y have shape (T,) and dtype torch.long.
- Exact 1-token shift: y[i] == x[i+1] for 0 <= i < T-1, and y[T-1] is the (T+1)-th token.
- No window ever crosses a document boundary.
- No network requests or dataset downloads occur at module import time.
"""

import re
from typing import List, Optional, Set, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from src.data.tokenizer import GPT2Tokenizer

# Top-level article header pattern in WikiText-2 raw.
# Example: " = Homura Akemi = \n"
# NOTE ON HEURISTIC:
# In WikiText-2, top-level article titles are flanked by a single '=' on each side.
# Section headers use nested '=' signs (e.g., ' = = Section = = ' or ' = = = Sub = = = ').
# This heuristic is specific to the standard WikiText-2 formatting convention.
# It is not a universal markup parser. Subsections and ordinary text lines remain
# within their parent document.
ARTICLE_HEADER_PATTERN = re.compile(r"^\s*=\s+([^=]+?)\s+=\s*$")


def is_article_header(line: str) -> bool:
    """Checks whether a line matches the top-level article header heuristic.

    Args:
        line: Raw text line to inspect.

    Returns:
        True if the line represents a top-level article header; False otherwise.
    """
    return bool(ARTICLE_HEADER_PATTERN.match(line.strip()))


def extract_documents_from_raw_lines(lines: List[str]) -> List[str]:
    """Groups raw lines into distinct documents using top-level header boundaries.

    Policy:
    - Top-level headers (' = Title = ') mark the beginning of a new article.
    - Subsections (' = = Heading = = ') and content lines belong to the active article.
    - Blank lines (whitespace only) are treated as paragraph breaks and are collapsed
      into newlines rather than creating empty documents or spurious boundaries.
    - If no lines in the input match the top-level header heuristic, all content lines
      are safely grouped into a single continuous document.

    Args:
        lines: Sequence of raw text lines (e.g. from WikiText rows).

    Returns:
        List of non-empty document strings.
    """
    documents: List[str] = []
    current_doc_lines: List[str] = []

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            # Blank line: paragraph separator; do not start a document solely from blank lines
            continue

        if is_article_header(stripped):
            # Finish previous document if it accumulated content
            if current_doc_lines:
                doc_text = "\n".join(current_doc_lines).strip()
                if doc_text:
                    documents.append(doc_text)
                current_doc_lines = []

        current_doc_lines.append(stripped)

    # Append any remaining accumulated lines
    if current_doc_lines:
        doc_text = "\n".join(current_doc_lines).strip()
        if doc_text:
            documents.append(doc_text)

    return documents


def tokenize_documents(
    documents: List[str],
    tokenizer: GPT2Tokenizer,
    append_eot: bool = True,
) -> List[List[int]]:
    """Encodes a list of document strings into token ID lists, optionally appending EOT.

    EOT Handling:
    - When append_eot is True, the tokenizer's actual integer EOT ID (50256) is appended
      directly to the token list exactly once per non-empty document.
    - Direct integer insertion avoids string-parsing ambiguity and guarantees that EOT
      is not duplicated or rejected by disallowed special-token guards.

    Args:
        documents: List of clean document strings.
        tokenizer: Initialized GPT2Tokenizer instance.
        append_eot: Whether to append EOT token ID at the end of each document.

    Returns:
        List of integer token ID lists, one per non-empty document.
    """
    tokenized_docs: List[List[int]] = []
    eot_id = tokenizer.eot_token_id

    for doc in documents:
        if not doc.strip():
            continue
        tokens = tokenizer.encode(doc)
        if not tokens:
            continue
        if append_eot:
            tokens.append(eot_id)
        tokenized_docs.append(tokens)

    return tokenized_docs


def build_chunks_from_token_lists(
    token_lists: List[List[int]],
    context_length: int = 64,
    stride: Optional[int] = None,
    include_end_chunk: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Constructs causal next-token prediction sample pairs (x, y) from token lists.

    For every sample:
    - A contiguous slice of exactly T + 1 tokens is extracted from the same document.
    - x = chunk[0:T]
    - y = chunk[1:T+1]
    - Both tensors have shape (T,) and dtype torch.long.
    - Exact 1-token shift invariant: y[i] == x[i+1] for 0 <= i < T-1.
    - y[T-1] is the (T+1)-th token in the chunk.

    Stride & Start Positions:
    - Regular starts begin at 0, stride, 2*stride, ... as long as s + T + 1 <= len(tokens).
    - If stride is None, stride defaults to context_length (non-overlapping input blocks).

    End-Anchored Window & Deduplication:
    - To ensure the model learns to predict EOT at document termination, when
      include_end_chunk is True and len(tokens) >= T + 1, a final window anchored
      at the end of the document is evaluated: s_end = len(tokens) - (T + 1).
    - Deduplication: If s_end was already generated by regular stride starts, it is NOT
      duplicated.
    - Note on Overlap: When added, s_end overlaps earlier windows, meaning some token
      transitions will be sampled more than once in training.

    Short Documents:
    - Documents with strictly fewer than T + 1 tokens are omitted without padding.

    Args:
        token_lists: List of token ID sequences (one list per document).
        context_length: Context window length T (default: 64).
        stride: Step size between consecutive window starts (default: context_length).
        include_end_chunk: Whether to include an end-anchored window to capture EOT.

    Returns:
        List of (x, y) tensor tuples.
    """
    if context_length < 1:
        raise ValueError(f"context_length must be >= 1, got {context_length}.")

    t = context_length
    s_step = stride if stride is not None else t
    if s_step < 1:
        raise ValueError(f"stride must be >= 1, got {s_step}.")

    samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for tokens in token_lists:
        m = len(tokens)
        if m < t + 1:
            # Document too short to form a single full (T + 1)-token window; skip without padding
            continue

        used_starts: Set[int] = set()

        # 1. Regular stride-based starts: s = 0, stride, 2*stride, ...
        s = 0
        while s + t + 1 <= m:
            chunk = tokens[s : s + t + 1]
            x_tensor = torch.tensor(chunk[:t], dtype=torch.long)
            y_tensor = torch.tensor(chunk[1 : t + 1], dtype=torch.long)
            samples.append((x_tensor, y_tensor))
            used_starts.add(s)
            s += s_step

        # 2. Optional end-anchored window: s_end = m - (t + 1)
        if include_end_chunk:
            s_end = m - (t + 1)
            if s_end not in used_starts:
                chunk = tokens[s_end : s_end + t + 1]
                x_tensor = torch.tensor(chunk[:t], dtype=torch.long)
                y_tensor = torch.tensor(chunk[1 : t + 1], dtype=torch.long)
                samples.append((x_tensor, y_tensor))
                used_starts.add(s_end)

    return samples


class GPTDataset(Dataset):
    """PyTorch Dataset yielding (x, y) tensor pairs for causal language modeling."""

    def __init__(self, samples: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Initializes the dataset with pre-chunked (x, y) tensor pairs."""
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def create_dataloader(
    dataset: GPTDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Wraps a GPTDataset in a standard PyTorch DataLoader.

    Yields batches of inputs and targets with shape (B, T) and dtype torch.long.

    Args:
        dataset: Initialized GPTDataset.
        batch_size: Batch size B.
        shuffle: Whether to shuffle samples at each epoch.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        PyTorch DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def load_wikitext_split(
    split: str,
    dataset_name: str = "Salesforce/wikitext",
    config_name: str = "wikitext-2-raw-v1",
    cache_dir: Optional[str] = None,
) -> List[str]:
    """Loads raw text rows for the specified split from Hugging Face Datasets.

    IMPORTANT SAFETY:
    - This function connects to Hugging Face and downloads/caches data ONLY when called.
    - It is never called during module import or offline unit tests.
    - Validates split name to prevent accidentally mixing train, validation, or test data.

    Args:
        split: One of 'train', 'validation', or 'test'.
        dataset_name: Hugging Face dataset identifier.
        config_name: Dataset configuration name.
        cache_dir: Optional custom cache directory.

    Returns:
        List of raw string lines from the dataset.
    """
    valid_splits = {"train", "validation", "test"}
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Expected one of {sorted(valid_splits)}.")

    from datasets import load_dataset

    hf_dataset = load_dataset(
        dataset_name,
        config_name,
        split=split,
        cache_dir=cache_dir,
    )
    return [row["text"] for row in hf_dataset]
