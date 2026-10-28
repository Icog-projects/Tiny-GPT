
"""Dataset loading, document extraction, and context-window creation.

This module supports:
1. Loading WikiText-2 train, validation, and test splits.
2. Extracting documents using a top-level article-header heuristic.
3. Tokenizing documents and optionally appending the EOT token.
4. Creating causal next-token prediction samples.
5. Providing a PyTorch Dataset and DataLoader.

Each sample uses T + 1 consecutive tokens:
    x = chunk[:T]
    y = chunk[1:T + 1]

Therefore, y is shifted one token ahead of x.

Windows never cross document boundaries. Dataset loading occurs only when
load_wikitext_split() is called.
"""

import re
from typing import List, Optional, Set, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.tokenizer import GPT2Tokenizer


# Matches top-level headers such as:
# " = Homura Akemi = "
ARTICLE_HEADER_PATTERN = re.compile(r"^\s*=\s+([^=]+?)\s+=\s*$")


def is_article_header(line: str) -> bool:
    """Return True for a top-level article header.

    WikiText-2 table-column labels can also match the general header pattern.
    Those labels commonly contain semicolons, so exclude them.

    This is a dataset-specific heuristic, not a universal Wikipedia parser.
    """
    match = ARTICLE_HEADER_PATTERN.match(line.strip())

    if not match:
        return False

    # Avoid treating table-column labels as article boundaries.
    if ";" in match.group(1):
        return False

    return True


def extract_documents_from_raw_lines(lines: List[str]) -> List[str]:
    """Group raw text lines into documents using article-header boundaries.

    Policy:
    - A recognized top-level article header starts a new document.
    - Blank or whitespace-only lines are ignored.
    - Other non-empty lines remain in the current document.
    - If no headers are found, all non-empty lines form one document.
    """
    documents: List[str] = []
    current_doc_lines: List[str] = []

    for raw_line in lines:
        stripped = raw_line.strip()

        if not stripped:
            continue

        if is_article_header(stripped):
            if current_doc_lines:
                document = "\n".join(current_doc_lines).strip()

                if document:
                    documents.append(document)

                current_doc_lines = []

        current_doc_lines.append(stripped)

    if current_doc_lines:
        document = "\n".join(current_doc_lines).strip()

        if document:
            documents.append(document)

    return documents


def tokenize_documents(
    documents: List[str],
    tokenizer: GPT2Tokenizer,
    append_eot: bool = True,
) -> List[List[int]]:
    """Encode each non-empty document into GPT-2 token IDs.

    When append_eot is True, append the tokenizer's end-of-text token ID
    once to each non-empty document. Empty documents are skipped.
    """
    tokenized_docs: List[List[int]] = []
    eot_id = tokenizer.eot_token_id

    for document in documents:
        if not document.strip():
            continue

        tokens = tokenizer.encode(document)

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
    """Create input/target pairs for causal next-token prediction.

    For context length T, each sample contains T input tokens and T target
    tokens. The target is shifted one position ahead of the input.

    Regular windows start at positions 0, stride, 2*stride, and so on.
    If stride is omitted, it defaults to context_length.

    If include_end_chunk is True, an additional window anchored at the end
    of a document is included when its starting position was not already
    used. This can cause some token transitions to appear more than once.

    Documents shorter than T + 1 tokens are skipped without padding.
    Windows never cross document boundaries.
    """
    if context_length < 1:
        raise ValueError(
            f"context_length must be >= 1, got {context_length}."
        )

    step = stride if stride is not None else context_length

    if step < 1:
        raise ValueError(f"stride must be >= 1, got {step}.")

    samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
    t = context_length

    for tokens in token_lists:
        n_tokens = len(tokens)

        if n_tokens < t + 1:
            continue

        used_starts: Set[int] = set()
        start = 0

        # Create regular windows containing T + 1 tokens each.
        while start + t + 1 <= n_tokens:
            chunk = tokens[start : start + t + 1]

            x = torch.tensor(chunk[:t], dtype=torch.long)
            y = torch.tensor(chunk[1 : t + 1], dtype=torch.long)

            samples.append((x, y))
            used_starts.add(start)

            start += step

        # Optionally include a final window aligned with the document end.
        if include_end_chunk:
            end_start = n_tokens - (t + 1)

            if end_start not in used_starts:
                chunk = tokens[end_start : end_start + t + 1]

                x = torch.tensor(chunk[:t], dtype=torch.long)
                y = torch.tensor(chunk[1 : t + 1], dtype=torch.long)

                samples.append((x, y))
                used_starts.add(end_start)

    return samples


class GPTDataset(Dataset):
    """PyTorch Dataset that returns input/target tensor pairs."""

    def __init__(
        self,
        samples: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
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
    """Create a DataLoader yielding batches of input and target tensors.

    Each batch has shape (batch_size, context_length), except that the final
    batch can be smaller when drop_last=False.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

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
    """Load raw text rows from a Hugging Face WikiText dataset split.

    Valid splits are train, validation, and test. The dataset is loaded only
    when this function is called, not when this module is imported.

    Args:
        split: Dataset split to load.
        dataset_name: Hugging Face dataset identifier.
        config_name: Dataset configuration.
        cache_dir: Optional custom dataset cache directory.

    Returns:
        A list containing the text from each dataset row.
    """
    valid_splits = {"train", "validation", "test"}

    if split not in valid_splits:
        raise ValueError(
            f"Invalid split '{split}'. "
            f"Expected one of {sorted(valid_splits)}."
        )

    from datasets import load_dataset

    hf_dataset = load_dataset(
        dataset_name,
        config_name,
        split=split,
        cache_dir=cache_dir,
    )

    return [row["text"] for row in hf_dataset]
