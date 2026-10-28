"""Offline unit tests for Task 1: Tokenization, preprocessing, and context windowing.

All unit tests in this module run strictly on synthetic data and do NOT require
an internet connection or any dataset downloading.
"""

from unittest.mock import patch
import pytest
import torch
from src.data.dataset import (
    GPTDataset,
    build_chunks_from_token_lists,
    create_dataloader,
    extract_documents_from_raw_lines,
    is_article_header,
    load_wikitext_split,
    tokenize_documents,
)
from src.data.tokenizer import GPT2Tokenizer, demonstrate_tokenizer


# ---------------------------------------------------------------------------
# 1. Tokenizer Unit Tests
# ---------------------------------------------------------------------------

def test_tokenizer_vocab_size_and_eot_id():
    """Verifies that GPT-2 tokenizer has exactly 50,257 tokens and EOT is 50,256."""
    tokenizer = GPT2Tokenizer()
    assert tokenizer.vocab_size == 50257
    assert tokenizer.eot_token_id == 50256


def test_tokenizer_encode_decode_roundtrip():
    """Verifies that encoding normal text produces valid token IDs and decodes faithfully."""
    tokenizer = GPT2Tokenizer()
    sample_text = "From ChatGPT to a Tiny GPT: How Language Models Work."
    token_ids = tokenizer.encode(sample_text)

    # Check return type and contents
    assert isinstance(token_ids, list)
    assert len(token_ids) > 0
    assert all(isinstance(tid, int) for tid in token_ids)
    assert all(0 <= tid < 50257 for tid in token_ids)

    # Roundtrip check
    decoded_text = tokenizer.decode(token_ids)
    assert decoded_text == sample_text


def test_tokenizer_invalid_input_handling():
    """Verifies that tokenizer rejects invalid inputs with clear type or range errors."""
    tokenizer = GPT2Tokenizer()

    # Non-string input to encode
    with pytest.raises(TypeError):
        tokenizer.encode(12345)  # type: ignore

    # Out-of-bounds token ID in decode
    with pytest.raises(ValueError):
        tokenizer.decode([50257])  # 50257 is out of range [0, 50256]

    with pytest.raises(ValueError):
        tokenizer.decode([-1])


def test_tokenizer_encode_disallowed_special_by_default():
    """Verifies that tokenizer.encode raises ValueError when text contains special tokens by default."""
    tokenizer = GPT2Tokenizer()
    with pytest.raises(ValueError, match="disallowed special token"):
        tokenizer.encode("<|endoftext|>")


# ---------------------------------------------------------------------------
# 2. Document Boundary & Heuristic Tests
# ---------------------------------------------------------------------------

def test_is_article_header_heuristic():
    """Tests the top-level article header heuristic against representative synthetic lines."""
    # Valid top-level Wikipedia article headers in WikiText
    assert is_article_header(" = Homura Akemi = \n") is True
    assert is_article_header(" = Valkyria Chronicles III = ") is True
    assert is_article_header("= Simple Title =") is True

    # Subsection headers (must NOT be treated as top-level article boundaries)
    assert is_article_header(" = = Overview = = \n") is False
    assert is_article_header(" = = = Early Life = = = ") is False
    assert is_article_header("==== Deep Subsection ====") is False

    # Plain text and sentences containing equals signs
    assert is_article_header("This is a normal paragraph sentence.") is False
    assert is_article_header("Formula: x = y + z = 10.") is False
    assert is_article_header("") is False
    assert is_article_header("   \n") is False


def test_table_column_labels_are_not_article_headers():
    """Verifies that table-column labels with semicolons are not treated as article headers."""
    table_column_labels = [
        "= Position ; GP =",
        "= Goals ; A =",
        "= Points ; PIM =",
        "= Games Played ; TOI =",
        "= Wins ; L =",
        "= Overtime Losses ; GA =",
        "= Saves ; Sv % =",
    ]
    for label in table_column_labels:
        assert is_article_header(label) is False

    assert is_article_header("= Valkyria Chronicles III =") is True
    assert is_article_header("= Plain maskray =") is True

    assert is_article_header("= = Gameplay = =") is False


def test_extract_documents_from_raw_lines():
    """Tests that raw lines are grouped into distinct documents without splitting on blank lines."""
    synthetic_rows = [
        " = Article One = \n",
        "This is the first paragraph of article one.\n",
        "\n",  # Blank line should not start a new document
        " = = Section 1 = = \n",
        "Section 1 details under article one.\n",
        "\n",
        " = Article Two = \n",
        "Article two opening sentence.\n",
        "Additional article two sentence.\n",
    ]

    docs = extract_documents_from_raw_lines(synthetic_rows)
    assert len(docs) == 2

    # Verify Article 1 preserves its title, content, and subsections
    assert docs[0].startswith("= Article One =")
    assert "first paragraph of article one." in docs[0]
    assert "= = Section 1 = =" in docs[0]
    assert "Section 1 details" in docs[0]

    # Verify Article 2 is separated
    assert docs[1].startswith("= Article Two =")
    assert "Article two opening sentence." in docs[1]


def test_extract_documents_without_headers_fallback():
    """Tests that if no lines match the header heuristic, content is preserved as one document."""
    rows = [
        "First line of unstructured text.\n",
        "\n",
        "Second line of unstructured text.\n",
    ]
    docs = extract_documents_from_raw_lines(rows)
    assert len(docs) == 1
    assert "First line" in docs[0]
    assert "Second line" in docs[0]


def test_extract_documents_preamble_preservation():
    """Verifies that non-empty text before the first recognized article heading is preserved."""
    raw_lines = [
        "This is preamble text before any header.\n",
        "Second line of preamble.\n",
        " = First Official Article = \n",
        "Body of the first article.\n",
    ]
    docs = extract_documents_from_raw_lines(raw_lines)
    assert len(docs) == 2
    assert docs[0] == "This is preamble text before any header.\nSecond line of preamble."
    assert docs[1] == "= First Official Article =\nBody of the first article."


def test_extract_documents_consecutive_headings():
    """Verifies that consecutive article headings are neither discarded nor merged,
    and documents explicitly how a heading-only document is handled when it is too short to produce a training window.
    """
    raw_lines = [
        " = Article One = \n",
        " = Article Two = \n",
        "Body paragraph of article two.\n",
    ]
    docs = extract_documents_from_raw_lines(raw_lines)
    assert len(docs) == 2
    assert docs[0] == "= Article One ="
    assert docs[1] == "= Article Two =\nBody paragraph of article two."

    # Heading-only document (docs[0]) has fewer than T + 1 tokens (e.g. for T = 8).
    # Verify it produces zero samples without padding.
    tokenizer = GPT2Tokenizer()
    token_lists = tokenize_documents([docs[0]], tokenizer, append_eot=True)
    chunks = build_chunks_from_token_lists(token_lists, context_length=8)
    assert len(chunks) == 0


# ---------------------------------------------------------------------------
# 3. Document Tokenization & EOT Appending Tests
# ---------------------------------------------------------------------------

def test_tokenize_documents_appends_eot_exactly_once():
    """Verifies that tokenize_documents appends EOT (50256) once at the end of each document."""
    tokenizer = GPT2Tokenizer()
    docs = ["First document text.", "Second document text."]

    token_lists = tokenize_documents(docs, tokenizer, append_eot=True)
    assert len(token_lists) == 2

    for tokens in token_lists:
        assert len(tokens) > 1
        # The very last token must be EOT (50256)
        assert tokens[-1] == tokenizer.eot_token_id
        # EOT should only appear at the end, not duplicated in the body
        assert tokens.count(tokenizer.eot_token_id) == 1


# ---------------------------------------------------------------------------
# 4. Context Windowing & Next-Token Invariants (Pure Synthetic)
# ---------------------------------------------------------------------------

def test_exact_window_slicing_and_target_alignment():
    """Verifies exact start positions, (T + 1) chunking, and 1-token shift."""
    context_length = 4
    # Synthetic tokens: 11 tokens [100, 101, ..., 110]
    synthetic_tokens = list(range(100, 111))

    # Stride = 4, include_end_chunk=False
    # Valid starts: s0 = 0 (chunk 0..5), s1 = 4 (chunk 4..9)
    # Next start s2 = 8: 8 + 5 = 13 > 11 (tail dropped)
    chunks = build_chunks_from_token_lists(
        [synthetic_tokens],
        context_length=context_length,
        stride=context_length,
        include_end_chunk=False,
    )
    assert len(chunks) == 2

    # Sample 0 check
    x0, y0 = chunks[0]
    assert x0.shape == (context_length,)
    assert y0.shape == (context_length,)
    assert x0.dtype == torch.long
    assert y0.dtype == torch.long
    assert x0.tolist() == [100, 101, 102, 103]
    assert y0.tolist() == [101, 102, 103, 104]
    # Invariant: y[i] == x[i+1]
    assert torch.equal(y0[:-1], x0[1:])
    assert y0[-1].item() == 104

    # Sample 1 check
    x1, y1 = chunks[1]
    assert x1.tolist() == [104, 105, 106, 107]
    assert y1.tolist() == [105, 106, 107, 108]
    assert torch.equal(y1[:-1], x1[1:])
    assert y1[-1].item() == 108


def test_end_anchored_window_and_deduplication():
    """Verifies end-anchored window correctly captures EOT and avoids duplicate starts."""
    context_length = 4  # T = 4, chunk length = 5
    eot_id = 50256

    # Case A: Document of 7 tokens [100, 101, 102, 103, 104, 105, 50256]
    # Regular start: s0 = 0 (chunk 0..5, tokens [100..104])
    # Next regular start s = 4 is invalid (4 + 5 = 9 > 7)
    # End-anchored start: s_end = 7 - 5 = 2 (not in {0}) -> chunk [102, 103, 104, 105, 50256]
    tokens_a = [100, 101, 102, 103, 104, 105, eot_id]
    chunks_a = build_chunks_from_token_lists(
        [tokens_a],
        context_length=context_length,
        stride=context_length,
        include_end_chunk=True,
    )
    assert len(chunks_a) == 2
    # The end-anchored sample must end with EOT in its target
    x_end, y_end = chunks_a[-1]
    assert x_end.tolist() == [102, 103, 104, 105]
    assert y_end.tolist() == [103, 104, 105, eot_id]
    assert y_end[-1].item() == eot_id

    # Case B: Document of exactly 5 tokens (T + 1)
    # Regular start: s0 = 0.
    # End start: s_end = 5 - 5 = 0.
    # Since 0 is already in used_starts, it must be deduplicated (len == 1, not 2).
    tokens_b = [200, 201, 202, 203, eot_id]
    chunks_b = build_chunks_from_token_lists(
        [tokens_b],
        context_length=context_length,
        stride=context_length,
        include_end_chunk=True,
    )
    assert len(chunks_b) == 1
    assert chunks_b[0][1][-1].item() == eot_id


def test_end_window_deduplication_at_2t_plus_1():
    """For context length T, verifies that when document length M = 2T + 1,
    the end-anchored window is deduplicated because s_end = (2T + 1) - (T + 1) = T,
    which is already covered by the second regular stride start (s1 = T).
    """
    T = 4
    M = 2 * T + 1  # 9
    tokens = list(range(100, 100 + M))
    # Regular starts: s0 = 0 (chunk 0..5), s1 = 4 (chunk 4..9)
    # End window start: s_end = 9 - 5 = 4
    # Since 4 is already in used starts, deduplication ensures only 2 chunks are produced.
    chunks = build_chunks_from_token_lists([tokens], context_length=T, stride=T, include_end_chunk=True)
    assert len(chunks) == 2
    starts = [c[0][0].item() - 100 for c in chunks]
    assert starts == [0, 4]
    # Verify the target of the final chunk ends with index M - 1 (token 108)
    assert chunks[1][1][-1].item() == 100 + M - 1


def test_document_boundary_isolation():
    """Asserts that no context window combines tokens across distinct documents."""
    context_length = 3  # Chunk length = 4
    doc_a = [10, 11, 12, 13]  # Exactly 1 chunk
    doc_b = [20, 21, 22, 23]  # Exactly 1 chunk

    chunks = build_chunks_from_token_lists(
        [doc_a, doc_b],
        context_length=context_length,
        stride=context_length,
    )
    assert len(chunks) == 2

    x0, y0 = chunks[0]
    x1, y1 = chunks[1]

    # Verify document A tokens are strictly isolated
    sample_a_tokens = x0.tolist() + [y0[-1].item()]
    sample_b_tokens = x1.tolist() + [y1[-1].item()]

    assert sample_a_tokens == doc_a
    assert sample_b_tokens == doc_b
    assert not any(t in sample_b_tokens for t in sample_a_tokens)


def test_short_and_empty_documents_produce_no_samples():
    """Verifies that sequences shorter than T + 1 produce zero chunks without padding."""
    context_length = 8  # Needs 9 tokens
    short_doc = [1, 2, 3, 4]  # 4 < 9
    empty_doc: list[int] = []

    chunks = build_chunks_from_token_lists(
        [short_doc, empty_doc],
        context_length=context_length,
    )
    assert len(chunks) == 0


def test_short_window_boundary_exact_length_t():
    """For context length T, verifies that a document of exactly T tokens produces zero samples
    because every sample strictly requires T + 1 consecutive tokens.
    """
    T = 8
    tokens_exactly_t = list(range(100, 100 + T))
    assert len(tokens_exactly_t) == T
    chunks = build_chunks_from_token_lists([tokens_exactly_t], context_length=T, include_end_chunk=True)
    assert len(chunks) == 0


# ---------------------------------------------------------------------------
# 5. PyTorch Dataset & DataLoader Batching Tests
# ---------------------------------------------------------------------------

def test_dataset_and_dataloader_batch_shapes_and_dtypes():
    """Verifies that GPTDataset and DataLoader output batches of shape (B, T) with torch.long."""
    T = 16
    B = 4
    num_samples = 10

    # Create synthetic samples of exact shape (T,)
    synthetic_samples = [
        (
            torch.arange(i, i + T, dtype=torch.long),
            torch.arange(i + 1, i + 1 + T, dtype=torch.long),
        )
        for i in range(num_samples)
    ]

    dataset = GPTDataset(synthetic_samples)
    assert len(dataset) == num_samples

    loader = create_dataloader(dataset, batch_size=B, shuffle=False)
    batch_x, batch_y = next(iter(loader))

    assert batch_x.shape == (B, T)
    assert batch_y.shape == (B, T)
    assert batch_x.dtype == torch.long
    assert batch_y.dtype == torch.long
    # Shift check across the entire batch
    assert torch.equal(batch_y[:, :-1], batch_x[:, 1:])


# ---------------------------------------------------------------------------
# 6. Split Separation & Loader Validation Tests
# ---------------------------------------------------------------------------

def test_load_wikitext_split_validates_split_names():
    """Verifies that load_wikitext_split rejects invalid split names to prevent data leakage."""
    with pytest.raises(ValueError, match="Invalid split 'train_val'"):
        load_wikitext_split("train_val")

    with pytest.raises(ValueError, match="Invalid split 'dev'"):
        load_wikitext_split("dev")


def test_load_wikitext_split_isolation():
    """Verifies that load_wikitext_split requests the exact official split without cross-mixing."""
    with patch("datasets.load_dataset") as mock_load:
        mock_load.return_value = [{"text": "Sample text line"}]

        # Test train split
        train_rows = load_wikitext_split("train")
        mock_load.assert_called_with(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            split="train",
            cache_dir=None,
        )
        assert train_rows == ["Sample text line"]

        # Test validation split
        val_rows = load_wikitext_split("validation")
        mock_load.assert_called_with(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            split="validation",
            cache_dir=None,
        )
        assert val_rows == ["Sample text line"]


def test_demonstrate_tokenizer_five_sentences():
    """Verifies that demonstrate_tokenizer covers at least 5 sentences with correct roundtrips."""
    results = demonstrate_tokenizer()
    assert len(results) >= 5
    for item in results:
        assert "original" in item
        assert "token_ids" in item
        assert "decoded" in item
        assert item["decoded"] == item["original"]
        assert all(isinstance(tid, int) for tid in item["token_ids"])
        assert all(0 <= tid < 50257 for tid in item["token_ids"])
