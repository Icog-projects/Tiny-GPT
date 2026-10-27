"""GPT-2 tokenizer wrapper around tiktoken.

This module provides text-to-token encoding and token-to-text decoding using
OpenAI's Byte-Pair Encoding (BPE) for GPT-2 via the `tiktoken` library.

IMPORTANT:
This module operates purely as a text tokenizer. It does NOT load, initialize,
or require pretrained GPT-2 neural network weights.
"""

from typing import List, Optional, Set
import tiktoken


class GPT2Tokenizer:
    """Wrapper around tiktoken GPT-2 BPE tokenizer.

    Attributes:
        encoding_name (str): Name of the encoding (defaults to 'gpt2').
        vocab_size (int): Size of the vocabulary (50,257 for GPT-2).
        eot_token_id (int): Integer token ID for <|endoftext|> (50,256).
    """

    EXPECTED_VOCAB_SIZE: int = 50257
    EXPECTED_EOT_ID: int = 50256

    def __init__(self, encoding_name: str = "gpt2") -> None:
        """Initializes the tokenizer and validates expected vocabulary attributes."""
        self.encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

        # Explicit validation of GPT-2 vocabulary and EOT token ID
        if self._encoding.n_vocab != self.EXPECTED_VOCAB_SIZE:
            raise ValueError(
                f"Unexpected vocabulary size for '{encoding_name}': "
                f"got {self._encoding.n_vocab}, expected {self.EXPECTED_VOCAB_SIZE}."
            )
        if self._encoding.eot_token != self.EXPECTED_EOT_ID:
            raise ValueError(
                f"Unexpected EOT token ID for '{encoding_name}': "
                f"got {self._encoding.eot_token}, expected {self.EXPECTED_EOT_ID}."
            )

    @property
    def vocab_size(self) -> int:
        """Returns the vocabulary size (50,257)."""
        return self._encoding.n_vocab

    @property
    def eot_token_id(self) -> int:
        """Returns the integer token ID of <|endoftext|> (50,256)."""
        return self._encoding.eot_token

    def encode(
        self,
        text: str,
        allowed_special: Optional[Set[str]] = None,
        disallowed_special: str | Set[str] = "all",
    ) -> List[int]:
        """Encodes a string into a list of integer token IDs.

        Args:
            text: Input string to encode.
            allowed_special: Explicit set of special token strings allowed in text.
                Defaults to None (no special tokens allowed; raises ValueError if present).
            disallowed_special: Set of special tokens to disallow or "all".
                Defaults to "all" for strict safety.

        Returns:
            List of integer token IDs in range [0, 50256].
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str input for encoding, got {type(text).__name__}.")

        allowed = allowed_special if allowed_special is not None else set()
        return self._encoding.encode(
            text,
            allowed_special=allowed,
            disallowed_special=disallowed_special,
        )

    def decode(self, token_ids: List[int]) -> str:
        """Decodes a list of integer token IDs back into a string.

        Args:
            token_ids: Sequence of token IDs to decode.

        Returns:
            Decoded string.

        Raises:
            TypeError: If token_ids is not an iterable of integers.
            ValueError: If any token ID is outside the valid vocabulary range.
        """
        if not isinstance(token_ids, list):
            token_ids = list(token_ids)

        for tid in token_ids:
            if not isinstance(tid, int):
                raise TypeError(f"Token IDs must be integers, got {type(tid).__name__}.")
            if not (0 <= tid < self.vocab_size):
                raise ValueError(
                    f"Token ID {tid} is outside valid vocabulary range [0, {self.vocab_size - 1}]."
                )

        return self._encoding.decode(token_ids)


def demonstrate_tokenizer(sentences: Optional[List[str]] = None) -> List[dict]:
    """Demonstrates tokenization on at least five distinct example sentences.

    Prints:
    - Original input sentence.
    - Resulting integer token IDs.
    - Decoded text (verifying perfect round-trip fidelity).
    - Conceptual explanation that token IDs are categorical vocabulary indices,
      NOT continuous semantic scores or numerical measures of word meaning.

    Args:
        sentences: Optional list of sentences to demonstrate. If None, uses 5 default
            representative sentences covering punctuation, grammar, numbers, and casing.

    Returns:
        List of dictionaries with demonstration metadata.
    """
    if sentences is None:
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Language models predict the next token based on context.",
            "Tokenization breaks text into subword units.",
            "Mathematics, science, and coding require precision: 1 + 1 = 2.",
            "Natural language processing is evolving rapidly!",
        ]

    tokenizer = GPT2Tokenizer()
    results = []

    print("=" * 70)
    print("GPT-2 BPE Tokenizer Demonstration (5 Sentences)")
    print("=" * 70)
    print("CONCEPTUAL FOUNDATION:")
    print("Token IDs are discrete categorical indices in a lookup vocabulary")
    print("spanning [0, 50256]. They are NOT semantic scores, sentiment values,")
    print("or continuous measures of meaning. For example, token ID 500 is not")
    print("'greater than' or 'better than' token ID 200.")
    print("=" * 70)

    for i, sent in enumerate(sentences, 1):
        token_ids = tokenizer.encode(sent)
        decoded = tokenizer.decode(token_ids)
        assert decoded == sent, f"Roundtrip failed for: {sent}"

        print(f"\n[Sentence {i}]")
        print(f"Original Text: {sent!r}")
        print(f"Token Count:   {len(token_ids)}")
        print(f"Token IDs:     {token_ids}")
        print(f"Decoded Text:  {decoded!r}")
        print(f"Roundtrip OK:  {decoded == sent}")

        results.append({
            "index": i,
            "original": sent,
            "token_ids": token_ids,
            "decoded": decoded,
            "num_tokens": len(token_ids),
        })

    print("\n" + "=" * 70)
    print("Demonstration successfully verified for all sentences.")
    print("=" * 70)
    return results


if __name__ == "__main__":
    demonstrate_tokenizer()
