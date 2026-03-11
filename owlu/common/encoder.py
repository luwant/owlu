"""Shared text encoders for Discovery (SemanticMatcher) and Absorption.

Provides encoders that produce unit-norm dense vectors from text.
Two implementations:

* ``BertEncoder`` — uses a local ``transformers`` BERT model
  (mean-pooling + L2 norm).  No extra dependencies beyond
  ``transformers`` + ``torch``.
* ``SentenceTransformerEncoder`` — wraps the ``sentence-transformers``
  library (optional dependency).
"""

from __future__ import annotations

from typing import Sequence

from .types import Vector


# ======================================================================
# BertEncoder — local transformers model
# ======================================================================

class BertEncoder:
    """Encode text with a local BERT model via HuggingFace transformers.

    Uses mean-pooling over token embeddings + L2 normalisation to produce
    a fixed-size unit-norm vector (768-dim for ``bert-base-uncased``).

    Parameters
    ----------
    model_path : str
        Path (or HF hub name) of the pretrained model.
    device : str | None
        Torch device string.  ``None`` = auto-detect.
    max_length : int
        Maximum token length for the tokenizer.
    """

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        max_length: int = 128,
    ):
        self._model_path = model_path
        self._device = device
        self._max_length = max_length
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            self._model = AutoModel.from_pretrained(self._model_path)

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(self._device).eval()
        return self._model, self._tokenizer

    @property
    def dimension(self) -> int:
        model, _ = self._load()
        return model.config.hidden_size

    def encode(self, text: str) -> Vector:
        """Encode a single string → unit-norm dense vector."""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: Sequence[str]) -> list[Vector]:
        """Batch encode → list of unit-norm dense vectors."""
        import torch

        model, tokenizer = self._load()
        inputs = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pooling over tokens (respecting attention mask)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        token_embs = outputs.last_hidden_state * mask
        sentence_embs = token_embs.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # L2 normalise
        sentence_embs = torch.nn.functional.normalize(sentence_embs, p=2, dim=1)
        return [[float(v) for v in row] for row in sentence_embs.cpu()]

    def as_dense_encoder(self):
        """Return ``Callable[[str], list[float]]`` for SemanticMatcher."""
        return self.encode

    def as_text_encoder(self):
        """Return ``Callable[[str, int], Vector]`` for Absorption."""

        def _encode_with_dim(text: str, dim: int) -> Vector:
            vec = self.encode(text)
            if len(vec) != dim:
                raise ValueError(
                    f"Requested dim={dim} but encoder outputs dim={len(vec)}."
                )
            return vec

        return _encode_with_dim


# ======================================================================
# SentenceTransformerEncoder — sentence-transformers wrapper (optional)
# ======================================================================

class SentenceTransformerEncoder:
    """Lazy-loading wrapper around a sentence-transformers model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (default: ``all-MiniLM-L6-v2``,
        384-dim, fast, good general-purpose quality).
    device : str | None
        Torch device string.  ``None`` = auto-detect.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        self._model_name = model_name
        self._device = device
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(
                self._model_name, device=self._device
            )
        return self._model

    @property
    def dimension(self) -> int:
        model = self._load()
        return int(model.get_sentence_embedding_dimension())  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Core encode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> Vector:
        """Encode a single string → unit-norm dense vector."""
        model = self._load()
        vec = model.encode(text, normalize_embeddings=True)  # type: ignore[union-attr]
        return [float(v) for v in vec]

    def encode_batch(self, texts: Sequence[str]) -> list[Vector]:
        """Batch encode → list of unit-norm dense vectors."""
        model = self._load()
        matrix = model.encode(list(texts), normalize_embeddings=True)  # type: ignore[union-attr]
        return [[float(v) for v in row] for row in matrix]

    # ------------------------------------------------------------------
    # Adapter for SemanticMatcher  (str → list[float])
    # ------------------------------------------------------------------

    def as_dense_encoder(self):
        """Return ``Callable[[str], list[float]]`` for SemanticMatcher."""
        return self.encode

    # ------------------------------------------------------------------
    # Adapter for Absorption  (str, int → list[float])
    # ------------------------------------------------------------------

    def as_text_encoder(self):
        """Return ``Callable[[str, int], Vector]`` for fast/slow sync.

        The *dim* argument is validated but ignored (the model dimension
        is fixed).  Raises ``ValueError`` if the requested dim does not
        match the model's output dimension.
        """

        def _encode_with_dim(text: str, dim: int) -> Vector:
            vec = self.encode(text)
            if len(vec) != dim:
                raise ValueError(
                    f"Requested dim={dim} but encoder outputs dim={len(vec)}.  "
                    f"Ensure model_state['E'] columns == encoder dimension."
                )
            return vec

        return _encode_with_dim
