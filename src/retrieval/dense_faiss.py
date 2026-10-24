"""
Dense retrieval using Contriever and FAISS inner-product (cosine similarity) index.

Supports:
1. Real Contriever mode (facebook/contriever via transformers + torch with mean pooling + L2 norm).
2. Fast deterministic 768-dim semantic hash / trigram projection embedder for CPU unit tests and demos.
3. FAISS IndexFlatIP with pure NumPy exact inner-product fallback when faiss is not installed.
"""

from typing import List, Optional, Tuple
import hashlib
import importlib
import logging
import re
import numpy as np

from src.schemas import CorpusDocument, QuerySample

logger = logging.getLogger(__name__)


def _get_faiss():
    """Dynamically load faiss if installed."""
    try:
        return importlib.import_module("faiss")
    except Exception:
        return None


def _get_transformers():
    """Dynamically load transformers if installed."""
    try:
        return importlib.import_module("transformers")
    except Exception:
        return None


def _get_torch():
    """Dynamically load torch if installed."""
    try:
        return importlib.import_module("torch")
    except Exception:
        return None


class DenseFaissRetriever:
    """Encodes documents and queries into 768-dim L2-normalized embeddings and retrieves via FAISS."""

    def __init__(
        self,
        model_name: str = "facebook/contriever",
        embedding_dim: int = 768,
        device: str = "cpu",
        use_mock_embedder: bool = False,
    ):
        """Initialize the DenseFaissRetriever.

        Args:
            model_name: HuggingFace model identifier (default: "facebook/contriever").
            embedding_dim: Dimension of embeddings (default: 768).
            device: Device for neural inference ("cpu" or "cuda").
            use_mock_embedder: If True, uses fast deterministic mock embedder (<0.5s offline).
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = device
        self.use_mock_embedder = use_mock_embedder
        self.corpus_docs: List[CorpusDocument] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self.index = None

        self._tokenizer = None
        self._model = None
        self._token_cache: dict = {}

    def _init_transformer_model(self) -> bool:
        """Attempt to load real HuggingFace tokenizer and model."""
        if self._model is not None and self._tokenizer is not None:
            return True

        transformers = _get_transformers()
        torch = _get_torch()
        if transformers is None or torch is None:
            logger.info("transformers or torch not available; falling back to deterministic mock embedder.")
            self.use_mock_embedder = True
            return False

        try:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
            self._model = transformers.AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            return True
        except Exception as e:
            logger.warning(
                "Failed to load transformer model %s (%s). Falling back to mock embedder.",
                self.model_name,
                e,
            )
            self.use_mock_embedder = True
            return False

    def _mock_encode_single(self, text: str) -> np.ndarray:
        """Deterministic 768-dim semantic hash / trigram projection embedder."""
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            tokens = ["empty"]

        vec = np.zeros(self.embedding_dim, dtype=np.float32)

        # 1. Word token projections
        for tok in tokens:
            if tok not in self._token_cache:
                seed = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16)
                rng = np.random.default_rng(seed)
                w_vec = rng.standard_normal(self.embedding_dim).astype(np.float32)
                w_norm = np.linalg.norm(w_vec)
                self._token_cache[tok] = (w_vec / max(w_norm, 1e-12)).astype(np.float32)
            vec += self._token_cache[tok]

        # 2. Subword character 3-grams for stem/inflection overlap
        lowered = text.lower()
        if len(lowered) >= 3:
            trigrams = [lowered[i : i + 3] for i in range(len(lowered) - 2)]
            tri_scale = 0.25 / max(len(trigrams), 1)
            for tri in trigrams:
                tri_key = f"__tri_{tri}"
                if tri_key not in self._token_cache:
                    seed = int(hashlib.sha256(tri_key.encode("utf-8")).hexdigest()[:8], 16)
                    rng = np.random.default_rng(seed)
                    t_vec = rng.standard_normal(self.embedding_dim).astype(np.float32)
                    t_norm = np.linalg.norm(t_vec)
                    self._token_cache[tri_key] = (t_vec / max(t_norm, 1e-12)).astype(np.float32)
                vec += tri_scale * self._token_cache[tri_key]

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = (vec / norm).astype(np.float32)
        else:
            vec = np.zeros(self.embedding_dim, dtype=np.float32)
            vec[0] = 1.0
        return vec

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts into L2-normalized dense embeddings.

        Args:
            texts: List of text strings to encode.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim) with dtype float32 and unit L2 norm.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        if not self.use_mock_embedder:
            if not self._init_transformer_model():
                pass  # Fell back to mock embedder inside _init_transformer_model

        if not self.use_mock_embedder:
            try:
                torch = _get_torch()
                with torch.no_grad():
                    inputs = self._tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    ).to(self.device)
                    outputs = self._model(**inputs)
                    token_embeddings = outputs[0]
                    mask = inputs["attention_mask"]
                    # Mean pooling masked fill
                    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
                    sum_embeddings = token_embeddings.sum(dim=1)
                    sum_mask = mask.sum(dim=1)[..., None].clamp(min=1e-9)
                    pooled = sum_embeddings / sum_mask
                    # L2-normalization
                    normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    return normalized.cpu().numpy().astype(np.float32)
            except Exception as e:
                logger.warning("Error running transformer encoder: %s. Falling back to mock embedder.", e)
                self.use_mock_embedder = True

        # Fast deterministic mock embedder
        embeddings = np.stack([self._mock_encode_single(t) for t in texts], axis=0)
        return embeddings.astype(np.float32)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Alias for encode_texts for backward compatibility."""
        return self.encode_texts(texts)

    def encode_query(self, question: str) -> np.ndarray:
        """Encode a single query string into a 1D L2-normalized embedding.

        Args:
            question: Query text string.

        Returns:
            np.ndarray of shape (embedding_dim,) with unit L2 norm.
        """
        vecs = self.encode_texts([question])
        vec = vecs[0].astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = (vec / norm).astype(np.float32)
        return vec

    def index_corpus(self, docs: List[CorpusDocument]) -> None:
        """Encode and index a corpus of documents using FAISS IndexFlatIP (or NumPy fallback).

        Args:
            docs: List of CorpusDocument instances to index.
        """
        self.corpus_docs = list(docs)
        if not docs:
            self.doc_embeddings = np.empty((0, self.embedding_dim), dtype=np.float32)
            self.index = None
            return

        texts = [f"{d.title} {d.text}".strip() for d in docs]
        self.doc_embeddings = self.encode_texts(texts)

        faiss = _get_faiss()
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(self.doc_embeddings)
        else:
            self.index = None

    def build_index(self, corpus: List[CorpusDocument]) -> None:
        """Alias for index_corpus for backward compatibility."""
        self.index_corpus(corpus)

    def search(self, query_vec: np.ndarray, top_n: int = 15) -> List[Tuple[int, float]]:
        """Search indexed corpus using query vector.

        Args:
            query_vec: 1D or 2D array of shape (embedding_dim,) or (1, embedding_dim).
            top_n: Maximum number of candidates to retrieve.

        Returns:
            List of tuples: (corpus_index, cosine_similarity_score) sorted descending by score.
        """
        if not self.corpus_docs or self.doc_embeddings is None or len(self.doc_embeddings) == 0:
            return []

        q_vec = query_vec.astype(np.float32).flatten()
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = (q_vec / norm).astype(np.float32)

        k = min(top_n, len(self.corpus_docs))
        if k <= 0:
            return []

        if self.index is not None:
            q_mat = q_vec.reshape(1, -1)
            scores, indices = self.index.search(q_mat, k)
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0:
                    results.append((int(idx), float(score)))
            return results
        else:
            # Pure NumPy exact inner product (unit vectors -> cosine similarity)
            scores = np.dot(self.doc_embeddings, q_vec)
            top_indices = np.argsort(-scores)[:k]
            return [(int(idx), float(scores[idx])) for idx in top_indices]

    def retrieve(
        self,
        query: QuerySample,
        top_n: int = 15,
    ) -> List[Tuple[CorpusDocument, float, int]]:
        """Retrieve top-N documents for query.

        Args:
            query: Input QuerySample.
            top_n: Number of candidates to retrieve.

        Returns:
            List of tuples: (CorpusDocument, dense_score, dense_rank) with 1-indexed rank.
        """
        q_vec = self.encode_query(query.question)
        hits = self.search(q_vec, top_n=top_n)
        return [
            (self.corpus_docs[idx], score, rank + 1)
            for rank, (idx, score) in enumerate(hits)
        ]
