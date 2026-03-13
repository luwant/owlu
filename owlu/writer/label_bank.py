"""Cross-document label accumulation and promotion decisions.

Migrated from the original top-level label_bank.py — logic is unchanged.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Callable

from ..common.types import (
    CandidatePhrase,
    ClusterState,
    LabelInfo,
    MatchResult,
    ProtoLabelCluster,
)


class LabelBank:
    """In-memory label bank for cross-document accumulation and audit packaging."""

    def __init__(
        self,
        min_freq: int = 3,
        min_source_docs: int = 2,
        min_agreement: float = 0.5,
        min_semantic_distance: float = 0.3,
        dense_encoder: Callable[[str], list[float]] | None = None,
        cluster_merge_threshold: float = 0.84,
        cluster_merge_margin: float = 0.04,
    ):
        self.min_freq = int(min_freq)
        self.min_source_docs = int(min_source_docs)
        self.min_agreement = float(min_agreement)
        self.min_semantic_distance = float(min_semantic_distance)
        self.dense_encoder = dense_encoder
        self.cluster_merge_threshold = float(cluster_merge_threshold)
        self.cluster_merge_margin = float(cluster_merge_margin)

        self.labels: dict[str, LabelInfo] = {}
        self.proto_label_clusters: dict[str, ProtoLabelCluster] = {}
        self.candidate_labels: dict[str, ProtoLabelCluster] = {}
        self.hold_pool: dict[str, ProtoLabelCluster] = {}
        self.promoted_labels: dict[str, ProtoLabelCluster] = {}
        self._phrase_embedding_cache: dict[str, list[float]] = {}
        self._next_cluster_index: int = 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_phrase(self, text: str) -> str:
        """Lowercase → remove punctuation → collapse whitespace → merge single chars.

        Examples:
            "Deep-Learning" → "deep learning"
            "N.L.P."        → "nlp"
        """
        lowered = (text or "").lower()
        # Remove all punctuation characters (Unicode category P)
        depuncted = re.sub(
            r"[^\w\s]",
            " ",
            unicodedata.normalize("NFKD", lowered),
        )
        tokens = depuncted.split()
        # Merge runs of single-character tokens (e.g. ["n","l","p"] → ["nlp"])
        merged: list[str] = []
        buf: list[str] = []
        for tok in tokens:
            if len(tok) == 1 and tok.isalnum():
                buf.append(tok)
            else:
                if buf:
                    merged.append("".join(buf))
                    buf = []
                merged.append(tok)
        if buf:
            merged.append("".join(buf))
        return " ".join(merged)

    def _naive_lemmatize(self, token: str) -> str:
        if len(token) > 4 and token.endswith("ing"):
            return token[:-3]
        if len(token) > 3 and token.endswith("ed"):
            return token[:-2]
        if len(token) > 3 and token.endswith("es"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    def _fallback_dense_encode(self, text: str, dim: int = 64) -> list[float]:
        vec = [0.0] * dim
        for token in self._normalize_phrase(text).split():
            lemma = self._naive_lemmatize(token)
            if not lemma:
                continue
            slot = hash(lemma) % dim
            vec[slot] += 1.0
        return vec

    def _encode_phrase(self, text: str) -> list[float]:
        normalized = self._normalize_phrase(text)
        cached = self._phrase_embedding_cache.get(normalized)
        if cached is not None:
            return list(cached)
        if self.dense_encoder is not None:
            embedding = [float(v) for v in self.dense_encoder(normalized)]
        else:
            embedding = self._fallback_dense_encode(normalized)
        self._phrase_embedding_cache[normalized] = list(embedding)
        return embedding

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(v * v for v in left))
        right_norm = math.sqrt(sum(v * v for v in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)

    def _allocate_cluster_id(self) -> str:
        cluster_id = f"cluster_{self._next_cluster_index:06d}"
        self._next_cluster_index += 1
        return cluster_id

    def _refresh_cluster_counter(self) -> None:
        max_index = 0
        for cluster_id in self.proto_label_clusters:
            if cluster_id.startswith("cluster_"):
                suffix = cluster_id[len("cluster_"):]
                if suffix.isdigit():
                    max_index = max(max_index, int(suffix))
        for cluster in self.promoted_labels.values():
            if cluster.cluster_id.startswith("cluster_"):
                suffix = cluster.cluster_id[len("cluster_"):]
                if suffix.isdigit():
                    max_index = max(max_index, int(suffix))
        self._next_cluster_index = max(self._next_cluster_index, max_index + 1)

    def _find_semantic_cluster(
        self, embedding: list[float]
    ) -> tuple[str | None, float]:
        scored: list[tuple[float, str]] = []
        for cluster_id, cluster in self.proto_label_clusters.items():
            if cluster.centroid_embedding is None:
                continue
            sim = self._cosine_similarity(embedding, cluster.centroid_embedding)
            scored.append((sim, cluster_id))

        if not scored:
            return None, 0.0

        scored.sort(reverse=True)
        best_sim, best_cluster_id = scored[0]
        second_best = scored[1][0] if len(scored) > 1 else -1.0
        if (
            best_sim >= self.cluster_merge_threshold
            and (best_sim - second_best) >= self.cluster_merge_margin
        ):
            return best_cluster_id, best_sim
        return None, best_sim

    def _resolve_cluster(
        self,
        phrase: CandidatePhrase,
        embedding: list[float],
    ) -> tuple[str, float]:
        matched_cluster_id, matched_similarity = self._find_semantic_cluster(embedding)
        if matched_cluster_id is not None:
            return matched_cluster_id, matched_similarity
        return self._allocate_cluster_id(), 0.0

    def _update_cluster_centroid(
        self,
        cluster: ProtoLabelCluster,
        embedding: list[float],
    ) -> None:
        if cluster.centroid_embedding is None:
            cluster.centroid_embedding = list(embedding)
            return

        prev_freq = max(cluster.freq - 1, 0)
        if prev_freq == 0:
            cluster.centroid_embedding = list(embedding)
            return

        updated: list[float] = []
        for old_value, new_value in zip(cluster.centroid_embedding, embedding):
            updated.append((old_value * prev_freq + new_value) / float(prev_freq + 1))
        cluster.centroid_embedding = updated

    def _representative_key(self, cluster: ProtoLabelCluster, phrase_text: str) -> tuple[float, int, str]:
        count = cluster.phrases.get(phrase_text, 0)
        if cluster.centroid_embedding is None:
            similarity = 0.0
        else:
            similarity = self._cosine_similarity(
                self._encode_phrase(phrase_text),
                cluster.centroid_embedding,
            )
        return (float(count) + similarity, -len(phrase_text), phrase_text)

    def _should_promote(self, cluster: ProtoLabelCluster) -> bool:
        """Four-gate promotion check (Dawid & Skene + X-MLClass).

        1. freq           >= min_freq            (annotation volume)
        2. source_docs    >= min_source_docs      (annotator independence)
        3. agreement      >= min_agreement         (annotator consistency)
        4. semantic_dist  >= min_semantic_distance  (label-space novelty)
        """
        if cluster.freq < self.min_freq:
            return False
        if cluster.source_doc_count < self.min_source_docs:
            return False
        if cluster.agreement < self.min_agreement:
            return False
        effective_distance = (
            cluster.nearest_label_distance
            if cluster.nearest_label_distance is not None
            else 1.0
        )
        if effective_distance < self.min_semantic_distance:
            return False
        return True

    # ------------------------------------------------------------------
    # Label registration
    # ------------------------------------------------------------------

    def register_label(
        self,
        label_id: str,
        canonical_text: str,
        aliases: list[str] | set[str] | None = None,
        description: str | None = None,
    ) -> None:
        normalized_canonical = self._normalize_phrase(canonical_text)
        alias_set = {normalized_canonical} if normalized_canonical else set()
        if aliases:
            alias_set.update(self._normalize_phrase(a) for a in aliases if a)

        info = self.labels.get(label_id)
        if info is None:
            self.labels[label_id] = LabelInfo(
                label_id=label_id,
                aliases=alias_set,
                description=(description or "").strip(),
            )
            return

        info.aliases.update(alias_set)
        if description:
            info.description = description.strip()

    def add_alias(
        self, label_id: str, phrase: str, description: str | None = None
    ) -> None:
        if label_id not in self.labels:
            self.register_label(
                label_id=label_id, canonical_text=phrase, description=description
            )
            return

        normalized = self._normalize_phrase(phrase)
        if normalized:
            self.labels[label_id].aliases.add(normalized)
        if description:
            desc = description.strip()
            if desc and desc not in self.labels[label_id].description:
                if self.labels[label_id].description:
                    self.labels[label_id].description += " "
                self.labels[label_id].description += desc

    def get_label_aliases(self, label_id: str) -> list[str]:
        info = self.labels.get(label_id)
        if info is None:
            return []
        return sorted(info.aliases)

    def get_label_description(self, label_id: str) -> str:
        info = self.labels.get(label_id)
        return "" if info is None else info.description

    # ------------------------------------------------------------------
    # Cluster management
    # ------------------------------------------------------------------

    def _upsert_cluster(
        self,
        phrase: CandidatePhrase,
        *,
        cluster_id: str | None = None,
        nearest_label_id: str | None = None,
        nearest_label_distance: float | None = None,
    ) -> ProtoLabelCluster:
        normalized_phrase = self._normalize_phrase(phrase.text)
        phrase_embedding = self._encode_phrase(normalized_phrase)
        cid = cluster_id
        semantic_similarity = 0.0
        if cid is None:
            cid, semantic_similarity = self._resolve_cluster(phrase, phrase_embedding)
        phrase.cluster_id = cid

        cluster = self.proto_label_clusters.get(cid)
        if cluster is None:
            cluster = ProtoLabelCluster(
                cluster_id=cid,
                representative_phrase=normalized_phrase,
                centroid_embedding=list(phrase_embedding),
            )
            self.proto_label_clusters[cid] = cluster

        cluster.freq += 1
        cluster.agreement_sum += float(phrase.agreement)
        cluster.agreement_count += 1
        self._update_cluster_centroid(cluster, phrase_embedding)
        if phrase.source_doc_id:
            cluster.source_docs.add(phrase.source_doc_id)
            cluster.evidence_docs.add(phrase.source_doc_id)

        if normalized_phrase:
            cluster.phrases[normalized_phrase] = (
                cluster.phrases.get(normalized_phrase, 0) + 1
            )

        if cluster.phrases:
            cluster.representative_phrase = max(
                cluster.phrases,
                key=lambda p: self._representative_key(cluster, p),
            )

        if nearest_label_id:
            cluster.nearest_label_id = nearest_label_id
        if nearest_label_distance is not None:
            if cluster.nearest_label_distance is None:
                cluster.nearest_label_distance = float(nearest_label_distance)
            else:
                cluster.nearest_label_distance = min(
                    cluster.nearest_label_distance, float(nearest_label_distance)
                )
        elif semantic_similarity > 0.0:
            semantic_distance = 1.0 - float(semantic_similarity)
            if cluster.nearest_label_distance is None:
                cluster.nearest_label_distance = semantic_distance

        return cluster

    def add_candidate(
        self,
        phrase: CandidatePhrase,
        *,
        cluster_id: str | None = None,
        nearest_label_id: str | None = None,
        nearest_label_distance: float | None = None,
    ) -> str:
        cluster = self._upsert_cluster(
            phrase,
            cluster_id=cluster_id,
            nearest_label_id=nearest_label_id,
            nearest_label_distance=nearest_label_distance,
        )

        if self._should_promote(cluster):
            cluster.state = "candidate"
            self.candidate_labels[cluster.cluster_id] = cluster
            self.hold_pool.pop(cluster.cluster_id, None)
            return "candidate"

        cluster.state = "hold"
        self.hold_pool[cluster.cluster_id] = cluster
        self.candidate_labels.pop(cluster.cluster_id, None)
        return "hold"

    def add_hold(
        self,
        phrase: CandidatePhrase,
        *,
        cluster_id: str | None = None,
        nearest_label_id: str | None = None,
        nearest_label_distance: float | None = None,
    ) -> str:
        cluster = self._upsert_cluster(
            phrase,
            cluster_id=cluster_id,
            nearest_label_id=nearest_label_id,
            nearest_label_distance=nearest_label_distance,
        )
        cluster.state = "hold"
        self.hold_pool[cluster.cluster_id] = cluster
        self.candidate_labels.pop(cluster.cluster_id, None)
        return "hold"

    # ------------------------------------------------------------------
    # Match result routing
    # ------------------------------------------------------------------

    def process_match_result(self, result: MatchResult) -> str:
        phrase = result.phrase
        phrase.cluster_id = None
        if result.action == "merge_pre" and result.target_label:
            self.add_alias(
                label_id=result.target_label,
                phrase=phrase.text,
                description=phrase.summary,
            )
            return "merge"

        nearest_distance = 1.0 - float(result.similarity)
        if result.action == "novel_pre":
            return self.add_candidate(
                phrase,
                nearest_label_id=result.target_label,
                nearest_label_distance=nearest_distance,
            )
        if result.action == "hold_pre":
            return self.add_hold(
                phrase,
                nearest_label_id=result.target_label,
                nearest_label_distance=nearest_distance,
            )
        return "discard"

    # ------------------------------------------------------------------
    # Query & review
    # ------------------------------------------------------------------

    def get_hold_cluster(self, cluster_id: str) -> ProtoLabelCluster | None:
        return self.hold_pool.get(cluster_id)

    def summarize_cluster(self, cluster_id: str) -> dict[str, object]:
        cluster = self.proto_label_clusters[cluster_id]
        return {
            "cluster_id": cluster.cluster_id,
            "state": cluster.state,
            "representative_phrase": cluster.representative_phrase,
            "freq": cluster.freq,
            "source_docs": cluster.source_doc_count,
            "agreement": cluster.agreement,
            "evidence_docs": sorted(cluster.evidence_docs),
            "nearest_label_id": cluster.nearest_label_id,
            "nearest_label_distance": cluster.nearest_label_distance,
        }

    def build_review_packet(self, cluster_id: str) -> dict[str, object]:
        summary = self.summarize_cluster(cluster_id)
        return {
            "representative_phrase": summary["representative_phrase"],
            "evidence_docs": summary["evidence_docs"],
            "nearest_label_distance": summary["nearest_label_distance"],
            "freq": summary["freq"],
            "source_docs": summary["source_docs"],
            "agreement": summary["agreement"],
        }

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def promote_cluster(self, cluster_id: str, new_label_id: str) -> None:
        cluster = self.proto_label_clusters.get(cluster_id)
        if cluster is None:
            raise KeyError(f"Unknown cluster_id: {cluster_id}")
        if cluster.state != "candidate":
            raise ValueError("Only candidate clusters can be promoted")

        cluster.state = "promoted"
        self.promoted_labels[new_label_id] = cluster

        aliases = sorted(cluster.phrases.keys())
        canonical = (
            cluster.representative_phrase
            if cluster.representative_phrase
            else new_label_id
        )
        self.register_label(
            label_id=new_label_id,
            canonical_text=canonical,
            aliases=aliases,
            description=f"Promoted from cluster {cluster_id}",
        )

        self.candidate_labels.pop(cluster_id, None)
        self.hold_pool.pop(cluster_id, None)
        self.proto_label_clusters.pop(cluster_id, None)
