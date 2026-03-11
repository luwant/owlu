"""Cross-document label accumulation and promotion decisions.

Migrated from the original top-level label_bank.py — logic is unchanged.
"""

from __future__ import annotations

import re
import unicodedata

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
    ):
        self.min_freq = int(min_freq)
        self.min_source_docs = int(min_source_docs)
        self.min_agreement = float(min_agreement)
        self.min_semantic_distance = float(min_semantic_distance)

        self.labels: dict[str, LabelInfo] = {}
        self.proto_label_clusters: dict[str, ProtoLabelCluster] = {}
        self.candidate_labels: dict[str, ProtoLabelCluster] = {}
        self.hold_pool: dict[str, ProtoLabelCluster] = {}
        self.promoted_labels: dict[str, ProtoLabelCluster] = {}

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

    def _cluster_id(self, phrase_text: str) -> str:
        return self._normalize_phrase(phrase_text)

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
        cid = cluster_id or self._cluster_id(phrase.text)
        normalized_phrase = self._normalize_phrase(phrase.text)

        cluster = self.proto_label_clusters.get(cid)
        if cluster is None:
            cluster = ProtoLabelCluster(
                cluster_id=cid, representative_phrase=normalized_phrase
            )
            self.proto_label_clusters[cid] = cluster

        cluster.freq += 1
        cluster.agreement_sum += float(phrase.agreement)
        cluster.agreement_count += 1
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
                key=lambda p: (cluster.phrases[p], -len(p), p),
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
