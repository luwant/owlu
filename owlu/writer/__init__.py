"""Ontology-Constrained Writer Module — writes candidate labels into the label ontology.

Sub-modules:
    label_bank  — Cross-document accumulation, clustering, and promotion
    constraints — Ontology constraint checking (naming format, parent existence)
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ..common.types import (
    CandidatePhrase,
    LabelInfo,
    MatchResult,
    ProtoLabelCluster,
)
from .constraints import OntologyConstraintChecker
from .label_bank import LabelBank


class OntologyWriter:
    """Facade: ingest match results from Discovery and write approved labels
    into the label ontology with constraint checking.

    Workflow:
        1. Receive MatchResult from CandidateDiscovery
        2. Route to LabelBank (merge alias / add candidate / hold)
        3. When clusters reach promotion threshold, validate against ontology constraints
        4. Promote approved clusters as new labels
    """

    def __init__(
        self,
        label_bank: LabelBank | None = None,
        constraint_checker: OntologyConstraintChecker | None = None,
        *,
        min_freq: int = 3,
        min_source_docs: int = 2,
    ):
        self.bank = label_bank or LabelBank(
            min_freq=min_freq, min_source_docs=min_source_docs
        )
        self.constraints = constraint_checker or OntologyConstraintChecker()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, result: MatchResult) -> str:
        """Process a single MatchResult from Discovery.

        Returns the resulting action: 'merge', 'candidate', 'hold', or 'discard'.
        """
        return self.bank.process_match_result(result)

    def ingest_batch(self, results: Sequence[MatchResult]) -> list[str]:
        """Process a batch of MatchResults."""
        return [self.bank.process_match_result(r) for r in results]

    # ------------------------------------------------------------------
    # Label registration (for bootstrapping existing ontology)
    # ------------------------------------------------------------------

    def register_existing_label(
        self,
        label_id: str,
        canonical_text: str,
        aliases: list[str] | set[str] | None = None,
        description: str | None = None,
    ) -> None:
        self.bank.register_label(
            label_id=label_id,
            canonical_text=canonical_text,
            aliases=aliases,
            description=description,
        )

    # ------------------------------------------------------------------
    # Promotion with constraint checking
    # ------------------------------------------------------------------

    def get_promotion_candidates(self) -> dict[str, ProtoLabelCluster]:
        """Return clusters that meet the statistical promotion threshold."""
        return dict(self.bank.candidate_labels)

    def promote(
        self, cluster_id: str, new_label_id: str, *, skip_constraints: bool = False
    ) -> bool:
        """Promote a candidate cluster to a new label.

        When skip_constraints=False (default), the candidate must pass
        ontology constraint validation before promotion.

        Returns True if promotion succeeded, False if blocked by constraints.
        """
        cluster = self.bank.proto_label_clusters.get(cluster_id)
        if cluster is None:
            raise KeyError(f"Unknown cluster_id: {cluster_id}")

        if not skip_constraints:
            violation = self.constraints.check(
                new_label_id=new_label_id,
                representative_phrase=cluster.representative_phrase,
                existing_label_ids=set(self.bank.labels.keys()),
            )
            if violation is not None:
                return False

        self.bank.promote_cluster(cluster_id, new_label_id)
        return True

    def auto_promote_all(self, *, skip_constraints: bool = False) -> list[str]:
        """Promote all eligible candidate clusters.

        Returns list of newly promoted label IDs.
        """
        promoted: list[str] = []
        for cluster_id, cluster in list(self.bank.candidate_labels.items()):
            new_label_id = cluster.representative_phrase.replace(" ", "_")
            ok = self.promote(
                cluster_id, new_label_id, skip_constraints=skip_constraints
            )
            if ok:
                promoted.append(new_label_id)
        return promoted

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_promoted_labels(self) -> dict[str, ProtoLabelCluster]:
        return dict(self.bank.promoted_labels)

    def get_label_inventory(self) -> dict[str, str]:
        """Build {label_id: canonical_text} dict for Discovery's label inventory."""
        inventory: dict[str, str] = {}
        for label_id, info in self.bank.labels.items():
            canonical = sorted(info.aliases)[0] if info.aliases else label_id
            inventory[label_id] = canonical
        return inventory

    def build_review_packet(self, cluster_id: str) -> dict[str, object]:
        return self.bank.build_review_packet(cluster_id)

    def summarize_cluster(self, cluster_id: str) -> dict[str, object]:
        return self.bank.summarize_cluster(cluster_id)
