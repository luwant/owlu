"""Ontology-Constrained Writer Module — writes candidate labels into the label ontology.

Sub-modules:
    label_bank  — Cross-document accumulation, clustering, and promotion
    constraints — Ontology constraint checking (naming format, parent existence)
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from ..common.types import (
    CandidatePhrase,
    LabelInfo,
    LtceTextSample,
    MatchResult,
    ProtoLabelCluster,
)
from .constraints import OntologyConstraintChecker
from .label_bank import LabelBank
from .persistence import LabelBankStore


class OntologyWriter:
    """Facade: ingest match results from Discovery and write approved labels
    into the label ontology with constraint checking.

    Workflow:
        1. Receive MatchResult from CandidateDiscovery
        2. Route to LabelBank (merge alias / add candidate / hold)
        3. When clusters reach promotion threshold, validate against ontology constraints
        4. Promote approved clusters as new labels

    If *db_path* is provided, every mutating operation is auto-persisted to SQLite.
    """

    def __init__(
        self,
        label_bank: LabelBank | None = None,
        constraint_checker: OntologyConstraintChecker | None = None,
        *,
        min_freq: int = 3,
        min_source_docs: int = 2,
        min_agreement: float = 0.5,
        min_semantic_distance: float = 0.3,
        dense_encoder: Callable[[str], list[float]] | None = None,
        cluster_merge_threshold: float = 0.84,
        cluster_merge_margin: float = 0.04,
        db_path: str | None = None,
    ):
        self.bank = label_bank or LabelBank(
            min_freq=min_freq,
            min_source_docs=min_source_docs,
            min_agreement=min_agreement,
            min_semantic_distance=min_semantic_distance,
            dense_encoder=dense_encoder,
            cluster_merge_threshold=cluster_merge_threshold,
            cluster_merge_margin=cluster_merge_margin,
        )
        self.constraints = constraint_checker or OntologyConstraintChecker()
        self._store: LabelBankStore | None = None
        if db_path is not None:
            self._store = LabelBankStore(db_path)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _auto_persist(self) -> None:
        """Persist to SQLite if a store is configured."""
        if self._store is not None:
            self._store.save(self.bank)

    def save(self) -> None:
        """Explicitly persist current state to SQLite.

        Raises RuntimeError if no db_path was configured.
        """
        if self._store is None:
            raise RuntimeError("No db_path configured — cannot save")
        self._store.save(self.bank)

    def load(self) -> None:
        """Restore LabelBank state from SQLite.

        Raises RuntimeError if no db_path was configured.
        """
        if self._store is None:
            raise RuntimeError("No db_path configured — cannot load")
        self.bank = self._store.load()  # type: ignore[assignment]

    @classmethod
    def from_db(
        cls,
        db_path: str,
        constraint_checker: OntologyConstraintChecker | None = None,
        dense_encoder: Callable[[str], list[float]] | None = None,
    ) -> "OntologyWriter":
        """Factory: reconstruct an OntologyWriter from a persisted database."""
        store = LabelBankStore(db_path)
        bank = store.load()
        bank.dense_encoder = dense_encoder
        writer = cls(
            label_bank=bank,
            constraint_checker=constraint_checker,
            db_path=db_path,
        )  # type: ignore[arg-type]
        return writer

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, result: MatchResult) -> str:
        """Process a single MatchResult from Discovery.

        Returns the resulting action: 'merge', 'candidate', 'hold', or 'discard'.
        """
        action = self.bank.process_match_result(result)
        self._auto_persist()
        return action

    def ingest_with_document(
        self,
        result: MatchResult,
        *,
        document_text: str,
        source_type: str = "discovery",
        split: str = "train",
    ) -> str:
        """Process a MatchResult and persist the backing document evidence.

        This is the preferred online-update path when the source document should
        later be available for slow-sync training.
        """
        action = self.bank.process_match_result(result)
        if self._store is not None:
            self._auto_persist()
            self._store.record_match_result(
                result,
                action,
                document_text=document_text,
                source_type=source_type,
                split=split,
                cluster_id=result.phrase.cluster_id,
            )
            return action

        self._auto_persist()
        return action

    def ingest_batch(self, results: Sequence[MatchResult]) -> list[str]:
        """Process a batch of MatchResults."""
        actions = [self.bank.process_match_result(r) for r in results]
        self._auto_persist()
        return actions

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
        self._auto_persist()

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
        self._auto_persist()
        if self._store is not None:
            self._store.approve_cluster_examples(cluster_id, new_label_id)
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

    def count_label_examples(
        self,
        label_id: str,
        *,
        review_status: str = "approved",
        split: str | None = None,
    ) -> int:
        """Count persisted document examples for a label."""
        if self._store is None:
            raise RuntimeError("No db_path configured — cannot count label examples")
        return self._store.count_label_examples(
            label_id,
            review_status=review_status,
            split=split,
        )

    def get_slow_sync_ready_labels(
        self,
        *,
        min_positive_examples: int = 3,
        review_status: str = "approved",
    ) -> list[str]:
        """Return promoted labels that have enough approved text evidence."""
        if self._store is None:
            raise RuntimeError("No db_path configured — cannot inspect training samples")
        promoted = set(self.bank.promoted_labels.keys())
        return [
            label_id
            for label_id in self._store.get_slow_sync_ready_labels(
                min_positive_examples=min_positive_examples,
                review_status=review_status,
            )
            if label_id in promoted
        ]

    def export_ltce_samples(
        self,
        *,
        label_ids: Sequence[str] | None = None,
        min_positive_examples: int = 1,
        review_status: str = "approved",
        promoted_only: bool = True,
    ) -> list[LtceTextSample]:
        """Export persisted text evidence as LtceTextSample objects."""
        if self._store is None:
            raise RuntimeError("No db_path configured — cannot export LTCE samples")

        selected_label_ids = list(label_ids) if label_ids is not None else None
        if promoted_only:
            promoted = set(self.bank.promoted_labels.keys())
            if selected_label_ids is None:
                selected_label_ids = sorted(promoted)
            else:
                selected_label_ids = [
                    label_id for label_id in selected_label_ids if label_id in promoted
                ]

        return self._store.export_ltce_samples(
            label_ids=selected_label_ids,
            min_positive_examples=min_positive_examples,
            review_status=review_status,
        )

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
