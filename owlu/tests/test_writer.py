"""Comprehensive unit tests for the OWLU Writer module.

Covers:
  - _normalize_phrase (punctuation, abbreviations, hyphens)
  - Four-gate promotion (freq, source_docs, agreement, semantic_distance)
  - MatchResult routing (merge / novel / hold / discard)
  - Cluster accumulation and representative phrase voting
  - Ontology constraint checking
  - SQLite persistence round-trip
  - Full pipeline data-flow demonstration
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

import pytest

from owlu.common.types import (
    CandidatePhrase,
    LtceTextSample,
    MatchResult,
    ProtoLabelCluster,
    OWLUConfig,
)
from owlu.writer.label_bank import LabelBank
from owlu.writer.constraints import OntologyConstraintChecker
from owlu.writer import OntologyWriter
from owlu.writer.persistence import LabelBankStore


# =====================================================================
# Helpers
# =====================================================================

def _phrase(text: str, doc_id: str = "doc_1", agreement: float = 1.0) -> CandidatePhrase:
    return CandidatePhrase(
        text=text,
        raw_text=text,
        source_doc_id=doc_id,
        timestamp=datetime.now(timezone.utc),
        agreement=agreement,
    )


def _match(
    text: str,
    action: str,
    doc_id: str = "doc_1",
    target_label: str | None = None,
    similarity: float = 0.0,
    agreement: float = 1.0,
) -> MatchResult:
    phrase = _phrase(text, doc_id=doc_id, agreement=agreement)
    return MatchResult(
        phrase=phrase,
        action=action,
        target_label=target_label,
        similarity=similarity,
        decision_reason="test",
        normalized_phrase=text.lower(),
    )


def _test_dense_encoder(text: str) -> list[float]:
    normalized = " ".join(str(text).lower().replace("-", " ").split())
    groups = {
        "graph neural network": [1.0, 0.0, 0.0, 0.0],
        "graph neural networks": [0.99, 0.01, 0.0, 0.0],
        "graph based neural network": [0.98, 0.02, 0.0, 0.0],
        "deep learning": [0.0, 1.0, 0.0, 0.0],
        "deep neural learning": [0.02, 0.98, 0.0, 0.0],
        "quantum computing": [0.0, 0.0, 1.0, 0.0],
        "bioinformatics": [0.0, 0.0, 0.0, 1.0],
    }
    return groups.get(normalized, [0.2, 0.2, 0.2, 0.2])


# =====================================================================
# 1. _normalize_phrase
# =====================================================================

class TestNormalizePhrase:
    def setup_method(self):
        self.bank = LabelBank()

    def test_basic_lowercase_and_whitespace(self):
        assert self.bank._normalize_phrase("  Machine   Learning  ") == "machine learning"

    def test_hyphen_removal(self):
        assert self.bank._normalize_phrase("Deep-Learning") == "deep learning"

    def test_abbreviation_dots_merged(self):
        """N.L.P. → nlp: dots removed, single chars merged."""
        assert self.bank._normalize_phrase("N.L.P.") == "nlp"

    def test_abbreviation_with_following_word(self):
        assert self.bank._normalize_phrase("U.S.A. policy") == "usa policy"

    def test_empty_string(self):
        assert self.bank._normalize_phrase("") == ""

    def test_none_input(self):
        assert self.bank._normalize_phrase(None) == ""

    def test_mixed_punctuation(self):
        """Parentheses, commas, colons should all be stripped."""
        result = self.bank._normalize_phrase("(multi-label) classification: a survey")
        assert result == "multi label classification a survey"


# =====================================================================
# 2. Four-gate promotion
# =====================================================================

class TestFourGatePromotion:
    def setup_method(self):
        self.bank = LabelBank(
            min_freq=3,
            min_source_docs=2,
            min_agreement=0.5,
            min_semantic_distance=0.3,
        )

    def _cluster(self, freq=5, n_docs=3, agreement_sum=4.0, agreement_count=5,
                 nearest_distance=0.5) -> ProtoLabelCluster:
        return ProtoLabelCluster(
            cluster_id="test",
            representative_phrase="test phrase",
            freq=freq,
            agreement_sum=agreement_sum,
            agreement_count=agreement_count,
            source_docs={f"d{i}" for i in range(n_docs)},
            nearest_label_distance=nearest_distance,
            state="hold",
        )

    def test_all_gates_pass(self):
        c = self._cluster()
        assert self.bank._should_promote(c) is True

    def test_fail_freq(self):
        c = self._cluster(freq=2)
        assert self.bank._should_promote(c) is False

    def test_fail_source_docs(self):
        c = self._cluster(n_docs=1)
        assert self.bank._should_promote(c) is False

    def test_fail_agreement(self):
        """agreement = 1.0/5 = 0.2 < 0.5 threshold."""
        c = self._cluster(agreement_sum=1.0, agreement_count=5)
        assert self.bank._should_promote(c) is False

    def test_fail_semantic_distance(self):
        """distance 0.1 < 0.3 threshold → too close to existing label."""
        c = self._cluster(nearest_distance=0.1)
        assert self.bank._should_promote(c) is False

    def test_none_distance_defaults_to_1(self):
        """When no nearest label found, distance defaults to 1.0 → passes."""
        c = self._cluster(nearest_distance=None)
        c.nearest_label_distance = None
        assert self.bank._should_promote(c) is True

    def test_boundary_agreement_exactly_at_threshold(self):
        """agreement = 2.5/5 = 0.5 exactly equals threshold → passes."""
        c = self._cluster(agreement_sum=2.5, agreement_count=5)
        assert self.bank._should_promote(c) is True


# =====================================================================
# 3. MatchResult routing
# =====================================================================

class TestMatchResultRouting:
    def setup_method(self):
        self.bank = LabelBank(min_freq=2, min_source_docs=2,
                              min_agreement=0.0, min_semantic_distance=0.0,
                              dense_encoder=_test_dense_encoder,
                              cluster_merge_threshold=0.80,
                              cluster_merge_margin=0.01)
        self.bank.register_label("cs.AI", "Artificial Intelligence")

    def test_merge_pre_adds_alias(self):
        result = _match("AI research", "merge_pre", target_label="cs.AI", similarity=0.9)
        action = self.bank.process_match_result(result)
        assert action == "merge"
        assert "ai research" in self.bank.labels["cs.AI"].aliases

    def test_novel_pre_creates_cluster(self):
        result = _match("quantum computing", "novel_pre", similarity=0.1)
        action = self.bank.process_match_result(result)
        assert action == "hold"  # first occurrence, not enough freq
        assert len(self.bank.proto_label_clusters) == 1
        cluster = next(iter(self.bank.proto_label_clusters.values()))
        assert cluster.representative_phrase == "quantum computing"

    def test_hold_pre_creates_hold(self):
        result = _match("bioinformatics", "hold_pre", similarity=0.6)
        action = self.bank.process_match_result(result)
        assert action == "hold"
        assert len(self.bank.hold_pool) == 1
        cluster = next(iter(self.bank.hold_pool.values()))
        assert cluster.representative_phrase == "bioinformatics"

    def test_discard_action(self):
        result = _match("etc", "discard", similarity=0.0)
        action = self.bank.process_match_result(result)
        assert action == "discard"

    def test_novel_promotes_after_enough_evidence(self):
        """Two docs × 1 phrase each → freq=2, docs=2 → candidate."""
        r1 = _match("quantum computing", "novel_pre", doc_id="d1", similarity=0.1)
        r2 = _match("quantum computing", "novel_pre", doc_id="d2", similarity=0.1)
        self.bank.process_match_result(r1)
        action = self.bank.process_match_result(r2)
        assert action == "candidate"
        assert len(self.bank.candidate_labels) == 1
        cluster = next(iter(self.bank.candidate_labels.values()))
        assert cluster.representative_phrase == "quantum computing"


# =====================================================================
# 4. Cluster accumulation
# =====================================================================

class TestClusterAccumulation:
    def setup_method(self):
        self.bank = LabelBank(min_freq=3, min_source_docs=2,
                              min_agreement=0.5, min_semantic_distance=0.0,
                              dense_encoder=_test_dense_encoder,
                              cluster_merge_threshold=0.80,
                              cluster_merge_margin=0.01)

    def test_representative_phrase_by_vote(self):
        """The most frequent surface form wins."""
        for _ in range(3):
            self.bank._upsert_cluster(_phrase("Deep Learning", "d1"))
        self.bank._upsert_cluster(_phrase("deep learning", "d2"))

        cluster = next(iter(self.bank.proto_label_clusters.values()))
        assert cluster.representative_phrase == "deep learning"
        assert cluster.freq == 4

    def test_agreement_aggregation(self):
        self.bank._upsert_cluster(_phrase("NLP", "d1", agreement=0.8))
        self.bank._upsert_cluster(_phrase("NLP", "d2", agreement=0.6))
        cluster = next(iter(self.bank.proto_label_clusters.values()))
        assert abs(cluster.agreement - 0.7) < 1e-9

    def test_source_doc_dedup(self):
        for _ in range(5):
            self.bank._upsert_cluster(_phrase("XAI", "same_doc"))
        cluster = next(iter(self.bank.proto_label_clusters.values()))
        assert cluster.freq == 5
        assert cluster.source_doc_count == 1  # only 1 unique doc

    def test_semantic_variants_merge_into_same_cluster(self):
        self.bank._upsert_cluster(_phrase("graph neural network", "d1"))
        self.bank._upsert_cluster(_phrase("graph neural networks", "d2"))
        self.bank._upsert_cluster(_phrase("graph-based neural network", "d3"))
        assert len(self.bank.proto_label_clusters) == 1
        cluster = next(iter(self.bank.proto_label_clusters.values()))
        assert cluster.freq == 3
        assert cluster.source_doc_count == 3
        assert cluster.representative_phrase in {
            "graph neural network",
            "graph neural networks",
            "graph based neural network",
        }


# =====================================================================
# 5. Ontology constraints
# =====================================================================

class TestOntologyConstraints:
    def setup_method(self):
        self.checker = OntologyConstraintChecker.for_aapd()

    def test_valid_label(self):
        v = self.checker.check("cs.NLP", "nlp", set())
        assert v is None

    def test_bad_naming_format(self):
        v = self.checker.check("deeplearning", "deep learning", set())
        assert v is not None
        assert v.rule == "naming_format"

    def test_unknown_parent(self):
        v = self.checker.check("xx.AI", "ai", set())
        assert v is not None
        assert v.rule == "parent_existence"

    def test_duplicate_rejection(self):
        v = self.checker.check("cs.AI", "ai", {"cs.ai"})
        assert v is not None
        assert v.rule == "duplicate"

    def test_case_insensitive_duplicate(self):
        """CS.AI matches naming pattern, so use a valid-format duplicate."""
        v = self.checker.check("cs.AI", "ai", {"cs.AI"})
        assert v is not None
        assert v.rule == "duplicate"


# =====================================================================
# 6. SQLite persistence
# =====================================================================

class TestSQLitePersistence:
    def setup_method(self):
        self._tmpfile = os.path.join(tempfile.gettempdir(), "owlu_test_persist.db")

    def teardown_method(self):
        try:
            os.unlink(self._tmpfile)
        except (OSError, PermissionError):
            pass

    def test_save_and_load_labels(self):
        writer = OntologyWriter(db_path=self._tmpfile)
        writer.register_existing_label("cs.AI", "Artificial Intelligence",
                                       aliases=["AI", "machine intelligence"])
        writer.register_existing_label("stat.ML", "Machine Learning")
        writer.save()

        writer2 = OntologyWriter.from_db(self._tmpfile)
        inv = writer2.get_label_inventory()
        assert "cs.AI" in inv
        assert "stat.ML" in inv
        assert "ai" in writer2.bank.labels["cs.AI"].aliases
        writer2._store.close()
        writer._store.close()

    def test_save_and_load_clusters(self):
        writer = OntologyWriter(
            db_path=self._tmpfile,
            min_freq=2, min_source_docs=2,
            min_agreement=0.0, min_semantic_distance=0.0,
            dense_encoder=_test_dense_encoder,
            cluster_merge_threshold=0.80,
            cluster_merge_margin=0.01,
        )
        writer.register_existing_label("cs.AI", "AI")
        r1 = _match("quantum computing", "novel_pre", doc_id="d1", similarity=0.1)
        r2 = _match("quantum computing", "novel_pre", doc_id="d2", similarity=0.1)
        writer.ingest(r1)
        writer.ingest(r2)

        writer2 = OntologyWriter.from_db(self._tmpfile, dense_encoder=_test_dense_encoder)
        assert len(writer2.bank.candidate_labels) == 1
        cluster = next(iter(writer2.bank.candidate_labels.values()))
        assert cluster.freq == 2
        assert cluster.source_doc_count == 2
        writer2._store.close()
        writer._store.close()

    def test_params_persisted(self):
        writer = OntologyWriter(
            db_path=self._tmpfile,
            min_freq=5, min_source_docs=3,
            min_agreement=0.7, min_semantic_distance=0.4,
        )
        writer.save()

        writer2 = OntologyWriter.from_db(self._tmpfile)
        assert writer2.bank.min_freq == 5
        assert writer2.bank.min_source_docs == 3
        assert abs(writer2.bank.min_agreement - 0.7) < 1e-9
        assert abs(writer2.bank.min_semantic_distance - 0.4) < 1e-9
        writer2._store.close()
        writer._store.close()

    def test_promoted_cluster_round_trip_uses_explicit_mapping(self):
        writer = OntologyWriter(
            db_path=self._tmpfile,
            min_freq=2,
            min_source_docs=2,
            min_agreement=0.0,
            min_semantic_distance=0.0,
            dense_encoder=_test_dense_encoder,
            cluster_merge_threshold=0.80,
            cluster_merge_margin=0.01,
        )
        writer.register_existing_label("cs.AI", "Artificial Intelligence")
        writer.ingest(_match("graph neural network", "novel_pre", doc_id="d1", similarity=0.1))
        writer.ingest(_match("graph neural networks", "novel_pre", doc_id="d2", similarity=0.1))
        cluster_id = next(iter(writer.bank.candidate_labels))
        assert writer.promote(
            cluster_id,
            "cs.GNN",
            skip_constraints=True,
        ) is True

        writer2 = OntologyWriter.from_db(self._tmpfile, dense_encoder=_test_dense_encoder)
        promoted = writer2.get_promoted_labels()
        assert "cs.GNN" in promoted
        assert promoted["cs.GNN"].cluster_id.startswith("cluster_")
        writer2._store.close()
        writer._store.close()

    def test_document_examples_promote_and_export(self):
        writer = OntologyWriter(
            db_path=self._tmpfile,
            min_freq=2,
            min_source_docs=2,
            min_agreement=0.0,
            min_semantic_distance=0.0,
            dense_encoder=_test_dense_encoder,
            cluster_merge_threshold=0.80,
            cluster_merge_margin=0.01,
        )

        doc_1 = "Graph neural networks are used for molecular property prediction."
        doc_2 = "This work studies graph neural networks for traffic forecasting."

        r1 = _match("graph neural network", "novel_pre", doc_id="d1", similarity=0.1)
        r2 = _match("graph neural networks", "novel_pre", doc_id="d2", similarity=0.1)

        assert writer.ingest_with_document(r1, document_text=doc_1) == "hold"
        assert writer.ingest_with_document(r2, document_text=doc_2) == "candidate"

        cluster_id = next(iter(writer.bank.candidate_labels))
        assert writer.promote(cluster_id, "cs.GNN", skip_constraints=True) is True
        assert writer.count_label_examples("cs.GNN") == 2
        assert writer.get_slow_sync_ready_labels(min_positive_examples=2) == ["cs.GNN"]

        samples = writer.export_ltce_samples(min_positive_examples=2)
        assert len(samples) == 2
        assert all(isinstance(sample, LtceTextSample) for sample in samples)
        assert {sample.doc_id for sample in samples} == {"d1", "d2"}
        assert all(sample.true_labels == {"cs.GNN"} for sample in samples)

        writer._store.close()

    def test_merge_examples_are_exportable_for_existing_labels(self):
        writer = OntologyWriter(db_path=self._tmpfile)
        writer.register_existing_label("cs.AI", "Artificial Intelligence")

        result = _match(
            "AI research",
            "merge_pre",
            doc_id="merge_doc",
            target_label="cs.AI",
            similarity=0.95,
        )
        writer.ingest_with_document(
            result,
            document_text="AI research helps improve planning systems.",
        )

        assert writer.count_label_examples("cs.AI") == 1

        samples = writer.export_ltce_samples(
            label_ids=["cs.AI"],
            min_positive_examples=1,
            promoted_only=False,
        )
        assert len(samples) == 1
        assert samples[0].true_labels == {"cs.AI"}
        writer._store.close()


# =====================================================================
# 7. Full pipeline data-flow demonstration
# =====================================================================

class TestFullPipelineDataFlow:
    """Simulates a realistic multi-document ingestion flow and verifies
    every intermediate state transition, printing detailed data for inspection."""

    def test_end_to_end_flow(self, capsys):
        """Walk through the complete lifecycle:
        doc1..doc5 → ingestion → accumulation → promotion → persistence.
        """
        print("\n" + "=" * 72)
        print("  OWLU Writer Pipeline — End-to-End Data Flow Demonstration")
        print("=" * 72)

        # --- Setup ---
        checker = OntologyConstraintChecker.for_aapd()
        writer = OntologyWriter(
            constraint_checker=checker,
            min_freq=3,
            min_source_docs=2,
            min_agreement=0.5,
            min_semantic_distance=0.3,
        )

        # Bootstrap existing labels
        existing = {
            "cs.AI": "Artificial Intelligence",
            "cs.CL": "Computation and Language",
            "stat.ML": "Machine Learning",
        }
        for lid, name in existing.items():
            writer.register_existing_label(lid, name)

        print(f"\n[INIT] Registered {len(existing)} existing labels:")
        for lid, name in existing.items():
            print(f"  • {lid}: {name}")

        # --- Simulate discovery results from 5 documents ---
        # Each MatchResult represents what Discovery would produce
        discovery_results = [
            # doc_1: LLM finds "graph neural network" (new) + "AI" (merge)
            _match("graph neural network", "novel_pre", "doc_1",
                   similarity=0.15, agreement=0.9),
            _match("AI research", "merge_pre", "doc_1",
                   target_label="cs.AI", similarity=0.92, agreement=1.0),

            # doc_2: LLM finds "graph neural network" again + "deep learning" (new)
            _match("graph neural network", "novel_pre", "doc_2",
                   similarity=0.15, agreement=0.8),
            _match("Deep-Learning", "novel_pre", "doc_2",
                   similarity=0.20, agreement=0.7),

            # doc_3: "graph neural network" 3rd time + "deep learning" 2nd
            _match("graph neural networks", "novel_pre", "doc_3",
                   similarity=0.15, agreement=0.85),
            _match("deep learning", "novel_pre", "doc_3",
                   similarity=0.20, agreement=0.6),

            # doc_4: "deep learning" 3rd time (should promote) + low-agreement junk
            _match("deep learning", "novel_pre", "doc_4",
                   similarity=0.20, agreement=0.75),
            _match("misc stuff", "novel_pre", "doc_4",
                   similarity=0.10, agreement=0.2),  # low agreement

            # doc_5: something too close to existing (high similarity)
            _match("machine learning", "hold_pre", "doc_5",
                   similarity=0.85, agreement=0.9),
        ]

        print(f"\n[DISCOVERY] Simulated {len(discovery_results)} MatchResults from 5 documents:\n")
        print(f"  {'Phrase':<30} {'Action':<12} {'Doc':<8} {'Sim':>6} {'Agr':>6} {'Target'}")
        print(f"  {'─'*30} {'─'*12} {'─'*8} {'─'*6} {'─'*6} {'─'*10}")
        for r in discovery_results:
            print(f"  {r.phrase.text:<30} {r.action:<12} {r.phrase.source_doc_id:<8} "
                  f"{r.similarity:>6.2f} {r.phrase.agreement:>6.2f} {r.target_label or '—'}")

        # --- Ingest ---
        print(f"\n[INGEST] Processing each MatchResult through Writer:\n")
        print(f"  {'Phrase':<30} {'Route Action':<15} {'Writer Result'}")
        print(f"  {'─'*30} {'─'*15} {'─'*15}")

        actions = []
        for r in discovery_results:
            action = writer.ingest(r)
            actions.append((r, action))
            print(f"  {r.phrase.text:<30} {r.action:<15} → {action}")

        # --- State after ingestion ---
        print(f"\n[STATE] After ingestion:")
        print(f"  Labels (with aliases):")
        for lid, info in writer.bank.labels.items():
            print(f"    {lid}: aliases={sorted(info.aliases)}")

        print(f"\n  Hold pool ({len(writer.bank.hold_pool)} clusters):")
        for cid, cluster in writer.bank.hold_pool.items():
            print(f"    [{cid}] freq={cluster.freq}, docs={cluster.source_doc_count}, "
                  f"agr={cluster.agreement:.2f}, dist={cluster.nearest_label_distance}, "
                  f"state={cluster.state}")

        print(f"\n  Candidate pool ({len(writer.bank.candidate_labels)} clusters):")
        for cid, cluster in writer.bank.candidate_labels.items():
            print(f"    [{cid}] freq={cluster.freq}, docs={cluster.source_doc_count}, "
                  f"agr={cluster.agreement:.2f}, dist={cluster.nearest_label_distance}, "
                  f"repr='{cluster.representative_phrase}', state={cluster.state}")
            print(f"      phrases: {dict(cluster.phrases)}")
            print(f"      source_docs: {sorted(cluster.source_docs)}")

        # --- Why some clusters did NOT promote ---
        print(f"\n[ANALYSIS] Why hold-pool clusters didn't promote:")
        bank = writer.bank
        for cid, cluster in bank.hold_pool.items():
            reasons = []
            if cluster.freq < bank.min_freq:
                reasons.append(f"freq={cluster.freq} < min_freq={bank.min_freq}")
            if cluster.source_doc_count < bank.min_source_docs:
                reasons.append(f"docs={cluster.source_doc_count} < min_docs={bank.min_source_docs}")
            if cluster.agreement < bank.min_agreement:
                reasons.append(f"agreement={cluster.agreement:.2f} < min_agr={bank.min_agreement}")
            dist = cluster.nearest_label_distance if cluster.nearest_label_distance is not None else 1.0
            if dist < bank.min_semantic_distance:
                reasons.append(f"sem_dist={dist:.2f} < min_dist={bank.min_semantic_distance}")
            if not reasons:
                reasons.append("(would pass, but was routed as hold_pre)")
            print(f"  [{cid}]: {'; '.join(reasons)}")

        # --- Promotion ---
        print(f"\n[PROMOTE] Attempting to promote candidate clusters:")
        candidates = writer.get_promotion_candidates()
        for cid, cluster in candidates.items():
            # Generate arXiv-style label for testing
            new_label = f"cs.{cluster.representative_phrase.replace(' ', '-').title().replace(' ', '')}"
            ok = writer.promote(cid, new_label, skip_constraints=True)
            print(f"  {cid} → {new_label}: {'✓ promoted' if ok else '✗ blocked'}")

        print(f"\n[FINAL] Promoted labels:")
        for lid, cluster in writer.get_promoted_labels().items():
            print(f"  {lid}: repr='{cluster.representative_phrase}', "
                  f"freq={cluster.freq}, docs={cluster.source_doc_count}")

        print(f"\n[FINAL] Complete label inventory for Discovery feedback:")
        for lid, text in sorted(writer.get_label_inventory().items()):
            print(f"  {lid}: '{text}'")

        # --- Persistence test ---
        tmpfile = os.path.join(tempfile.gettempdir(), "owlu_e2e_test.db")
        try:
            writer_p = OntologyWriter(
                constraint_checker=checker,
                min_freq=3, min_source_docs=2,
                min_agreement=0.5, min_semantic_distance=0.3,
                db_path=tmpfile,
            )
            for lid, name in existing.items():
                writer_p.register_existing_label(lid, name)
            for r in discovery_results:
                writer_p.ingest(r)
            writer_p.save()

            writer_loaded = OntologyWriter.from_db(tmpfile, constraint_checker=checker)
            print(f"\n[PERSIST] SQLite round-trip verification:")
            print(f"  Labels restored: {sorted(writer_loaded.bank.labels.keys())}")
            print(f"  Clusters restored: {len(writer_loaded.bank.proto_label_clusters)}")
            print(f"  Candidates restored: {list(writer_loaded.bank.candidate_labels.keys())}")
            print(f"  Hold pool restored: {list(writer_loaded.bank.hold_pool.keys())}")
            print(f"  Params: min_freq={writer_loaded.bank.min_freq}, "
                  f"min_docs={writer_loaded.bank.min_source_docs}, "
                  f"min_agr={writer_loaded.bank.min_agreement}, "
                  f"min_dist={writer_loaded.bank.min_semantic_distance}")
            writer_loaded._store.close()
            writer_p._store.close()
        finally:
            try:
                os.unlink(tmpfile)
            except (OSError, PermissionError):
                pass

        print("\n" + "=" * 72)
        print("  Pipeline demonstration complete")
        print("=" * 72)

        # --- Assertions for CI ---
        assert "cs.AI" in writer.bank.labels
        assert "ai research" in writer.bank.labels["cs.AI"].aliases
        assert len(writer.get_promoted_labels()) > 0
        assert len(writer.bank.hold_pool) > 0
