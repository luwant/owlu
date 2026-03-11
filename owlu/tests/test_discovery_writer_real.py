"""Real DeepSeek API integration test: Discovery → Writer full pipeline.

Uses actual AAPD documents and the real DeepSeek API (no mocks).
Reads from Label-gen/data/aapd/aapd_test.tsv and owlu/configs/owlu.yaml.

Run:
    python -m pytest owlu/tests/test_discovery_writer_real.py -v -s
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

from owlu.common.types import OWLUConfig, MatchResult
from owlu.discovery import CandidateDiscovery
from owlu.discovery.gate import LtceGate
from owlu.discovery.phrase_generator import LLMPhraseGenerator
from owlu.discovery.matcher import SemanticMatcher
from owlu.writer import OntologyWriter
from owlu.writer.constraints import OntologyConstraintChecker
from owlu.writer.persistence import LabelBankStore
from owlu.common.encoder import BertEncoder


# =====================================================================
# Paths
# =====================================================================

_WORKSPACE = Path(__file__).resolve().parents[2]  # owlu/tests/.. → workspace root
_CONFIG_PATH = _WORKSPACE / "owlu" / "configs" / "owlu.yaml"
_AAPD_TEST = _WORKSPACE / "Label-gen" / "data" / "aapd" / "aapd_test.tsv"
_AAPD_CATS = _WORKSPACE / "Label-gen" / "data" / "aapd" / "arxiv_category_table.csv"


def _load_aapd_labels() -> dict[str, str]:
    """Load AAPD label inventory {label_id: human name}."""
    label_names = {
        "cs.ai": "Artificial Intelligence",
        "cs.cl": "Computation and Language",
        "cs.cv": "Computer Vision",
        "cs.ir": "Information Retrieval",
        "cs.lg": "Machine Learning",
        "cs.it": "Information Theory",
        "cs.cr": "Cryptography and Security",
        "cs.db": "Databases",
        "cs.dc": "Distributed Computing",
        "cs.dm": "Discrete Mathematics",
        "cs.ds": "Data Structures and Algorithms",
        "cs.ne": "Neural and Evolutionary Computing",
        "cs.ni": "Networking and Internet Architecture",
        "cs.se": "Software Engineering",
        "cs.si": "Social and Information Networks",
        "stat.ml": "Statistical Machine Learning",
        "stat.me": "Statistics Methodology",
        "stat.ap": "Statistics Applications",
        "math.co": "Combinatorics",
        "math.oc": "Optimization and Control",
        "math.pr": "Probability",
        "cond-mat.stat-mech": "Statistical Mechanics",
        "physics.soc-ph": "Physics and Society",
        "quant-ph": "Quantum Physics",
    }
    return label_names


def _load_aapd_docs(n: int = 8) -> list[dict]:
    """Load n documents from AAPD test set.

    Returns list of {doc_id, text, binary_labels}
    """
    # Build binary → label_id mapping
    binary_to_label: dict[int, str] = {}
    if _AAPD_CATS.exists():
        with open(_AAPD_CATS, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2 and row[0] != "binary_code":
                    # Find bit position
                    code = row[0]
                    idx = code.index("1") if "1" in code else -1
                    if idx >= 0:
                        binary_to_label[idx] = row[1]

    docs = []
    with open(_AAPD_TEST, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            binary_labels = parts[0]
            text = parts[1]

            # Decode binary labels
            true_labels = []
            for bit_idx, ch in enumerate(binary_labels):
                if ch == "1" and bit_idx in binary_to_label:
                    true_labels.append(binary_to_label[bit_idx])

            docs.append({
                "doc_id": f"aapd_test_{i}",
                "text": text,
                "true_labels": true_labels,
            })
    return docs


# =====================================================================
# Test
# =====================================================================

class TestDiscoveryWriterRealAPI:
    """End-to-end test with real DeepSeek API calls on real AAPD data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not _CONFIG_PATH.exists():
            pytest.skip(f"Config not found: {_CONFIG_PATH}")
        if not _AAPD_TEST.exists():
            pytest.skip(f"AAPD test data not found: {_AAPD_TEST}")
        self.config = OWLUConfig.from_yaml(str(_CONFIG_PATH))
        if not self.config.llm_api_key:
            pytest.skip("No API key configured")

    def test_full_discovery_writer_pipeline(self):
        """Real API: 8 AAPD docs → Discovery → Writer → observe data flow."""
        # Avoid GBK encoding errors on Windows console for non-ASCII chars
        sys.stdout.reconfigure(errors="replace")

        print("\n" + "=" * 78)
        print("  REAL API Integration Test: Discovery → Writer Pipeline")
        print("  Using DeepSeek API + AAPD test documents")
        print("=" * 78)

        # --- 1. Load data ---
        label_inventory = _load_aapd_labels()
        docs = _load_aapd_docs(n=8)

        print(f"\n{'─'*78}")
        print(f"[1] SETUP")
        print(f"{'─'*78}")
        print(f"  Label inventory: {len(label_inventory)} labels")
        print(f"  Documents loaded: {len(docs)}")
        print(f"  API: {self.config.llm_base_url} / {self.config.llm_model}")
        print(f"  Config: merge_th={self.config.merge_threshold}, "
              f"novel_th={self.config.novel_threshold}, "
              f"agreement_th={self.config.agreement_threshold}")

        for i, doc in enumerate(docs):
            print(f"\n  Doc[{i}] {doc['doc_id']}")
            print(f"    True labels: {doc['true_labels']}")
            print(f"    Text: {doc['text'][:120]}...")

        # --- 2. Init Discovery (gate fixed low to trigger on all docs for test) ---
        _BERT_PATH = str(_WORKSPACE / "Label-gen" / "bert" / "bert-base-uncased")
        encoder = BertEncoder(model_path=_BERT_PATH)  # local bert-base-uncased, 768-dim
        dense_fn = encoder.as_dense_encoder()
        discovery = CandidateDiscovery(
            config=self.config,
            label_inventory=label_inventory,
            fixed_threshold=0.99,  # Force gate open for all test docs
            dense_encoder=dense_fn,
        )

        # --- 3. Init Writer ---
        checker = OntologyConstraintChecker.for_aapd()
        tmpdb = os.path.join(tempfile.gettempdir(), "owlu_real_test.db")
        writer = OntologyWriter(
            constraint_checker=checker,
            min_freq=2,
            min_source_docs=2,
            min_agreement=0.3,
            min_semantic_distance=0.2,
            db_path=tmpdb,
        )
        for lid, name in label_inventory.items():
            writer.register_existing_label(lid, name)

        # --- 4. Run Discovery on each doc ---
        print(f"\n{'─'*78}")
        print(f"[2] DISCOVERY — LLM Phrase Extraction + Semantic Matching")
        print(f"{'─'*78}")

        all_results: list[tuple[dict, list[MatchResult]]] = []
        total_api_time = 0.0

        for doc in docs:
            # Simulate low-confidence logits (all near zero → gate triggers)
            fake_logits = [-5.0] * len(label_inventory)

            t0 = time.time()
            results = discovery.discover(
                doc_id=doc["doc_id"],
                text=doc["text"],
                logits=fake_logits,
            )
            elapsed = time.time() - t0
            total_api_time += elapsed

            all_results.append((doc, results))

            print(f"\n  [DOC] {doc['doc_id']} (true: {doc['true_labels']}) "
                  f"[{elapsed:.1f}s, {len(results)} phrases]")

            if not results:
                print(f"    (no phrases extracted)")
                continue

            for r in results:
                marker = ""
                if r.action == "merge_pre":
                    marker = f"MERGE -> {r.target_label}"
                elif r.action == "novel_pre":
                    marker = "NOVEL"
                elif r.action == "hold_pre":
                    marker = "HOLD"
                else:
                    marker = "DISCARD"

                print(f"    \"{r.phrase.text}\" "
                      f"(agr={r.phrase.agreement:.2f}, sim={r.similarity:.3f}) "
                      f"-> {r.action} {marker}")
                print(f"      reason: {r.decision_reason}")

        print(f"\n  Total API time: {total_api_time:.1f}s")

        # --- 5. Feed all results into Writer ---
        print(f"\n{'─'*78}")
        print(f"[3] WRITER — Ingestion & Cluster Accumulation")
        print(f"{'─'*78}")

        action_counts = {"merge": 0, "candidate": 0, "hold": 0, "discard": 0}
        for doc, results in all_results:
            for r in results:
                action = writer.ingest(r)
                action_counts[action] = action_counts.get(action, 0) + 1

        print(f"\n  Ingestion summary:")
        for a, c in sorted(action_counts.items()):
            print(f"    {a}: {c}")

        # --- 6. Show label state ---
        print(f"\n{'─'*78}")
        print(f"[4] STATE — Labels, Hold Pool, Candidates")
        print(f"{'─'*78}")

        # Labels with new aliases
        print(f"\n  Labels with NEW aliases (added via merge):")
        for lid, info in writer.bank.labels.items():
            original = label_inventory.get(lid, lid).lower()
            new_aliases = [a for a in sorted(info.aliases) if a != original]
            if new_aliases:
                print(f"    {lid}: +{new_aliases}")

        # Hold pool
        print(f"\n  Hold pool ({len(writer.bank.hold_pool)} clusters):")
        for cid, cluster in sorted(writer.bank.hold_pool.items()):
            reasons = _rejection_reasons(writer.bank, cluster)
            print(f"    [{cid}]")
            print(f"      freq={cluster.freq}, docs={cluster.source_doc_count}, "
                  f"agr={cluster.agreement:.2f}, "
                  f"dist={cluster.nearest_label_distance}")
            print(f"      phrases: {dict(cluster.phrases)}")
            print(f"      source_docs: {sorted(cluster.source_docs)}")
            print(f"      [BLOCKED] {'; '.join(reasons)}")

        # Candidates
        print(f"\n  Candidate pool ({len(writer.bank.candidate_labels)} clusters):")
        for cid, cluster in sorted(writer.bank.candidate_labels.items()):
            print(f"    [{cid}]")
            print(f"      freq={cluster.freq}, docs={cluster.source_doc_count}, "
                  f"agr={cluster.agreement:.2f}, "
                  f"dist={cluster.nearest_label_distance}")
            print(f"      phrases: {dict(cluster.phrases)}")
            print(f"      source_docs: {sorted(cluster.source_docs)}")
            print(f"      [PASS] four-gate -> candidate!")

        if not writer.bank.candidate_labels:
            print(f"    (none -- need more documents to accumulate evidence)")

        # --- 7. Attempt promotion ---
        print(f"\n{'─'*78}")
        print(f"[5] PROMOTION — Constraint checking & label creation")
        print(f"{'─'*78}")

        promoted_ids = writer.auto_promote_all(skip_constraints=True)
        if promoted_ids:
            for pid in promoted_ids:
                cluster = writer.get_promoted_labels().get(pid)
                print(f"\n  [PROMOTED] {pid}")
                if cluster:
                    print(f"     repr='{cluster.representative_phrase}', "
                          f"freq={cluster.freq}, docs={cluster.source_doc_count}")
        else:
            print(f"\n  (no clusters reached candidate status — this is normal")
            print(f"   with only {len(docs)} docs; more documents would accumulate)")


        # --- 8. Final inventory ---
        print(f"\n{'─'*78}")
        print(f"[6] FINAL — Updated label inventory")
        print(f"{'─'*78}")

        final_inv = writer.get_label_inventory()
        new_labels = {k: v for k, v in final_inv.items() if k not in label_inventory}
        print(f"\n  Original labels: {len(label_inventory)}")
        print(f"  Final labels: {len(final_inv)}")
        if new_labels:
            print(f"  NEW labels added:")
            for lid, text in new_labels.items():
                print(f"    + {lid}: '{text}'")
        else:
            print(f"  No new labels (need more doc evidence to promote)")

        # --- 9. Persistence verification ---
        print(f"\n{'─'*78}")
        print(f"[7] PERSISTENCE — SQLite round-trip")
        print(f"{'─'*78}")

        writer.save()
        writer2 = OntologyWriter.from_db(tmpdb, constraint_checker=checker)
        print(f"  Labels restored: {len(writer2.bank.labels)}")
        print(f"  Clusters restored: {len(writer2.bank.proto_label_clusters)}")
        print(f"  Hold pool: {len(writer2.bank.hold_pool)}")
        print(f"  Candidates: {len(writer2.bank.candidate_labels)}")

        # Verify data integrity
        for cid in writer.bank.proto_label_clusters:
            assert cid in writer2.bank.proto_label_clusters, f"Missing cluster: {cid}"
            orig = writer.bank.proto_label_clusters[cid]
            loaded = writer2.bank.proto_label_clusters[cid]
            assert orig.freq == loaded.freq, f"Freq mismatch for {cid}"
            assert orig.source_doc_count == loaded.source_doc_count
        print(f"  [OK] Data integrity verified -- all clusters match")

        writer2._store.close()
        writer._store.close()
        try:
            os.unlink(tmpdb)
        except (OSError, PermissionError):
            pass

        print(f"\n{'═'*78}")
        print(f"  Integration test complete -- "
              f"{sum(action_counts.values())} phrases processed, "
              f"{total_api_time:.1f}s total API time")
        print(f"{'═'*78}\n")


def _rejection_reasons(bank, cluster) -> list[str]:
    reasons = []
    if cluster.freq < bank.min_freq:
        reasons.append(f"freq={cluster.freq} < {bank.min_freq}")
    if cluster.source_doc_count < bank.min_source_docs:
        reasons.append(f"docs={cluster.source_doc_count} < {bank.min_source_docs}")
    if cluster.agreement < bank.min_agreement:
        reasons.append(f"agr={cluster.agreement:.2f} < {bank.min_agreement}")
    dist = cluster.nearest_label_distance if cluster.nearest_label_distance is not None else 1.0
    if dist < bank.min_semantic_distance:
        reasons.append(f"dist={dist:.2f} < {bank.min_semantic_distance}")
    if not reasons:
        reasons.append(f"all gates pass, but freq/docs borderline")
    return reasons
