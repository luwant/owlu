"""SQLite persistence for LabelBank state.

Stores labels, aliases, clusters, phrases, and source-doc links
so that the LabelBank can survive process restarts.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..common.types import LabelInfo, ProtoLabelCluster


# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS labels (
    label_id    TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS label_aliases (
    label_id TEXT NOT NULL,
    alias    TEXT NOT NULL,
    PRIMARY KEY (label_id, alias),
    FOREIGN KEY (label_id) REFERENCES labels(label_id)
);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id              TEXT PRIMARY KEY,
    representative_phrase   TEXT NOT NULL,
    freq                    INTEGER NOT NULL DEFAULT 0,
    agreement_sum           REAL    NOT NULL DEFAULT 0.0,
    agreement_count         INTEGER NOT NULL DEFAULT 0,
    nearest_label_id        TEXT,
    nearest_label_distance  REAL,
    state                   TEXT NOT NULL DEFAULT 'hold'
);

CREATE TABLE IF NOT EXISTS cluster_phrases (
    cluster_id TEXT NOT NULL,
    phrase     TEXT NOT NULL,
    count      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (cluster_id, phrase),
    FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id)
);

CREATE TABLE IF NOT EXISTS cluster_source_docs (
    cluster_id TEXT NOT NULL,
    doc_id     TEXT NOT NULL,
    PRIMARY KEY (cluster_id, doc_id),
    FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id)
);

CREATE TABLE IF NOT EXISTS cluster_evidence_docs (
    cluster_id TEXT NOT NULL,
    doc_id     TEXT NOT NULL,
    PRIMARY KEY (cluster_id, doc_id),
    FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id)
);

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class LabelBankStore:
    """SQLite-backed persistence for :class:`LabelBank`.

    Usage::

        store = LabelBankStore("owlu_state.db")
        store.save(label_bank)          # dump current state
        label_bank = store.load()       # reconstruct from disk
        store.save(label_bank)          # idempotent upsert
    """

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, label_bank: object) -> None:
        """Persist full LabelBank state (upsert semantics)."""
        from ..writer.label_bank import LabelBank

        if not isinstance(label_bank, LabelBank):
            raise TypeError("Expected a LabelBank instance")

        conn = self._get_conn()
        with conn:
            # --- labels & aliases ---
            conn.execute("DELETE FROM label_aliases")
            conn.execute("DELETE FROM labels")
            for label_id, info in label_bank.labels.items():
                conn.execute(
                    "INSERT OR REPLACE INTO labels (label_id, description) VALUES (?, ?)",
                    (label_id, info.description),
                )
                for alias in sorted(info.aliases):
                    conn.execute(
                        "INSERT OR REPLACE INTO label_aliases (label_id, alias) VALUES (?, ?)",
                        (label_id, alias),
                    )

            # --- clusters ---
            conn.execute("DELETE FROM cluster_evidence_docs")
            conn.execute("DELETE FROM cluster_source_docs")
            conn.execute("DELETE FROM cluster_phrases")
            conn.execute("DELETE FROM clusters")

            all_clusters: dict[str, ProtoLabelCluster] = {}
            all_clusters.update(label_bank.proto_label_clusters)
            # promoted_labels keys are label_ids, not cluster_ids,
            # but they carry the cluster data we need to persist
            for lid, cluster in label_bank.promoted_labels.items():
                all_clusters[cluster.cluster_id] = cluster

            for cid, cluster in all_clusters.items():
                conn.execute(
                    "INSERT OR REPLACE INTO clusters "
                    "(cluster_id, representative_phrase, freq, agreement_sum, "
                    " agreement_count, nearest_label_id, nearest_label_distance, state) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        cid,
                        cluster.representative_phrase,
                        cluster.freq,
                        cluster.agreement_sum,
                        cluster.agreement_count,
                        cluster.nearest_label_id,
                        cluster.nearest_label_distance,
                        cluster.state,
                    ),
                )
                for phrase, count in cluster.phrases.items():
                    conn.execute(
                        "INSERT OR REPLACE INTO cluster_phrases (cluster_id, phrase, count) "
                        "VALUES (?, ?, ?)",
                        (cid, phrase, count),
                    )
                for doc_id in cluster.source_docs:
                    conn.execute(
                        "INSERT OR REPLACE INTO cluster_source_docs (cluster_id, doc_id) "
                        "VALUES (?, ?)",
                        (cid, doc_id),
                    )
                for doc_id in cluster.evidence_docs:
                    conn.execute(
                        "INSERT OR REPLACE INTO cluster_evidence_docs (cluster_id, doc_id) "
                        "VALUES (?, ?)",
                        (cid, doc_id),
                    )

            # --- metadata ---
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("min_freq", str(label_bank.min_freq)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("min_source_docs", str(label_bank.min_source_docs)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("min_agreement", str(label_bank.min_agreement)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("min_semantic_distance", str(label_bank.min_semantic_distance)),
            )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> object:
        """Reconstruct a LabelBank from persisted state."""
        from ..writer.label_bank import LabelBank

        conn = self._get_conn()

        # --- metadata ---
        meta: dict[str, str] = {}
        for row in conn.execute("SELECT key, value FROM metadata"):
            meta[row[0]] = row[1]

        bank = LabelBank(
            min_freq=int(meta.get("min_freq", "3")),
            min_source_docs=int(meta.get("min_source_docs", "2")),
            min_agreement=float(meta.get("min_agreement", "0.5")),
            min_semantic_distance=float(meta.get("min_semantic_distance", "0.3")),
        )

        # --- labels ---
        for row in conn.execute("SELECT label_id, description FROM labels"):
            label_id, description = row
            bank.labels[label_id] = LabelInfo(
                label_id=label_id,
                aliases=set(),
                description=description or "",
            )

        for row in conn.execute("SELECT label_id, alias FROM label_aliases"):
            label_id, alias = row
            if label_id in bank.labels:
                bank.labels[label_id].aliases.add(alias)

        # --- clusters ---
        for row in conn.execute(
            "SELECT cluster_id, representative_phrase, freq, agreement_sum, "
            "agreement_count, nearest_label_id, nearest_label_distance, state "
            "FROM clusters"
        ):
            (
                cid,
                rep,
                freq,
                asum,
                acount,
                nearest_lid,
                nearest_dist,
                state,
            ) = row
            cluster = ProtoLabelCluster(
                cluster_id=cid,
                representative_phrase=rep,
                freq=freq,
                agreement_sum=asum,
                agreement_count=acount,
                nearest_label_id=nearest_lid,
                nearest_label_distance=nearest_dist,
                state=state,
            )

            # phrases
            for prow in conn.execute(
                "SELECT phrase, count FROM cluster_phrases WHERE cluster_id = ?",
                (cid,),
            ):
                cluster.phrases[prow[0]] = prow[1]

            # source docs
            for drow in conn.execute(
                "SELECT doc_id FROM cluster_source_docs WHERE cluster_id = ?",
                (cid,),
            ):
                cluster.source_docs.add(drow[0])

            # evidence docs
            for erow in conn.execute(
                "SELECT doc_id FROM cluster_evidence_docs WHERE cluster_id = ?",
                (cid,),
            ):
                cluster.evidence_docs.add(erow[0])

            bank.proto_label_clusters[cid] = cluster
            if state == "candidate":
                bank.candidate_labels[cid] = cluster
            elif state == "hold":
                bank.hold_pool[cid] = cluster
            elif state == "promoted":
                # promoted clusters are indexed by label_id in bank.promoted_labels
                # Try to find the corresponding label_id
                for lid, info in bank.labels.items():
                    if info.description == f"Promoted from cluster {cid}":
                        bank.promoted_labels[lid] = cluster
                        break

        return bank
