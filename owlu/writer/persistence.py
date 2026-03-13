"""SQLite persistence for LabelBank state and slow-sync training samples."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..common.types import LabelInfo, LtceTextSample, MatchResult, ProtoLabelCluster


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
    centroid_json           TEXT,
    freq                    INTEGER NOT NULL DEFAULT 0,
    agreement_sum           REAL    NOT NULL DEFAULT 0.0,
    agreement_count         INTEGER NOT NULL DEFAULT 0,
    nearest_label_id        TEXT,
    nearest_label_distance  REAL,
    promoted_label_id       TEXT,
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

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    text         TEXT NOT NULL,
    source_type  TEXT NOT NULL DEFAULT 'discovery',
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS label_examples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id         TEXT NOT NULL,
    phrase_text    TEXT NOT NULL,
    cluster_id     TEXT,
    label_id       TEXT,
    review_status  TEXT NOT NULL DEFAULT 'pending',
    is_positive    INTEGER NOT NULL DEFAULT 1,
    split          TEXT NOT NULL DEFAULT 'train',
    source_type    TEXT NOT NULL DEFAULT 'discovery',
    created_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (doc_id, phrase_text, source_type)
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
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(clusters)")
        }
        if "centroid_json" not in columns:
            conn.execute(
                "ALTER TABLE clusters ADD COLUMN centroid_json TEXT"
            )
        if "promoted_label_id" not in columns:
            conn.execute(
                "ALTER TABLE clusters ADD COLUMN promoted_label_id TEXT"
            )
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Training sample persistence
    # ------------------------------------------------------------------

    def record_match_result(
        self,
        result: MatchResult,
        writer_action: str,
        *,
        document_text: str,
        source_type: str = "discovery",
        split: str = "train",
        cluster_id: str | None = None,
    ) -> None:
        """Persist a document and its current label-evidence linkage."""
        doc_id = result.phrase.source_doc_id
        if not doc_id or not document_text:
            return

        review_status = "pending"
        label_id = None
        if writer_action == "merge" and result.target_label:
            label_id = result.target_label
            review_status = "approved"
        elif writer_action not in {"candidate", "hold"}:
            return

        conn = self._get_conn()
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents (doc_id, text, source_type) VALUES (?, ?, ?)",
                (doc_id, document_text, source_type),
            )
            conn.execute(
                "INSERT INTO label_examples "
                "(doc_id, phrase_text, cluster_id, label_id, review_status, is_positive, split, source_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(doc_id, phrase_text, source_type) DO UPDATE SET "
                "cluster_id=excluded.cluster_id, "
                "label_id=COALESCE(excluded.label_id, label_examples.label_id), "
                "review_status=excluded.review_status, "
                "is_positive=excluded.is_positive, "
                "split=excluded.split",
                (
                    doc_id,
                    result.phrase.text,
                    cluster_id,
                    label_id,
                    review_status,
                    1,
                    split,
                    source_type,
                ),
            )

    def approve_cluster_examples(self, cluster_id: str, new_label_id: str) -> None:
        """Promote all examples linked to a cluster into approved label positives."""
        conn = self._get_conn()
        with conn:
            conn.execute(
                "UPDATE label_examples "
                "SET label_id = ?, review_status = 'approved', is_positive = 1 "
                "WHERE cluster_id = ?",
                (new_label_id, cluster_id),
            )

    def count_label_examples(
        self,
        label_id: str,
        *,
        review_status: str = "approved",
        split: str | None = None,
    ) -> int:
        """Count distinct documents that support a given label."""
        conn = self._get_conn()
        query = (
            "SELECT COUNT(DISTINCT doc_id) FROM label_examples "
            "WHERE label_id = ? AND review_status = ? AND is_positive = 1"
        )
        params: list[object] = [label_id, review_status]
        if split is not None:
            query += " AND split = ?"
            params.append(split)
        row = conn.execute(query, tuple(params)).fetchone()
        return 0 if row is None else int(row[0] or 0)

    def get_slow_sync_ready_labels(
        self,
        *,
        min_positive_examples: int = 3,
        review_status: str = "approved",
    ) -> list[str]:
        """Return labels with enough approved evidence to justify slow-sync."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT label_id "
            "FROM label_examples "
            "WHERE label_id IS NOT NULL AND review_status = ? AND is_positive = 1 "
            "GROUP BY label_id "
            "HAVING COUNT(DISTINCT doc_id) >= ? "
            "ORDER BY label_id",
            (review_status, int(min_positive_examples)),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def export_ltce_samples(
        self,
        *,
        label_ids: list[str] | None = None,
        min_positive_examples: int = 1,
        review_status: str = "approved",
    ) -> list[LtceTextSample]:
        """Export approved positive examples as LtceTextSample objects."""
        conn = self._get_conn()

        eligible_labels = set(label_ids or self.get_slow_sync_ready_labels(
            min_positive_examples=min_positive_examples,
            review_status=review_status,
        ))
        if label_ids is not None:
            eligible_labels = {
                label_id
                for label_id in eligible_labels
                if self.count_label_examples(
                    label_id,
                    review_status=review_status,
                ) >= min_positive_examples
            }

        if not eligible_labels:
            return []

        placeholders = ", ".join("?" for _ in eligible_labels)
        query = (
            "SELECT e.doc_id, d.text, e.split, e.label_id "
            "FROM label_examples e "
            "JOIN documents d ON d.doc_id = e.doc_id "
            f"WHERE e.review_status = ? AND e.is_positive = 1 AND e.label_id IN ({placeholders}) "
            "ORDER BY e.doc_id, e.label_id"
        )
        params = [review_status, *sorted(eligible_labels)]

        grouped: dict[tuple[str, str, str], set[str]] = {}
        for doc_id, text, split, label_id in conn.execute(query, tuple(params)):
            key = (str(doc_id), str(text), str(split))
            grouped.setdefault(key, set()).add(str(label_id))

        samples: list[LtceTextSample] = []
        for (doc_id, text, split), labels in grouped.items():
            samples.append(
                LtceTextSample(
                    doc_id=doc_id,
                    text=text,
                    true_labels=labels,
                    split=split,  # type: ignore[arg-type]
                )
            )
        return samples

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
            conn.execute("DELETE FROM cluster_source_docs")
            conn.execute("DELETE FROM cluster_phrases")
            conn.execute("DELETE FROM clusters")

            all_clusters: dict[str, ProtoLabelCluster] = {}
            all_clusters.update(label_bank.proto_label_clusters)
            promoted_cluster_ids: dict[str, str] = {}
            for lid, cluster in label_bank.promoted_labels.items():
                all_clusters[cluster.cluster_id] = cluster
                promoted_cluster_ids[cluster.cluster_id] = lid

            for cid, cluster in all_clusters.items():
                conn.execute(
                    "INSERT OR REPLACE INTO clusters "
                    "(cluster_id, representative_phrase, centroid_json, freq, agreement_sum, "
                    " agreement_count, nearest_label_id, nearest_label_distance, promoted_label_id, state) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        cid,
                        cluster.representative_phrase,
                        (
                            json.dumps(cluster.centroid_embedding)
                            if cluster.centroid_embedding is not None
                            else None
                        ),
                        cluster.freq,
                        cluster.agreement_sum,
                        cluster.agreement_count,
                        cluster.nearest_label_id,
                        cluster.nearest_label_distance,
                        promoted_cluster_ids.get(cid),
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
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("cluster_merge_threshold", str(label_bank.cluster_merge_threshold)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("cluster_merge_margin", str(label_bank.cluster_merge_margin)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("next_cluster_index", str(label_bank._next_cluster_index)),
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
            cluster_merge_threshold=float(
                meta.get("cluster_merge_threshold", "0.84")
            ),
            cluster_merge_margin=float(meta.get("cluster_merge_margin", "0.04")),
        )
        bank._next_cluster_index = int(meta.get("next_cluster_index", "1"))

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
            "SELECT cluster_id, representative_phrase, centroid_json, freq, agreement_sum, "
            "agreement_count, nearest_label_id, nearest_label_distance, promoted_label_id, state "
            "FROM clusters"
        ):
            (
                cid,
                rep,
                centroid_json,
                freq,
                asum,
                acount,
                nearest_lid,
                nearest_dist,
                promoted_label_id,
                state,
            ) = row
            cluster = ProtoLabelCluster(
                cluster_id=cid,
                representative_phrase=rep,
                centroid_embedding=(
                    [float(v) for v in json.loads(centroid_json)]
                    if centroid_json
                    else None
                ),
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
                cluster.evidence_docs.add(drow[0])

            bank.proto_label_clusters[cid] = cluster
            if state == "candidate":
                bank.candidate_labels[cid] = cluster
            elif state == "hold":
                bank.hold_pool[cid] = cluster
            elif state == "promoted":
                label_id = str(promoted_label_id) if promoted_label_id else None
                if label_id is None:
                    for lid, info in bank.labels.items():
                        if info.description == f"Promoted from cluster {cid}":
                            label_id = lid
                            break
                if label_id is not None:
                    bank.promoted_labels[label_id] = cluster

        bank._refresh_cluster_counter()
        return bank
