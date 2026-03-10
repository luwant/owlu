"""SQLite persistence for OWLU Task 3."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Iterator

from .label_bank import LabelBank


@dataclass(frozen=True)
class SQLiteConfig:
    db_path: str


class Task3SQLiteStore:
    """Minimal SQLite data-access layer with transactional writes."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(Path(db_path))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS label_snapshots (
                    run_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    label_id TEXT NOT NULL,
                    aliases_json TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, phase, label_id)
                );

                CREATE TABLE IF NOT EXISTS promoted_label_records (
                    run_id TEXT NOT NULL,
                    label_id TEXT NOT NULL,
                    cluster_id TEXT NOT NULL,
                    representative_phrase TEXT NOT NULL,
                    evidence_docs_json TEXT NOT NULL,
                    freq INTEGER NOT NULL,
                    source_docs INTEGER NOT NULL,
                    agreement REAL NOT NULL,
                    nearest_label_id TEXT,
                    nearest_label_distance REAL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, label_id)
                );

                CREATE TABLE IF NOT EXISTS slow_sync_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS e2e_reports (
                    run_id TEXT NOT NULL,
                    scenario TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, scenario)
                );
                """
            )

    def record_label_snapshot(self, run_id: str, phase: str, label_bank: LabelBank) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.transaction() as conn:
            for label_id, info in label_bank.labels.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO label_snapshots
                    (run_id, phase, label_id, aliases_json, description, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        phase,
                        label_id,
                        json.dumps(sorted(info.aliases), ensure_ascii=False),
                        info.description or "",
                        now,
                    ),
                )

    def record_promoted_labels(self, run_id: str, label_bank: LabelBank) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.transaction() as conn:
            for label_id, cluster in label_bank.promoted_labels.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO promoted_label_records
                    (run_id, label_id, cluster_id, representative_phrase, evidence_docs_json,
                     freq, source_docs, agreement, nearest_label_id, nearest_label_distance, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        label_id,
                        cluster.cluster_id,
                        cluster.representative_phrase,
                        json.dumps(sorted(cluster.evidence_docs), ensure_ascii=False),
                        int(cluster.freq),
                        int(cluster.source_doc_count),
                        float(cluster.agreement),
                        cluster.nearest_label_id,
                        cluster.nearest_label_distance,
                        now,
                    ),
                )

    def record_slow_sync_run(self, run_id: str, report: dict[str, object]) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO slow_sync_runs (run_id, created_at, payload_json)
                VALUES (?, ?, ?)
                """,
                (
                    run_id,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(report, ensure_ascii=False, sort_keys=True),
                ),
            )

    def record_e2e_report(self, run_id: str, scenario: str, report: dict[str, object]) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO e2e_reports (run_id, scenario, created_at, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    run_id,
                    scenario,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(report, ensure_ascii=False, sort_keys=True),
                ),
            )

    def fetch_slow_sync_run(self, run_id: str) -> dict[str, object] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM slow_sync_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(json.loads(row["payload_json"]))

    def fetch_e2e_reports(self, run_id: str) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT scenario, payload_json FROM e2e_reports WHERE run_id = ? ORDER BY scenario ASC",
                (run_id,),
            ).fetchall()
        for row in rows:
            out[str(row["scenario"])] = dict(json.loads(row["payload_json"]))
        return out

    def fetch_label_snapshot(self, run_id: str, phase: str) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT label_id, aliases_json, description, created_at
                FROM label_snapshots
                WHERE run_id = ? AND phase = ?
                ORDER BY label_id ASC
                """,
                (run_id, phase),
            ).fetchall()
        for row in rows:
            out[str(row["label_id"])] = {
                "aliases": list(json.loads(row["aliases_json"])),
                "description": str(row["description"]),
                "created_at": str(row["created_at"]),
            }
        return out

    def fetch_promoted_labels(self, run_id: str) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT label_id, cluster_id, representative_phrase, evidence_docs_json,
                       freq, source_docs, agreement, nearest_label_id, nearest_label_distance, created_at
                FROM promoted_label_records
                WHERE run_id = ?
                ORDER BY label_id ASC
                """,
                (run_id,),
            ).fetchall()
        for row in rows:
            out[str(row["label_id"])] = {
                "cluster_id": str(row["cluster_id"]),
                "representative_phrase": str(row["representative_phrase"]),
                "evidence_docs": list(json.loads(row["evidence_docs_json"])),
                "freq": int(row["freq"]),
                "source_docs": int(row["source_docs"]),
                "agreement": float(row["agreement"]),
                "nearest_label_id": row["nearest_label_id"],
                "nearest_label_distance": row["nearest_label_distance"],
                "created_at": str(row["created_at"]),
            }
        return out

    def list_run_ids(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id FROM slow_sync_runs
                UNION
                SELECT run_id FROM e2e_reports
                UNION
                SELECT run_id FROM label_snapshots
                UNION
                SELECT run_id FROM promoted_label_records
                ORDER BY run_id ASC
                """
            ).fetchall()
        return [str(row["run_id"]) for row in rows]
