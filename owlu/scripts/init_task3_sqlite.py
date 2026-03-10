"""Initialize SQLite schema for OWLU Task 3.

Usage:
  python -m owlu.scripts.init_task3_sqlite --db e:/lwt/workspace/owlu/owlu_task3.db
"""

from __future__ import annotations

import argparse
from pathlib import Path

from owlu.task3_sqlite import Task3SQLiteStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize OWLU Task 3 SQLite schema.")
    parser.add_argument("--db", required=True, help="Path to SQLite database file.")
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    store = Task3SQLiteStore(db_path)
    store.init_schema()
    print(f"Initialized Task 3 SQLite schema: {db_path}")


if __name__ == "__main__":
    main()
