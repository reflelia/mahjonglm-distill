"""Look up a round snapshot by its decision_id."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decision_id", help="decision_id from an all/actual review JSON")
    parser.add_argument("--db", type=Path, default=Path("data/round_states.sqlite"))
    args = parser.parse_args()
    if not args.db.is_file():
        parser.error(f"Database not found: {args.db}; run build_round_state_db.py first")
    with sqlite3.connect(f"file:{args.db.resolve().as_posix()}?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT state_json FROM decisions WHERE decision_id = ?", (args.decision_id,)
        ).fetchone()
    if row is None:
        print(f"decision_id not found: {args.decision_id}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(json.loads(row[0]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
