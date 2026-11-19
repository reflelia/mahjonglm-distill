"""Build a SQLite lookup database of public round state keyed by decision_id."""

import argparse
import json
from pathlib import Path
import sqlite3

from mahjonglm_review_common import board_from_decision, iter_decisions
import re


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/mahjonglm_2024_decisions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/round_states.sqlite"))
    parser.add_argument("--replace", action="store_true", help="Replace an existing database")
    args = parser.parse_args()
    if args.output.exists() and not args.replace:
        parser.error(f"Database exists: {args.output}; use --replace to rebuild")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.output)
    count = 0
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("""CREATE TABLE IF NOT EXISTS decisions (
            decision_id TEXT PRIMARY KEY,
            game_id TEXT NOT NULL,
            round_id TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            state_json TEXT NOT NULL
        )""")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_decisions_round ON decisions(round_id)")
        if args.replace:
            connection.execute("DELETE FROM decisions")
        connection.execute("BEGIN")
        for row_index, decision in iter_decisions(args.input, 0, 2**63 - 1):
            board = board_from_decision(decision)
            tokens = decision["context_tokens"]
            wind = next((int(m.group(1)) for t in tokens if (m := re.fullmatch(r"bakaze_(\d+)", t))), None)
            kyoku = next((int(m.group(1)) for t in tokens if (m := re.fullmatch(r"kyoku_(\d+)", t))), None)
            public_players = {
                seat: {
                    "자리풍": board["seat_winds"].get(seat),
                    "discards": player["discards"],
                    "melds": player["melds"],
                    "riichi": player["riichi"],
                }
                for seat, player in board["players"].items()
            }
            state = {
                "identifiers": {
                    key: decision.get(key) for key in (
                        "game_id", "round_id", "decision_id", "round_index",
                        "actor_seat", "actor_discard_number", "round_discard_number",
                    )
                },
                "공개_국면": {
                    "장풍": wind,
                    "장풍_한글": board["round_wind"],
                    "국": kyoku + 1 if kyoku is not None else None,
                    "친_좌석": board["dealer_seat"],
                    "자리풍_안내": board["seat_wind_note"],
                    "인원수": decision["seat_count"],
                    "도라_표시패": board["dora_indicators"],
                    "플레이어": public_players,
                },
            }
            connection.execute(
                "INSERT INTO decisions VALUES (?, ?, ?, ?, ?)",
                (decision["decision_id"], decision["game_id"], decision["round_id"],
                 row_index, json.dumps(state, ensure_ascii=False)),
            )
            count += 1
            if count % 500 == 0:
                connection.commit()
                connection.execute("BEGIN")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    print(f"Indexed {count} decisions in {args.output}")


if __name__ == "__main__":
    main()
