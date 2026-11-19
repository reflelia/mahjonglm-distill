"""Extract player-visible discard decisions from MahjongLM 2024 Arrow shards."""

import argparse
import json
from pathlib import Path
import re

import pyarrow as pa


DISCARD = re.compile(r"discard_(\d+)_(m[0-9]|p[0-9]|s[0-9]|z[1-7])_(tedashi|tsumogiri)$")


def load_vocabulary(path: Path) -> dict[int, str]:
    tokenizer = json.loads(path.read_text(encoding="utf-8"))
    vocab = tokenizer["model"]["vocab"]
    return {int(index): token for token, index in vocab.items()}


def extract_row(row: dict, vocabulary: dict[int, str]):
    if row["view_type"] != "imperfect":
        return
    tokens = [vocabulary[index] for index in row["input_ids"]]
    round_index = -1
    round_start = 0
    game_header = []
    viewer = None
    actor_discard_counts = [0] * row["seat_count"]
    round_discard_number = 0
    for position, token in enumerate(tokens):
        if token == "round_start":
            round_index += 1
            round_start = position
            viewer = None
            actor_discard_counts = [0] * row["seat_count"]
            round_discard_number = 0
            if round_index == 0:
                game_header = tokens[:position]
        if token.startswith("haipai_") and token.removeprefix("haipai_").isdigit():
            viewer = int(token.removeprefix("haipai_"))
        match = DISCARD.fullmatch(token)
        if match:
            seat = int(match.group(1))
            actor_discard_counts[seat] += 1
            round_discard_number += 1
        if match and viewer is not None and int(match.group(1)) == viewer:
            round_id = f"{row['game_id']}:{round_index}"
            actor_discard_number = actor_discard_counts[viewer]
            yield {
                "game_id": row["game_id"],
                "round_id": round_id,
                "decision_id": f"{round_id}:{viewer}:{actor_discard_number}",
                "year": row["year"],
                "seat_count": row["seat_count"],
                "viewer_seat": row["viewer_seat"],
                "actor_seat": viewer,
                "round_index": round_index,
                "round_discard_number": round_discard_number,
                "actor_discard_number": actor_discard_number,
                "token_position": position,
                "discard_tile": match.group(2),
                "discard_type": match.group(3),
                "context_tokens": game_header + tokens[round_start:position],
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/mahjonglm_dataset"))
    parser.add_argument("--output", type=Path, default=Path("data/mahjonglm_2024_decisions.jsonl"))
    parser.add_argument("--limit", type=int, help="Maximum number of decisions to write")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    vocab_path = args.dataset / "tokenizer" / "tokenizer.json"
    if not vocab_path.is_file():
        parser.error(f"Tokenizer missing: {vocab_path}; rerun download_mahjonglm_2024.py")
    shards = sorted((args.dataset / "2024").glob("*.arrow"))
    if not shards:
        parser.error(f"No Arrow shards found in {args.dataset / '2024'}")
    vocabulary = load_vocabulary(vocab_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as output:
        for shard in shards:
            with pa.memory_map(str(shard)) as source:
                for batch in pa.ipc.open_stream(source):
                    for start in range(0, batch.num_rows, 16):
                        for row in batch.slice(start, 16).to_pylist():
                            for decision in extract_row(row, vocabulary) or ():
                                output.write(json.dumps(decision, ensure_ascii=False) + "\n")
                                count += 1
                                if args.limit is not None and count >= args.limit:
                                    print(f"Saved {count} decisions to {args.output}")
                                    return
    print(f"Saved {count} decisions to {args.output}")


if __name__ == "__main__":
    main()
