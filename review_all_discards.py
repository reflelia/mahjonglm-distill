"""Score every distinct tile in a MahjongLM player's current hand."""

import argparse
import json
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from openai import OpenAI

from mahjonglm_review_common import (
    AllTileReviews, ORDER_INDEX, board_from_decision, bounded_map,
    decision_identifiers, describe_melds, enrich_saved_review, evaluate,
    iter_decisions, tile_name,
)


def process_one(index: int, decision: dict, args, client_options: dict) -> str:
    board = board_from_decision(decision)
    required = list(board["hand"])
    task = (
        f"Score each of these discard tile types exactly once: {required}. "
        "For every tile, write a distinct 2-4 sentence Korean reason grounded "
        "in the current hand and visible board."
    )
    if args.sum_scores_to_100:
        task += (
            " The integer scores for all candidate tiles must sum to exactly 100. "
            "Allocate the 100 points according to their relative desirability; "
            "check the total before returning."
        )
    if args.dry_run:
        return json.dumps({"row": index, "task": task, "board": board}, ensure_ascii=False, indent=2)
    output = args.output or Path("data/llm_reviews") / f"row-{index}-all.json"
    if args.output and args.limit > 1:
        output = args.output.with_name(f"{args.output.stem}-row-{index}{args.output.suffix or '.json'}")
    if args.enrich_existing:
        if not output.exists():
            return f"missing: {output}"
        changed = enrich_saved_review(output, board, decision)
        return f"{'updated' if changed else 'already current'}: {output}"
    if output.exists():
        return f"skipped existing: {output}"

    def validate(review: AllTileReviews) -> None:
        actual = [candidate.tile for candidate in review.candidates]
        missing = sorted(set(required) - set(actual))
        extra = sorted(set(actual) - set(required))
        duplicate = sorted({tile for tile in actual if actual.count(tile) > 1})
        if missing or extra or duplicate:
            raise ValueError(
                f"Score each required tile exactly once: {required}; "
                f"missing={missing}, extra={extra}, duplicate={duplicate}"
            )
        if args.sum_scores_to_100 and sum(candidate.score for candidate in review.candidates) != 100:
            raise ValueError("Candidate scores must sum to exactly 100")
        if any(not re.search(r"[가-힣]", candidate.reason) for candidate in review.candidates):
            raise ValueError("Every reason must contain Korean text")

    with OpenAI(**client_options) as client:
        review = evaluate(
            client, board, AllTileReviews, task, args.model,
            validate=validate, max_attempts=args.retries + 1,
            max_output_tokens=args.max_output_tokens,
        )
    result = {
        candidate.tile: {
            "count": board["hand"][candidate.tile],
            "tile_name": tile_name(candidate.tile),
            "score": candidate.score,
            "reason": candidate.reason,
        }
        for candidate in sorted(review.candidates, key=lambda candidate: ORDER_INDEX[candidate.tile])
    }
    result["후로"] = describe_melds(board, {board["viewer_seat"]})
    result["식별자"] = decision_identifiers(decision)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return f"saved: {output}"


def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/mahjonglm_2024_decisions.jsonl"))
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1, help="Number of consecutive decisions to evaluate")
    parser.add_argument("--workers", type=int, default=1, help="Maximum concurrent API requests (default: 1)")
    parser.add_argument("--timeout", type=float, default=600, help="API timeout in seconds (default: 600)")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    parser.add_argument("--retries", type=int, default=2, help="Additional attempts after an invalid response")
    parser.add_argument("--max-output-tokens", type=int, default=10000)
    parser.add_argument(
        "--sum-scores-to-100", action="store_true",
        help="Prompt the model to allocate exactly 100 points across candidate tiles",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enrich-existing", action="store_true", help="Compact existing JSON to scores, own melds, and identifiers without API calls")
    args = parser.parse_args()
    if args.row < 0 or args.limit < 1 or args.retries < 0 or args.workers < 1 or args.max_output_tokens < 1 or args.timeout <= 0:
        parser.error("--row and --retries must be nonnegative; --limit, --workers, --max-output-tokens and --timeout must be positive")
    if not (args.dry_run or args.enrich_existing) and not os.getenv("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY or use --dry-run")
    client_options = {"timeout": args.timeout}
    if base_url := os.getenv("OPENAI_BASE_URL"):
        client_options["base_url"] = base_url
    items = iter_decisions(args.input, args.row, args.limit)
    handler = lambda index, decision: process_one(index, decision, args, client_options)
    workers = 1 if args.dry_run or args.enrich_existing else args.workers
    for message in bounded_map(items, handler, workers):
        print(message, flush=True)


if __name__ == "__main__":
    main()
