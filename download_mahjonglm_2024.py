"""Download only the 2024 portion of mitsutani/mahjonglm-dataset."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "mitsutani/mahjonglm-dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/mahjonglm_dataset"),
        help="Local directory that will contain the 2024/ subdirectory",
    )
    parser.add_argument("--tokenizer-only", action="store_true")
    args = parser.parse_args()
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=["tokenizer/**"] if args.tokenizer_only else ["2024/**", "tokenizer/**"],
        local_dir=args.output,
    )
    print(f"Downloaded 2024 data and tokenizer to {args.output.resolve()}")


if __name__ == "__main__":
    main()
