#!/usr/bin/env python
"""Score a predictions file against a split and print the markdown table."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blueprint_vlm import evaluate as E
from blueprint_vlm.data import load_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    records = load_records(args.split)
    preds = json.loads(Path(args.predictions).read_text())["predictions"]
    result = E.score(records, preds)
    title = args.title or Path(args.predictions).stem
    print(E.table(result, title))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        slim = {k: v for k, v in result.items() if k != "rows"}
        slim["errors"] = [r for r in result["rows"] if not r["correct"]][:50]
        Path(args.out).write_text(json.dumps(slim, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
