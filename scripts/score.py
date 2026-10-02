#!/usr/bin/env python
"""Score a predictions file against a split and print the markdown table."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blueprint_vlm import evaluate as E
from blueprint_vlm.evaluate import load_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    records = load_records(args.split)
    data = json.loads(Path(args.predictions).read_text())
    preds = data["predictions"]

    # Three ways to line predictions up with records, best first.
    if data.get("keys"):
        index = {f"{r['plan_id']}::{r['task']}": r for r in records}
        missing = [k for k in data["keys"] if k not in index]
        if missing:
            raise SystemExit(
                f"{len(missing)} scored records are not in {args.split} "
                f"(first: {missing[0]}). The split was rebuilt after inference ran.")
        records = [index[k] for k in data["keys"]]
    elif data.get("limit"):
        records = records[:data["limit"]]
    elif len(preds) < len(records):
        print(f"note: {len(preds)} predictions for {len(records)} records; "
              f"assuming the first {len(preds)} (a --limit run with no keys recorded)")
        records = records[:len(preds)]

    if len(records) != len(preds):
        raise SystemExit(
            f"cannot align: {len(records)} records vs {len(preds)} predictions. "
            f"Re-run inference so the predictions file records its keys.")

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
