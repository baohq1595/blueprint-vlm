#!/usr/bin/env python
"""Concatenate sharded prediction files back into one, in shard order."""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, help="in shard-id order")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    parts = [json.loads(Path(f).read_text()) for f in args.shards]
    parts.sort(key=lambda d: d.get("shard_id", 0))
    ids = [p.get("shard_id", 0) for p in parts]
    assert ids == list(range(len(parts))), f"missing or duplicate shard: {ids}"

    preds = [p for part in parts for p in part["predictions"]]
    keys = [k for part in parts for k in part.get("keys", [])]
    out = {"split": parts[0]["split"], "adapter": parts[0]["adapter"],
           "limit": parts[0].get("limit"), "predictions": preds}
    if keys:
        out["keys"] = keys
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"merged {len(parts)} shards -> {len(preds)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
