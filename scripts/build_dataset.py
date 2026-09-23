#!/usr/bin/env python
"""Generate plans + task records for every split. CPU only, no model needed."""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blueprint_vlm import synth, tasks
from blueprint_vlm.splits import HARD_SPLITS, SPLITS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--only", nargs="*", default=None, help="subset of splits")
    args = ap.parse_args()

    root = Path(args.out)
    for name, seeds in SPLITS.items():
        if args.only and name not in args.only:
            continue
        split_dir = root / name
        metas = synth.build(split_dir, seeds, hard=name in HARD_SPLITS)

        rng = random.Random(hash(name) & 0xFFFF)
        records = []
        for m in metas:
            m = dict(m, image=f"{name}/{m['image']}")
            records.extend(tasks.build(m, rng))
        (split_dir / "records.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records))
        print(f"{name:10s} {len(metas):5d} plans  {len(records):6d} records"
              f"{'  [shifted]' if name in HARD_SPLITS else ''}")


if __name__ == "__main__":
    main()
