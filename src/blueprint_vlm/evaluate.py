"""Scoring.

Reports exact match, but also the things that decide whether a takeoff is
usable in practice: whether the JSON parsed at all, how far off the count was
when it was wrong, and whether the model's own confidence label separates its
right answers from its wrong ones.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

_JSON = re.compile(r"\{.*\}", re.S)


def parse(raw: str) -> dict | None:
    if not raw:
        return None
    m = _JSON.search(raw)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        return None


DICT_KEYS = ("takeoff", "rooms")


def _correct(pred: dict, rec: dict) -> tuple[bool, float | None]:
    """(exact match, absolute error) -- error is None for count-map tasks.

    Dispatch is on the target's payload key rather than the task name, so a new
    count-map task scores correctly without touching this function.
    """
    tgt = rec["target"]
    for key in DICT_KEYS:
        if key in tgt:
            got = pred.get(key)
            if not isinstance(got, dict):
                return False, None
            want = tgt[key]
            norm = {str(k): got.get(str(k)) for k in want}
            return norm == want, None
    got = pred.get("answer")
    if not isinstance(got, (int, float)) or isinstance(got, bool):
        return False, None
    return int(got) == tgt["answer"], abs(int(got) - tgt["answer"])


def score(records: list[dict], predictions: list[str]) -> dict:
    assert len(records) == len(predictions), "records and predictions must align"
    by_task: dict[str, list] = defaultdict(list)
    rows = []

    for rec, raw in zip(records, predictions):
        pred = parse(raw)
        ok, err = (False, None) if pred is None else _correct(pred, rec)
        conf = (pred or {}).get("confidence")
        conf = conf if conf in ("HIGH", "LOW") else None
        row = {"task": rec["task"], "plan_id": rec["plan_id"],
               "valid_json": pred is not None, "correct": ok,
               "abs_err": err, "confidence": conf, "raw": raw}
        rows.append(row)
        by_task[rec["task"]].append(row)

    def agg(rs: list[dict]) -> dict:
        n = len(rs) or 1
        errs = [r["abs_err"] for r in rs if r["abs_err"] is not None]
        high = [r for r in rs if r["confidence"] == "HIGH"]
        return {
            "n": len(rs),
            "json_valid": round(sum(r["valid_json"] for r in rs) / n, 4),
            "exact_match": round(sum(r["correct"] for r in rs) / n, 4),
            "mae": round(sum(errs) / len(errs), 3) if errs else None,
            "off_by_one": round(sum(e == 1 for e in errs) / len(errs), 4) if errs else None,
            "high_coverage": round(len(high) / n, 4),
            "high_precision": round(sum(r["correct"] for r in high) / len(high), 4) if high else None,
        }

    return {
        "overall": agg(rows),
        "by_task": {t: agg(rs) for t, rs in sorted(by_task.items())},
        "rows": rows,
    }


def table(result: dict, title: str) -> str:
    """Markdown table, ready to paste into the README."""
    hdr = ("| split / task | n | JSON valid | exact match | MAE | off-by-1 | "
           "HIGH cov. | HIGH prec. |\n|---|--:|--:|--:|--:|--:|--:|--:|")
    def fmt(k, m):
        def pc(v): return "-" if v is None else f"{v*100:.1f}%"
        return (f"| {k} | {m['n']} | {pc(m['json_valid'])} | {pc(m['exact_match'])} | "
                f"{'-' if m['mae'] is None else m['mae']} | {pc(m['off_by_one'])} | "
                f"{pc(m['high_coverage'])} | {pc(m['high_precision'])} |")
    lines = [f"**{title}**", "", hdr, fmt("overall", result["overall"])]
    lines += [fmt(t, m) for t, m in result["by_task"].items()]
    return "\n".join(lines)
