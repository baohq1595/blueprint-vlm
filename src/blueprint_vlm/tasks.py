"""Turn a plan into prompt/target pairs.

Three task types, deliberately chosen so the legend lookup can be isolated:

  count_by_name  "how many windows"  -- no legend needed (control)
  count_by_code  "how many G3"       -- legend lookup required
  takeoff        full JSON by code   -- the product-shaped task

Comparing count_by_name against count_by_code measures what the cross-reference
step alone costs. That difference is the number worth reporting.
"""

from __future__ import annotations

import json
import random

SYSTEM = (
    "You read architectural floor plans. Answer only with JSON. "
    "Set confidence to HIGH when you are certain of the count, LOW when you are not."
)

Q_NAME = 'How many {name} symbols are on this plan? Reply {{"answer": <int>, "confidence": "HIGH"|"LOW"}}'
Q_CODE = ('Using the legend, how many {code} symbols are on this plan? '
          'Reply {{"answer": <int>, "confidence": "HIGH"|"LOW"}}')
Q_TAKE = ('Produce a quantity takeoff. Count every symbol in the legend and reply '
          '{"takeoff": {"<code>": <int>, ...}, "confidence": "HIGH"|"LOW"}')
Q_ROOMS = ('Read the room labels. Count rooms by name, ignoring any number suffix '
           '(BEDROOM 1 and BEDROOM 2 are two BEDROOM). Reply '
           '{"rooms": {"<NAME>": <int>, ...}, "confidence": "HIGH"|"LOW"}')


def build(meta: dict, rng: random.Random) -> list[dict]:
    """One plan -> a list of {task, image, question, target} records."""
    out = []
    legend, counts = meta["legend"], meta["counts"]

    code, name = rng.choice(list(legend.items()))
    out.append({
        "task": "count_by_name", "plan_id": meta["plan_id"], "image": meta["image"],
        "question": Q_NAME.format(name=name),
        "target": {"answer": counts[name], "confidence": "HIGH"},
    })

    code2, _ = rng.choice(list(legend.items()))
    out.append({
        "task": "count_by_code", "plan_id": meta["plan_id"], "image": meta["image"],
        "question": Q_CODE.format(code=code2),
        "target": {"answer": meta["counts_by_code"][code2], "confidence": "HIGH"},
    })

    out.append({
        "task": "takeoff", "plan_id": meta["plan_id"], "image": meta["image"],
        "question": Q_TAKE,
        "target": {"takeoff": meta["counts_by_code"], "confidence": "HIGH"},
    })

    # A count map, not a list and not a set. A set would throw away the count,
    # and "how many bedrooms" is exactly the question a takeoff has to answer.
    out.append({
        "task": "room_tally", "plan_id": meta["plan_id"], "image": meta["image"],
        "question": Q_ROOMS,
        "target": {"rooms": meta["room_counts"], "confidence": "HIGH"},
    })
    return out


def to_messages(rec: dict, image_root: str, with_answer: bool) -> list[dict]:
    """Qwen2.5-VL chat format."""
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [
            {"type": "image", "image": f"{image_root}/{rec['image']}"},
            {"type": "text", "text": rec["question"]},
        ]},
    ]
    if with_answer:
        msgs.append({"role": "assistant",
                     "content": [{"type": "text", "text": json.dumps(rec["target"])}]})
    return msgs
