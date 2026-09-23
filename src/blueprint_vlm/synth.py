"""Synthetic architectural floor plans with exact ground truth.

Each plan carries a legend mapping an arbitrary code (randomised per image) to a
symbol glyph. A question asked by code therefore cannot be answered from the
symbol alone -- the model has to read the legend, ground the glyph on the
drawing, and count. That is the cross-sheet reference pattern real drawings use,
reduced to something with exact ground truth.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1024, 768
MARGIN = 60
LEGEND_W = 250
WALL = 5

SYMBOLS = ["door", "window", "toilet", "sink", "bathtub", "stove"]
ROOM_NAMES = ["BEDROOM", "KITCHEN", "BATH", "LIVING", "HALL", "STUDY", "UTILITY"]
CODE_POOL = ["A1", "A2", "B1", "B3", "C2", "D1", "D4", "E2", "F1", "G3", "H2", "K4"]


@dataclass
class Plan:
    plan_id: str
    counts: dict[str, int]          # canonical symbol name -> count
    legend: dict[str, str]          # code -> canonical symbol name
    rooms: list[str]                # labels as drawn: ["BEDROOM 1", "BEDROOM 2", "KITCHEN"]
    room_counts: dict[str, int]     # the label shape: {"BEDROOM": 2, "KITCHEN": 1}
    seed: int
    image: Image.Image = field(repr=False, default=None)

    @property
    def counts_by_code(self) -> dict[str, int]:
        return {c: self.counts[n] for c, n in self.legend.items()}

    def meta(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "seed": self.seed,
            "counts": self.counts,
            "legend": self.legend,
            "counts_by_code": self.counts_by_code,
            "rooms": self.rooms,
            "room_counts": self.room_counts,
        }


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                      # Pillow < 10
        return ImageFont.load_default()


# ---------------------------------------------------------------- glyphs

def _door(d: ImageDraw.ImageDraw, x: int, y: int, s: int = 34, horiz: bool = True):
    """Architectural door: leaf line plus swing arc."""
    if horiz:
        d.line([(x, y), (x + s, y)], fill="white", width=WALL + 2)      # gap in wall
        d.line([(x, y), (x, y - s)], fill="black", width=3)             # leaf
        d.arc([x - s, y - s, x + s, y + s], 270, 360, fill="black", width=2)
    else:
        d.line([(x, y), (x, y + s)], fill="white", width=WALL + 2)
        d.line([(x, y), (x + s, y)], fill="black", width=3)
        d.arc([x - s, y - s, x + s, y + s], 0, 90, fill="black", width=2)


def _window(d: ImageDraw.ImageDraw, x: int, y: int, s: int = 40, horiz: bool = True):
    if horiz:
        d.line([(x, y), (x + s, y)], fill="white", width=WALL + 2)
        d.line([(x, y - 3), (x + s, y - 3)], fill="black", width=2)
        d.line([(x, y + 3), (x + s, y + 3)], fill="black", width=2)
    else:
        d.line([(x, y), (x, y + s)], fill="white", width=WALL + 2)
        d.line([(x - 3, y), (x - 3, y + s)], fill="black", width=2)
        d.line([(x + 3, y), (x + 3, y + s)], fill="black", width=2)


def _toilet(d, x, y, s=22):
    d.rounded_rectangle([x, y, x + s * 0.7, y + s], radius=7, outline="black", width=2)
    d.rectangle([x + s * 0.15, y - 6, x + s * 0.55, y], outline="black", width=2)


def _sink(d, x, y, s=22):
    d.rectangle([x, y, x + s, y + s], outline="black", width=2)
    d.ellipse([x + 4, y + 4, x + s - 4, y + s - 4], outline="black", width=2)


def _bathtub(d, x, y, s=22):
    d.rounded_rectangle([x, y, x + s * 1.7, y + s], radius=6, outline="black", width=2)
    d.ellipse([x + s * 1.3, y + s * 0.35, x + s * 1.5, y + s * 0.6], outline="black", width=2)


def _stove(d, x, y, s=22):
    d.rectangle([x, y, x + s, y + s], outline="black", width=2)
    for dx, dy in ((0.28, 0.28), (0.72, 0.28), (0.28, 0.72), (0.72, 0.72)):
        d.ellipse([x + s * dx - 3, y + s * dy - 3, x + s * dx + 3, y + s * dy + 3],
                  outline="black", width=1)


GLYPH = {"toilet": _toilet, "sink": _sink, "bathtub": _bathtub, "stove": _stove}


# ---------------------------------------------------------------- layout

def _split(rect, depth, rng, min_side=150):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    if depth <= 0:
        return [rect]
    if w >= h and w >= 2 * min_side:
        cut = rng.randint(x0 + min_side, x1 - min_side)
        return (_split((x0, y0, cut, y1), depth - 1, rng, min_side)
                + _split((cut, y0, x1, y1), depth - 1, rng, min_side))
    if h >= 2 * min_side:
        cut = rng.randint(y0 + min_side, y1 - min_side)
        return (_split((x0, y0, x1, cut), depth - 1, rng, min_side)
                + _split((x0, cut, x1, y1), depth - 1, rng, min_side))
    return [rect]


def generate(seed: int, hard: bool = False) -> Plan:
    rng = random.Random(seed)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_room, f_leg, f_title = _font(15), _font(14), _font(18)

    shell = (MARGIN, MARGIN, W - LEGEND_W - MARGIN, H - MARGIN)
    rooms = _split(shell, rng.randint(3, 4) if hard else rng.randint(2, 3), rng,
                   min_side=110 if hard else 150)
    counts = dict.fromkeys(SYMBOLS, 0)

    # Names are assigned up front so repeats can be numbered the way real plans
    # do. Two rooms both labelled "BEDROOM" are visually ambiguous; "BEDROOM 1"
    # and "BEDROOM 2" are not, and that is what an architect would draw.
    base_names = [rng.choice(ROOM_NAMES) for _ in rooms]
    tally = Counter(base_names)
    seen: dict[str, int] = defaultdict(int)
    room_labels = []
    for n in base_names:
        if tally[n] > 1:
            seen[n] += 1
            room_labels.append(f"{n} {seen[n]}")
        else:
            room_labels.append(n)

    for r in rooms:
        d.rectangle(r, outline="black", width=WALL)

    for idx, (x0, y0, x1, y1) in enumerate(rooms):
        d.text((x0 + 14, y0 + 12), room_labels[idx], fill="black", font=f_room)

        for _ in range(rng.randint(1, 2)):                       # doors
            if rng.random() < 0.5 and x1 - x0 > 120:
                _door(d, rng.randint(x0 + 30, x1 - 70), rng.choice([y0, y1]), horiz=True)
            else:
                _door(d, rng.choice([x0, x1]), rng.randint(y0 + 30, y1 - 70), horiz=False)
            counts["door"] += 1

        for _ in range(rng.randint(1, 3)):                       # windows on shell edges
            if y0 == shell[1] or y1 == shell[3]:
                _window(d, rng.randint(x0 + 30, max(x0 + 31, x1 - 70)),
                        shell[1] if y0 == shell[1] else shell[3], horiz=True)
                counts["window"] += 1
            elif x0 == shell[0] or x1 == shell[2]:
                _window(d, shell[0] if x0 == shell[0] else shell[2],
                        rng.randint(y0 + 30, max(y0 + 31, y1 - 70)), horiz=False)
                counts["window"] += 1

        for _ in range(rng.randint(1, 5) if hard else rng.randint(0, 3)):    # fixtures
            fx = rng.choice(["toilet", "sink", "bathtub", "stove"])
            px = rng.randint(x0 + 30, max(x0 + 31, x1 - 70))
            py = rng.randint(y0 + 45, max(y0 + 46, y1 - 60))
            GLYPH[fx](d, px, py)
            counts[fx] += 1

    # legend: codes shuffled per image so the mapping cannot be memorised
    present = [s for s in SYMBOLS if counts[s] > 0]
    codes = rng.sample(CODE_POOL, len(present))
    legend = dict(zip(codes, present))

    lx = W - LEGEND_W - 10
    d.rectangle([lx, MARGIN, W - 20, MARGIN + 42 + 34 * len(legend)], outline="black", width=2)
    d.text((lx + 14, MARGIN + 12), "LEGEND", fill="black", font=f_title)
    for i, (code, sym) in enumerate(legend.items()):
        cy = MARGIN + 48 + 34 * i
        if sym == "door":
            _door(d, lx + 20, cy + 16, s=18, horiz=True)
        elif sym == "window":
            _window(d, lx + 20, cy + 14, s=22, horiz=True)
        else:
            GLYPH[sym](d, lx + 20, cy + 2, s=18)
        d.text((lx + 74, cy + 6), code, fill="black", font=f_leg)

    return Plan(plan_id=f"plan_{seed:05d}", counts=counts, legend=legend,
                rooms=room_labels, room_counts=dict(tally), seed=seed, image=img)


def build(out_dir: Path, seeds: range, hard: bool = False) -> list[dict]:
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    metas = []
    for s in seeds:
        p = generate(s, hard=hard)
        p.image.save(out_dir / "images" / f"{p.plan_id}.png")
        m = p.meta()
        m["image"] = f"images/{p.plan_id}.png"
        metas.append(m)
    (out_dir / "plans.jsonl").write_text("\n".join(json.dumps(m) for m in metas))
    return metas
