"""Split discipline.

Seed ranges are disjoint and fixed here once. The held-out sets are generated
from seeds that no design-time script ever touches, and `test_hard` shifts the
distribution (more rooms, denser symbols) to separate "learned the task" from
"learned this generator".
"""

from __future__ import annotations

TRAIN = range(0, 2000)
DEV = range(2000, 2300)
TEST = range(900_000, 900_400)          # held out, same distribution
TEST_HARD = range(950_000, 950_200)     # held out, shifted distribution

SPLITS = {"train": TRAIN, "dev": DEV, "test": TEST, "test_hard": TEST_HARD}
HARD_SPLITS = {"test_hard"}
