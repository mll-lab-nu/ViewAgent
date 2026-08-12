"""
AI2-THOR scene catalog.

Covers the 120 iTHOR FloorPlans (4 room types * 30 plans each):
  - kitchen       : FloorPlan1..30
  - living_room   : FloorPlan201..230
  - bedroom       : FloorPlan301..330
  - bathroom      : FloorPlan401..430

Provides:
  - FULL_POOL: all 120 scene ids
  - default_test_subset(n=30): balanced subset across room types
"""
from __future__ import annotations

from typing import Dict, List


ROOM_RANGES: Dict[str, tuple[int, int]] = {
    "kitchen":     (1,   30),
    "living_room": (201, 230),
    "bedroom":     (301, 330),
    "bathroom":    (401, 430),
}


def _scenes_for(range_: tuple[int, int]) -> List[str]:
    lo, hi = range_
    return [f"FloorPlan{i}" for i in range(lo, hi + 1)]


def room_scenes(room: str) -> List[str]:
    if room not in ROOM_RANGES:
        raise KeyError(f"unknown room type: {room!r}; expected {list(ROOM_RANGES)}")
    return _scenes_for(ROOM_RANGES[room])


FULL_POOL: List[str] = sum((_scenes_for(r) for r in ROOM_RANGES.values()), [])


def default_test_subset(n_total: int = 30) -> List[str]:
    """
    Return a balanced subset across the 4 room types, preserving FloorPlan
    numeric ordering inside each type. If n_total is not divisible by 4, the
    remainder is distributed to the first types (kitchen, living_room, ...).
    """
    if n_total <= 0:
        return []
    rooms = list(ROOM_RANGES)
    base = n_total // len(rooms)
    rem = n_total % len(rooms)

    out: List[str] = []
    for i, room in enumerate(rooms):
        k = base + (1 if i < rem else 0)
        out.extend(room_scenes(room)[:k])
    return out


def parse_subset(spec) -> List[str]:
    """
    Parse scene subset spec.

    Accepts either a comma-separated string or a list/tuple of tokens:
      - "all"                   -> FULL_POOL (120 scenes)
      - "default"               -> default_test_subset(30)
      - "default:N"             -> default_test_subset(N)
      - "kitchen" / "bedroom"   -> that room's 30 scenes
      - "kitchen:8,bedroom:8"   -> per-room take
      - "FloorPlan5,FloorPlan7" -> explicit list
      - ["FloorPlan5", "FloorPlan7"]  -> same as above
    """
    if spec is None:
        s = ""
    elif isinstance(spec, (list, tuple)):
        s = ",".join(str(p) for p in spec)
    else:
        s = str(spec)
    s = s.strip()
    if not s or s == "all":
        return list(FULL_POOL)
    if s == "default":
        return default_test_subset(30)
    if s.startswith("default:"):
        return default_test_subset(int(s.split(":", 1)[1]))
    if s in ROOM_RANGES:
        return room_scenes(s)

    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[str] = []
    for p in parts:
        if ":" in p and p.split(":", 1)[0] in ROOM_RANGES:
            room, k = p.split(":", 1)
            out.extend(room_scenes(room)[: int(k)])
        else:
            if not p.startswith("FloorPlan"):
                raise ValueError(f"bad scene spec part: {p!r}")
            out.append(p)
    return out
