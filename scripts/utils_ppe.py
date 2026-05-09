"""Utilities for PPE inference: bbox geometry, association rules, time-series voting.

Classes (must match data.yaml):
    0 person, 1 helmet, 2 vest, 3 head
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

CLS_PERSON, CLS_HELMET, CLS_VEST, CLS_HEAD = 0, 1, 2, 3
CLASS_NAMES = ["person", "helmet", "vest", "head"]


# -------------------- bbox geometry --------------------

def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two xyxy boxes."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def containment(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fraction of `inner` area that lies inside `outer`. 1.0 = fully inside."""
    x1 = max(inner[0], outer[0]); y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2]); y2 = min(inner[3], outer[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_inner = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return float(inter / area_inner)


def sub_region(person: np.ndarray, y_lo: float, y_hi: float) -> np.ndarray:
    """Vertical sub-band of a person bbox in normalized coords [y_lo, y_hi]."""
    x1, y1, x2, y2 = person
    h = y2 - y1
    return np.array([x1, y1 + h * y_lo, x2, y1 + h * y_hi], dtype=np.float32)


# -------------------- PPE association --------------------

@dataclass
class PPEStatus:
    has_helmet: Optional[bool] = None  # None = unknown
    has_vest: Optional[bool] = None

    def to_dict(self) -> Dict[str, Optional[bool]]:
        return {"helmet": self.has_helmet, "vest": self.has_vest}


def associate_ppe(
    person_box: np.ndarray,
    helmets: List[np.ndarray],
    heads: List[np.ndarray],
    vests: List[np.ndarray],
    contain_thr: float = 0.5,
    vest_negative_min_px: int = 80,
) -> PPEStatus:
    """Decide PPE status for a single person based on geometric association.

    Helmet rule: if any helmet box has >= contain_thr containment in the upper
    region of the person -> wearing helmet. Else if any head box does ->
    NOT wearing helmet. Else unknown.
    Vest rule: any vest with >= contain_thr containment in the middle band
    of the person -> wearing vest. Otherwise we only declare "not wearing"
    when the person bbox is large enough (shorter side >= vest_negative_min_px),
    because on small/far persons the vest detector simply misses and we should
    not equate "not detected" with "not wearing".
    """
    upper = sub_region(person_box, 0.0, 0.40)
    middle = sub_region(person_box, 0.15, 0.75)

    status = PPEStatus()

    helmet_hit = any(containment(h, upper) >= contain_thr for h in helmets)
    if helmet_hit:
        status.has_helmet = True
    else:
        head_hit = any(containment(h, upper) >= contain_thr for h in heads)
        if head_hit:
            status.has_helmet = False
        # else: leave as None (unknown)

    vest_hit = any(containment(v, middle) >= contain_thr for v in vests)
    if vest_hit:
        status.has_vest = True
    else:
        short_side = float(min(person_box[2] - person_box[0],
                               person_box[3] - person_box[1]))
        status.has_vest = False if short_side >= vest_negative_min_px else None
    return status


# -------------------- temporal voting --------------------

@dataclass
class _TrackBuf:
    helmet: Deque[Optional[bool]] = field(default_factory=deque)
    vest: Deque[Optional[bool]] = field(default_factory=deque)
    last_seen: int = 0
    alerted_helmet: bool = False
    alerted_vest: bool = False


class TrackVoter:
    """Per-track sliding-window voter to suppress single-frame flicker.

    A violation alert (no-helmet / no-vest) is fired when within the last
    `window` frames at least `thresh` frames are confidently negative.
    Once fired, it does not fire again unless the track recovers (>= thresh
    positive frames within window) and degrades again.
    """

    def __init__(self, window: int = 15, thresh: int = 10, max_missing: int = 30) -> None:
        self.window = window
        self.thresh = thresh
        self.max_missing = max_missing
        self._tracks: Dict[int, _TrackBuf] = {}
        self._frame_idx = 0

    def _push(self, dq: Deque[Optional[bool]], v: Optional[bool]) -> None:
        dq.append(v)
        while len(dq) > self.window:
            dq.popleft()

    @staticmethod
    def _count(dq: Iterable[Optional[bool]], target: bool) -> int:
        return sum(1 for x in dq if x is target)

    def update(self, track_id: int, status: PPEStatus) -> Tuple[Optional[bool], Optional[bool], List[str]]:
        """Update one track. Returns (smoothed_helmet, smoothed_vest, new_alerts).

        smoothed_* is the majority-vote value over the window (None if unknown).
        new_alerts is a list of alert tags fired on this frame for this track.
        """
        buf = self._tracks.setdefault(track_id, _TrackBuf())
        buf.last_seen = self._frame_idx
        self._push(buf.helmet, status.has_helmet)
        self._push(buf.vest, status.has_vest)

        alerts: List[str] = []

        # helmet voting (sticky: once alerted, stay False until recovery)
        neg_h = self._count(buf.helmet, False)
        pos_h = self._count(buf.helmet, True)
        smoothed_h: Optional[bool]
        if neg_h >= self.thresh:
            smoothed_h = False
            if not buf.alerted_helmet:
                alerts.append("no_helmet")
                buf.alerted_helmet = True
        elif pos_h >= self.thresh:
            smoothed_h = True
            buf.alerted_helmet = False
        elif buf.alerted_helmet:
            smoothed_h = False  # stay red until clear recovery
        else:
            smoothed_h = None

        # vest voting (sticky: once alerted, stay False until recovery)
        neg_v = self._count(buf.vest, False)
        pos_v = self._count(buf.vest, True)
        if neg_v >= self.thresh:
            smoothed_v: Optional[bool] = False
            if not buf.alerted_vest:
                alerts.append("no_vest")
                buf.alerted_vest = True
        elif pos_v >= self.thresh:
            smoothed_v = True
            buf.alerted_vest = False
        elif buf.alerted_vest:
            smoothed_v = False
        else:
            smoothed_v = None

        return smoothed_h, smoothed_v, alerts

    def step(self) -> None:
        """Advance frame index and drop tracks that have been missing too long."""
        self._frame_idx += 1
        stale = [tid for tid, b in self._tracks.items()
                 if self._frame_idx - b.last_seen > self.max_missing]
        for tid in stale:
            self._tracks.pop(tid, None)
