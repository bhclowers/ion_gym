"""Server-global LAST-FLIGHT slot.

The `/flight` tab is its own Panel session and cannot see the dashboard
session's in-memory results, so the most recent fly is PUBLISHED here —
one slot, one server, single-user lab tool. Two dashboard tabs share one
"last flight"; the stamp on every record says which flight is showing.

The integrity contract (clear, clearable, and never
corrupted) is met by construction, not by hope:

* CLEAR AT START — the slot is module state; a fresh server holds None,
  and the viewer names emptiness rather than drawing nothing silently.
* CLEARABLE — `clear(reason)` empties the slot and RETURNS what it
  cleared (stamp + site); a skip or wipe is a reported item, never a
  silent no-op.
* NO TORN STATE — `publish` builds the COMPLETE record first and swaps
  the slot reference under a lock; a reader gets the old record or the
  new one, never a partial.
* NO MALFORMED RECORD — `publish` validates every field at the door and
  REFUSES by name (raises) rather than banking a record the viewer
  would choke on later.
* NO IN-PLACE MUTATION — every path array is copied and frozen
  (`writeable=False`), so a buggy writer or reader that mutates the
  banked flight raises AT THE MUTATION SITE instead of silently
  corrupting what the next reader sees.
* DETECTION AS BACKSTOP — a fingerprint (path count, total points,
  coordinate checksum) is computed at publish and re-verified on every
  `snapshot`; a mismatch raises naming the stamp instead of rendering
  garbage.
"""
from __future__ import annotations

import copy
import itertools
import threading
import time
from typing import List, Optional

import numpy as np

_LOCK = threading.Lock()
_SLOT: Optional[dict] = None
_SEQ = itertools.count(1)

_SUBJECT_KINDS = ("assembly", "single")


def _freeze_path(idx: int, p: dict) -> dict:
    """Validate ONE path dict and return an immutable copy. Refuses by
    name — a malformed path never enters the slot."""
    if not isinstance(p, dict) or "pts" not in p:
        raise ValueError(
            f"last_flight.publish: path {idx} is not a dict with 'pts' — "
            f"got {type(p).__name__}")
    pts = np.ascontiguousarray(np.asarray(p["pts"], dtype=float))
    if pts.ndim != 2 or pts.shape[1] != 3 or not len(pts):
        raise ValueError(
            f"last_flight.publish: path {idx} pts must be a non-empty "
            f"(N, 3) array, got shape {pts.shape}")
    if not np.all(np.isfinite(pts)):
        raise ValueError(
            f"last_flight.publish: path {idx} contains non-finite "
            f"coordinates — refusing to bank a corrupt trajectory")
    pts.setflags(write=False)
    out = {"pts": pts, "label": str(p.get("label", f"ion {idx}"))}
    if "t_us" in p and p["t_us"] is not None:
        t = np.ascontiguousarray(np.asarray(p["t_us"], dtype=float))
        if t.shape != (len(pts),):
            raise ValueError(
                f"last_flight.publish: path {idx} t_us length "
                f"{t.shape} does not match pts {len(pts)}")
        t.setflags(write=False)
        out["t_us"] = t
    return out


def _fingerprint(paths: List[dict]) -> dict:
    tot = int(sum(len(p["pts"]) for p in paths))
    csum = float(sum(float(p["pts"].sum()) for p in paths)) if paths else 0.0
    return {"n_paths": len(paths), "total_pts": tot,
            "coord_sum": f"{csum:.9e}"}


def publish(*, site: str, subject: dict, paths: List[dict],
            n_flown: int, note: str = "", m_amu: Optional[float] = None,
            detections: Optional[List] = None) -> dict:
    """Bank the most recent flight. Returns the banked record.

    subject: {"kind": "assembly", "doc": <instrument dict>} or
             {"kind": "single", "spec": <spec dict>} — SELF-CONTAINED,
    so the viewer always draws the geometry the flight actually flew,
    never whatever the dashboard holds now (staleness impossible by
    construction). paths may be EMPTY (a fly with trace storage off is
    still a flight; the viewer names the absence)."""
    global _SLOT
    if not site or not isinstance(site, str):
        raise ValueError("last_flight.publish: site must name the fly "
                         "path that produced this record")
    if (not isinstance(subject, dict)
            or subject.get("kind") not in _SUBJECT_KINDS):
        raise ValueError(
            f"last_flight.publish: subject.kind must be one of "
            f"{_SUBJECT_KINDS}, got {subject!r:.80}")
    key = "doc" if subject["kind"] == "assembly" else "spec"
    if not isinstance(subject.get(key), dict):
        raise ValueError(
            f"last_flight.publish: subject[{key!r}] must be the "
            f"{'instrument' if key == 'doc' else 'spec'} dict the "
            f"flight flew (self-contained record)")
    n_flown = int(n_flown)
    if n_flown < 0:
        raise ValueError("last_flight.publish: n_flown < 0")
    if m_amu is not None:
        m_amu = float(m_amu)
        if not (m_amu > 0 and np.isfinite(m_amu)):
            raise ValueError(f"last_flight.publish: m_amu must be a "
                             f"positive finite mass in amu, got {m_amu}")
    _det_frozen = None
    if detections:
        _det_frozen = []
        for _i, _d in enumerate(detections):
            _a = np.asarray(_d, float).reshape(-1)
            if _a.shape != (3,) or not np.all(np.isfinite(_a)):
                raise ValueError(
                    f"last_flight.publish: detections[{_i}] is not a "
                    f"finite (x,y,z) triple: {_d!r}")
            _det_frozen.append(tuple(_a.tolist()))
        _det_frozen = tuple(_det_frozen)
    frozen = [_freeze_path(i, p) for i, p in enumerate(paths or [])]
    record = {
        "stamp": next(_SEQ),
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "site": site,
        "subject": copy.deepcopy(subject),
        "paths": frozen,
        "n_flown": n_flown,
        "n_paths": len(frozen),
        "note": str(note),
        # KE coloring needs mass (0.5 m v^2); banked when the flight
        # declares ONE mass for the packet. None = mixed/unknown, and
        # the viewer REFUSES the KE scheme by name rather than guessing.
        "m_amu": m_amu,
        # registered detector crossings (x,y,z mm) — pass-through by
        # design; validated 3-vectors, frozen.
        "detections": _det_frozen,
        "fingerprint": _fingerprint(frozen),
    }
    with _LOCK:
        _SLOT = record          # atomic swap: complete record or nothing
    return record


def snapshot() -> Optional[dict]:
    """The banked record, integrity-verified, or None. A fingerprint
    mismatch RAISES naming the stamp — a corrupted slot is never
    rendered as though it were the flight."""
    with _LOCK:
        rec = _SLOT
    if rec is None:
        return None
    if _fingerprint(rec["paths"]) != rec["fingerprint"]:
        raise RuntimeError(
            f"last_flight slot CORRUPTED: record stamp {rec['stamp']} "
            f"({rec['site']}, {rec['when']}) no longer matches its "
            f"publish-time fingerprint — some code mutated the banked "
            f"flight in place. Refusing to render it; re-fly to re-bank.")
    return rec


def clear(reason: str) -> str:
    """Empty the slot. Returns a NAMED statement of what was cleared —
    clearing is a visible act, never a silent no-op."""
    global _SLOT
    if not reason:
        raise ValueError("last_flight.clear: a reason is required — "
                         "an unexplained wipe is a silent state change")
    with _LOCK:
        rec, _SLOT = _SLOT, None
    if rec is None:
        return f"last-flight slot was already empty ({reason})"
    return (f"cleared last-flight record stamp {rec['stamp']} "
            f"({rec['site']}, {rec['when']}, {rec['n_paths']} paths) — "
            f"{reason}")
