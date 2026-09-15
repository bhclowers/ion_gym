"""Quote-then-run and progress reporting for long notebook cells.

The "long jobs report progress" convention and the
quote-then-run rule say every cell that can run for more than a few
seconds must (a) print what it is about to do and what it will cost
BEFORE spending it, and (b) emit a counter as it goes. Both were being
honored ad hoc -- per-item lines in some cells, no up-front quote, no
"i of N" -- so this module makes that shape the default and the
same in every notebook.

Import-clean: no side effects, no heavy dependencies.

    from ion_gym.progress import quote, track

    quote("Leg 0a drift ladder", len(rungs), per_item_s=42,
          detail="stability gate + re-null + flown audit per rung")
    for dh in track(rungs, "drift ladder"):
        ...
"""
from __future__ import annotations

import sys
import time


def _hms(seconds):
    """Human duration; None when the caller has no estimate to give."""
    if seconds is None:
        return "unknown"
    s = float(seconds)
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s/60:.1f} min"
    return f"{s/3600:.1f} h"


def quote(what, n_items=None, *, per_item_s=None, total_s=None,
          detail=None, stream=None):
    """Print the cost of a job BEFORE it runs, and return the estimate.

    The reader must be able to abort before spending the time rather
    than after. Give either `per_item_s` (with `n_items`) or `total_s`;
    passing neither prints an explicit "unquoted" line rather than a
    fabricated number -- a made-up estimate is worse than none.
    """
    out = stream or sys.stdout
    est = total_s if total_s is not None else (
        None if (per_item_s is None or n_items is None)
        else float(per_item_s) * int(n_items))
    head = f"[quote] {what}"
    if n_items is not None:
        head += f": {int(n_items)} item(s)"
        if per_item_s is not None:
            head += f" x ~{float(per_item_s):.3g} s"
    if est is None:
        head += " -- COST NOT QUOTED (no per-item or total estimate "
        head += "supplied; measure one item first)"
    else:
        head += f" -> ~{_hms(est)}"
    print(head, file=out, flush=True)
    if detail:
        print(f"         {detail}", file=out, flush=True)
    return est


class track:
    """Iterate, printing `[i/N] label ... elapsed, eta` per item.

    A printed line per item with flush is the acceptable
    minimum and works headless, in nbclient, and in a terminal -- unlike
    a redrawing bar, which leaves a mangled log in an executed notebook.
    The ETA comes from measured items only; before the first completes
    it reports unknown rather than guessing.

    A skipped item still announces itself: call `.skip(reason)` inside
    the loop, because a resumable job that prints nothing on a skip is
    indistinguishable from a stall.
    """

    def __init__(self, iterable, label="", *, stream=None, every=1):
        self._items = list(iterable)
        self.label = label
        self.n = len(self._items)
        self._out = stream or sys.stdout
        self._every = max(1, int(every))
        self.t0 = None
        self.i = 0

    def __len__(self):
        return self.n

    def __iter__(self):
        self.t0 = time.time()
        for i, item in enumerate(self._items, 1):
            self.i = i
            yield item
            self._report(i, item)

    def _report(self, i, item):
        if i % self._every and i != self.n:
            return
        el = time.time() - self.t0
        eta = (el / i) * (self.n - i) if i else None
        tag = f"{self.label} " if self.label else ""
        print(f"  [{i}/{self.n}] {tag}done ({item!r}) — "
              f"elapsed {_hms(el)}, eta {_hms(eta) if i else 'unknown'}",
              file=self._out, flush=True)

    def skip(self, reason, item=None):
        """Announce a skipped/resumed item; never let it pass silently."""
        what = f" ({item!r})" if item is not None else ""
        print(f"  [{self.i}/{self.n}] {self.label} SKIPPED{what}: "
              f"{reason}", file=self._out, flush=True)
