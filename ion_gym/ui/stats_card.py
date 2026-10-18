"""
stats_card.py — the Impact-statistics Card (Panel) for the app.

MOVED HERE from physics/stats.py (L-548, D6a, 2026-10-15): a Panel widget
does not belong in a physics module. The reductions it renders stay in
physics/stats.py — compute_stats, stats_markdown, stations_markdown,
delivered_spread_text — so the card, the gate scripts and the notebooks
still read ONE authority and cannot disagree. This module only lays them
out; it computes nothing. The code below moved verbatim.
"""
from ion_gym.physics.stats import (
    ALL_PLANES, FATE_NAME, auto_transmitted_fate, compute_stats,
    delivered_spread_text, enabled_planes, mz_of_results, stations_markdown,
    stats_markdown)

_AUTO = "auto (bounds→3 else 2)"
_SRC_PLANES = "standard planes"
_SRC_STATIONS = "detector stations"


def stats_card(*, title="Impact statistics", collapsed=False):
    """Collapsible Card: markdown body + transmitted-fate selector.

    Wire-up (sim_app):
        self.stats = stats_card()                      # once, near self.pane
        ... pn.Column(self.pane, self.stats.card) ...  # in panel()
        self.stats.update(results, self._spec_for_stats())   # end of _redraw
    """
    import panel as pn

    class _Card:
        def __init__(self):
            self.md = pn.pane.Markdown("_run an ensemble to see statistics_",
                                       sizing_mode="stretch_width")
            self.sel = pn.widgets.Select(
                name="transmitted fate", value=_AUTO,
                options=[_AUTO] + [f"{k} · {v}" for k, v in FATE_NAME.items()],
                width=220)
            # IMPACT-PLANE selector: pick which declared
            # plane the landing stats are shown for. Options are filled
            # from the spec's enabled planes at update time; "all planes"
            # shows every plane's (correctly per-plane grouped) row.
            self.plane_sel = pn.widgets.Select(
                name="impact plane", value=ALL_PLANES,
                options=[ALL_PLANES], width=220)
            # SOURCE SWITCH: the card used to CONCATENATE
            # the station table and the plane table, so two different
            # groupings of the same flight were stacked with nothing saying
            # they were different questions. Now one is chosen. The station
            # option only becomes selectable when the flight actually
            # declared stations -- offering an empty view would read as
            # "no ions arrived" when the truth is "no detector declared".
            self.source_sel = pn.widgets.Select(
                name="statistics for", value=_SRC_PLANES,
                options=[_SRC_PLANES], width=220)
            # THE CARD HAS A SKELETON AND A SLOT: selectors are
            # PERMANENT; only `_body` swaps between the markdown table and
            # the per-m/z Tabs. The count-bar figure that briefly lived
            # here (per-plane / per-station hit counts) was REMOVED as
            # answering a question nobody was asking
            # -- position-style impact plots for
            # every surface live in the Impact Analysis tab's unified
            # surface dropdown instead. viz_core.stats_figure remains a
            # framework capability; this card just no longer draws it.
            self._body = pn.Column(self.md, sizing_mode="stretch_width")
            self.card = pn.Card(pn.Column(pn.Row(self.source_sel, self.sel),
                                          pn.Row(self.plane_sel),
                                          self._body),
                                title=title, collapsed=collapsed,
                                sizing_mode="stretch_width")
            self._last = None    # (results, spec, stations) for re-render
            self.sel.param.watch(lambda e: self._rerender(), "value")
            self.plane_sel.param.watch(lambda e: self._rerender(), "value")
            self.source_sel.param.watch(lambda e: self._rerender(), "value")

        def _fate(self, spec):
            if self.sel.value == _AUTO:
                return auto_transmitted_fate(spec)
            return int(self.sel.value.split("·")[0])

        def update(self, results, spec=None, stations=None):
            """stations: rows from viz_core.station_stats, or None.

            Optional so every existing caller keeps working; when
            present the card leads with a per-station table. Same
            declared StationSpec the figure draws from, so the panel
            and the plot cannot disagree (the station windows are
            exposed to the stats module for exactly this reason)."""
            self._last = (list(results), spec, list(stations or []))
            self._rerender()

        def _rerender(self):
            if not self._last:
                return
            results, spec, stations = self._last
            # DELIVERED SPREAD: one authority line above whichever
            # table renders; failures REPORT rather than blank the card.
            self._spread_line = ""
            if spec is not None:
                try:
                    self._spread_line = delivered_spread_text(
                        spec, markdown=True) + "\n\n"
                except (ValueError, AttributeError, TypeError, IndexError) as e:
                    # delivered_spread_cells can raise ValueError (no births),
                    # AttributeError (missing symmetry), TypeError/IndexError
                    # from numpy operations. Report, never silent.
                    self._spread_line = (f"_delivered spread unavailable: "
                                         f"{e}_\n\n")
            mz = mz_of_results(spec, len(results)) if spec is not None else None
            fate = self._fate(spec)

            # ONE REPORT PER m/z. A mixed ensemble pooled into a single table
            # is a category error: transmission, landing spot and KE are all
            # mass-dependent (that is the entire point of an m/z filter), so
            # the pooled centroid is a number no ion actually has. The pooled
            # view is kept as an "all" tab for the ensemble-level counts.
            groups = {}
            if mz is not None:
                for i, m in enumerate(mz):
                    if i < len(results):
                        groups.setdefault(float(m), []).append(results[i])

            planes = enabled_planes(spec)
            # keep the plane selector in sync with the spec's declared
            # planes (B); preserve the current choice when still valid.
            opts = [ALL_PLANES] + [lbl for lbl, _, _ in planes]
            if self.plane_sel.options != opts:
                cur = self.plane_sel.value
                self.plane_sel.options = opts
                self.plane_sel.value = cur if cur in opts else ALL_PLANES
            pf = self.plane_sel.value
            # keep the SOURCE options honest about what this flight has
            _opts = [_SRC_PLANES] + ([_SRC_STATIONS] if stations else [])
            if self.source_sel.options != _opts:
                _cur = self.source_sel.value
                self.source_sel.options = _opts
                self.source_sel.value = (_cur if _cur in _opts
                                         else _SRC_PLANES)
            _src = self.source_sel.value
            _stations_view = (_src == _SRC_STATIONS) and bool(stations)
            # the plane filter is meaningless under the station view
            self.plane_sel.visible = not _stations_view
            if _stations_view:
                self.md.object = (self._spread_line
                                  + stations_markdown(stations))
                self._set_body(self.md)
                return
            if len(groups) <= 1:
                st = compute_stats(results, transmitted_fate=fate, mz=mz,
                                   planes=planes)
                self.md.object = (self._spread_line
                                  + stats_markdown(st, plane_filter=pf))
                self._set_body(self.md)
                return

            tabs = []
            st_all = compute_stats(results, transmitted_fate=fate, mz=mz,
                                   planes=planes)
            tabs.append(("all", pn.pane.Markdown(
                stats_markdown(st_all, plane_filter=pf),
                sizing_mode="stretch_width")))
            for m in sorted(groups):
                rows = groups[m]
                st_m = compute_stats(rows, transmitted_fate=fate,
                                     mz=[m] * len(rows), planes=planes)
                tabs.append((f"m/z {m:g}", pn.pane.Markdown(
                    stats_markdown(st_m, plane_filter=pf),
                    sizing_mode="stretch_width")))
            self._set_body(pn.Column(
                pn.pane.Markdown(self._spread_line,
                                 sizing_mode="stretch_width"),
                pn.Tabs(*tabs, sizing_mode="stretch_width"),
                sizing_mode="stretch_width"))

        def _set_body(self, obj):
            """Swap ONLY the body slot (table <-> per-m/z tabs). Never the
            card itself: the selectors and the figure are permanent
            structure, and replacing card.objects wholesale is what made
            the impacts graph disappear on the first post-flight update."""
            if len(self._body.objects) and self._body.objects[0] is obj:
                return
            self._body.objects = [obj]

        def clear(self):
            self._last = None
            self.md.object = "_run an ensemble to see statistics_"
            self._set_body(self.md)

    return _Card()
