"""
ion_gym.ui.widget_options — the options/value invariant for Panel selectors
===========================================================================

ONE rule, in ONE place: after a selector's `options` are replaced, its
`value` is one of those options.

WHY THIS MODULE EXISTS. `pn.widgets.Select` does NOT re-point `value`
when `options` is reassigned. A value that is no longer in the list just
stays there, and the widget sits in a state its own options contradict::

    s = Select(options=["(refresh first)"])   # value '(refresh first)'
    s.options = ["a", "b"]                    # value STILL '(refresh first)'

    s = Select(options=[])                    # value None
    s.options = {"label": "/path.npz"}        # value STILL None

The browser renders the first real option as though it were chosen, so
the user sees a selection the server does not have — the displayed-is-not-
computed class, at the widget layer. Every control that reads
`widget.value` then behaves as if nothing is selected. Measured in v523:
the Cache tab's export button stayed disabled reading "nothing selected"
after its own refresh, "remove selected" answered "nothing to remove
(refresh first)" immediately after a refresh, and the Fields tab's "Load
field" answered "no field selected — pick one (↻ to rescan)" with a
matching field listed in the picker. Three dead controls, one cause.

Hand-repairing each call site is what produced the split in the first
place: of the app's option assignments, some restored the value, some
restored it only when it was still valid (leaving the invalid case — the
one that matters — unhandled), and some did nothing. So the rule lives
here and every site calls it, rather than each site remembering.

SELECTION POLICY (PI ruling 2026-09-14): keep the current selection when
it survives into the new options; otherwise take the first entry. NOT
"always reset to first" — the Cache tab's picker also drives "✕ remove
selected", and a refresh that silently re-points a destructive-adjacent
control at a different entry than the one the user was reading is the
dangerous direction. `prefer` overrides both, for the callers that have
just created the entry they want selected.
"""

from __future__ import annotations

import param


def _selector_classes():
    """The Panel widget classes this module knows how to re-point.

    Named explicitly rather than duck-typed on `hasattr(w, "options")`:
    `AutocompleteInput` also carries `options`, but it is a free-text
    field whose empty value is legitimate, and forcing it to
    `options[0]` would invent a selection the user never made. An
    unlisted class is refused by name in `set_options` instead of being
    guessed at. Resolved by getattr so a Panel release that drops or
    renames one of these narrows the allowlist rather than breaking
    import.
    """
    import panel as pn
    names = ("Select", "MultiSelect", "MultiChoice", "CheckBoxGroup",
             "CheckButtonGroup", "RadioBoxGroup", "RadioButtonGroup",
             "CrossSelector", "ToggleGroup")
    return tuple(c for c in (getattr(pn.widgets, n, None) for n in names)
                 if isinstance(c, type))


def selectable_values(options):
    """The values a selector will actually hold for these `options`.

    Panel accepts a list (the entries ARE the values) or a dict (the keys
    are labels, the VALUES are what `widget.value` becomes). Getting this
    backwards is the easy mistake — the Fields tab picker is a dict of
    display label -> file path, so its value is a path, never a label.
    """
    if isinstance(options, dict):
        return list(options.values())
    return list(options)


def set_options(widget, options, prefer=None):
    """Replace `widget.options` and leave `widget.value` valid.

    Returns the resulting value (a list for multi-value selectors).

    prefer: a value to select if it is present in the new options. Beats
        the surviving current selection, for callers that just produced
        the entry they want shown (a freshly loaded run, a newly created
        station). Ignored when absent from the options rather than
        silently forced — a caller asking for something the list does not
        contain is asking for a state the widget cannot represent.

    `value` is assigned ONLY when it actually changes, so watchers fire on
    real selection changes and not on every repopulation. Where a value
    does change, that is because the old one was not in the new options:
    the widget was already in a state its options contradicted, and a
    watcher firing on the repair is the correct consequence, not a
    side effect to suppress.
    """
    classes = _selector_classes()
    if not isinstance(widget, classes):
        raise TypeError(
            f"set_options: {type(widget).__name__} is not a selector this "
            f"module re-points (known: "
            f"{', '.join(sorted(c.__name__ for c in classes))}). Add it to "
            f"_selector_classes() once its value/options semantics are "
            f"established — do not route a free-text widget through here.")

    vals = selectable_values(options)
    is_multi = isinstance(widget.param["value"], param.List)
    current = widget.value
    widget.options = options

    if is_multi:
        kept = [v for v in (current or []) if v in vals]
        if kept != list(current or []):
            widget.value = kept
        return widget.value

    if prefer is not None and prefer in vals:
        chosen = prefer
    elif current is not None and current in vals:
        chosen = current
    elif vals:
        chosen = vals[0]
    else:
        chosen = None
    if widget.value != chosen:
        widget.value = chosen
    return widget.value
