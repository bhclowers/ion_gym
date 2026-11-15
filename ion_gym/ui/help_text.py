"""Per-tab help notes for the dashboard's control tabs.

One short note per tab: what the tab is for, where it sits in the
workflow, and the one or two things a first-time user is most likely to
miss. Widget-level detail stays in each widget's own tooltip and in the
user manual's tab-by-tab reference; repeating it here would make two
copies to keep in step.

`TAB_HELP` is keyed by the tab title exactly as it appears in the rail.
`help_for` refuses an unknown title rather than showing an empty note, so
a renamed tab without a note fails loudly in the layout smoke test.
"""

TAB_HELP = {
    "Load": (
        "**Load** is where a session starts.\n\n"
        "- **Deck > Example** stages a shipped example. Its cost card "
        "appears before you commit, and *Load example* applies it.\n"
        "- **Deck > User JSON** takes a deck of your own. The JSON box "
        "always holds the live specification, so edits elsewhere show up "
        "there too.\n"
        "- **Deck > Multi FA** loads a staged instrument, where each stage "
        "solves in its own grid and the ion is handed across a declared "
        "seam.\n"
        "- **Save** writes the loaded specification (or the whole "
        "instrument) to a file.\n"
        "- **Runs** keeps the flights of this session, each reloadable "
        "with the specification that flew it, plus saved fields and "
        "trajectories."),
    "Ion Source": (
        "**Ion Source** declares what is flown and where it is counted.\n\n"
        "- **Ions:** masses in Da, the charge state, how many per mass, "
        "and the birth distribution (box or Gaussian), kinetic energy "
        "range and launch direction.\n"
        "- **Beam:** a beam declared by propagation axis, or one derived "
        "from the packet the last flight produced.\n"
        "- **Stations:** detector and aperture planes. A `detect` station "
        "with *on hit = pass* counts arrivals without stopping the ion, and "
        "its window limits which arrivals count. Impact Analysis reads "
        "these stations.\n\n"
        "A *seeded run* repeats exactly. An unseeded run draws a fresh "
        "seed and prints it, so any run can be repeated."),
    "Voltages": (
        "**Voltages** sets every potential on the electrodes.\n\n"
        "- **Drive groups** define a waveform once (amplitude, frequency, "
        "phase, duty, offset) and list the electrodes it drives.\n"
        "- The **travelling-wave builder** assigns phases along an "
        "ordered ladder of electrodes.\n"
        "- **DC groups** feed a chain from both ends and interpolate the "
        "members by their number.\n\n"
        "Changing a voltage never needs a new field solve: each electrode "
        "is solved once at 1 V and the field is rebuilt by superposition."),
    "Physics": (
        "**Physics** holds what governs the flight itself.\n\n"
        "- **Gas:** collisions on or off, the collision model, buffer "
        "gas, temperature and pressure, and the cross-section source.\n"
        "- **Integration:** time step, maximum flight time and how much "
        "of each path is recorded. The red *dt too coarse* note in the "
        "top bar names the step the gas and RF physics call for.\n"
        "- **Bounds:** optional planes that end a flight. An open edge "
        "of the solved domain is otherwise the boundary."),
    "Display": (
        "**Display** changes only what is drawn, never what is "
        "computed.\n\n"
        "- **Field overlay:** view plane, equipotential lines (coloured "
        "by potential), and optional shading of |E| or the potential "
        "energy.\n"
        "- **Trajectories:** style, width, opacity, how many paths are "
        "drawn, and color. The default colors each path by its mass, "
        "with a legend. Fate, one solid color or a recorded channel such "
        "as kinetic energy are the other choices. A channel is drawn on "
        "the plasma scale, and its range is stated on the status line "
        "rather than in a color bar.\n"
        "- **Electrodes:** fills, labels, colour and transparency.\n\n"
        "*Show field (no ions)* draws the solved field before any flight."),
    "Impact Analysis": (
        "**Impact Analysis** reads where ions ended up.\n\n"
        "Pick the run, the fate that counts as transmitted, and a "
        "surface: a station, a boundary face or a cross-section plane. "
        "The histogram and scatter show the arrivals on that surface. A "
        "deck with no station shows only boundary faces. Add a `detect` "
        "station on the Ion Source tab to count arrivals inside the "
        "domain."),
    "Geometry Import": (
        "**Geometry Import** brings electrodes in from STL meshes.\n\n"
        "Set the pitch (mm per grid unit), preview the voxelization at a "
        "coarse pitch first, then *Commit geometry*. A 2-D mid-plane "
        "slice solves quickly, and *solve full 3-D* solves the full "
        "volume."),
    "Config": (
        "**Config** holds properties of this installation and session "
        "only. Nothing here travels with a deck or changes the physics.\n\n"
        "- **Compute Resources:** worker threads and the memory allowed "
        "for recorded trajectories.\n"
        "- **Session diagnostics:** a ping to the server console and a "
        "periodic memory log, for checking that the session is "
        "healthy."),
}


def help_for(tab_title):
    """The help note for a control tab, by its title in the rail.

    Raises KeyError naming the title and the titles that do have notes
    when the tab has none, so a new or renamed tab cannot ship with a
    blank help panel."""
    try:
        return TAB_HELP[tab_title]
    except KeyError as e:
        raise KeyError(
            f"no help note for control tab {tab_title!r}; notes exist for "
            f"{sorted(TAB_HELP)}") from e
