<!-- Generated from the ion_gym user manual and the shipped decks and notebooks. Edits made here are overwritten. -->

# 11. When the toolkit refuses: reading the guards

A consistent design choice runs through everything above: where a configuration cannot mean what it says, the toolkit refuses by name rather than flying something plausible. A refusal is not a crash — it is the tool telling you which declaration to change, usually with the arithmetic in hand. The ones you will actually meet:

**The trajectory record would not fit.** Before anything is built, the record's size is quoted exactly — ions × samples × channels × 8 bytes. An oversized run refuses; a merely large one warns and flies. The ceiling is this machine's memory, or the budget you declare (max record RAM on the Config tab; scripts pass record_budget_gb). The cheapest lever is almost always the recording interval: on a single-stage flight it is a storage decision, not a physics one — time of flight, fates, impact positions and collision counts are identical at any recording density.

**The recording cannot cover the flight.** With the per-ion record cap too low for the declared flight time, a single-stage run announces truncation — the stored trace stops, the flight continues to its true fate, and the summary reports that fate. A staged flight refuses instead, because its arrival times are read off the recorded trace: a crossing after the buffer filled would be invisible, and the ion falsely reported as never detected.

**The solve is priced before it runs.** Loading an example or changing resolution quotes the node count and the expected solve time for this machine before you commit, and the smallest-feature field warns when the pitch cannot resolve the gap you care about (less than about three cells across it).

**A station's window names its own axis, or its pair is inverted.** Both refuse: a plane cannot be windowed along its own normal, and a backwards (lo, hi) pair would silently turn a wall into a pass-through.

**A declared bound sits inside its stage's exit seam** (assemblies): refused as a dead handoff — section 10.

**The ballistic extension's premise fails** (section 7.6): background gas is declared, or the escape face carries residual field above the stated ceiling. The message quotes the measured field; the remedy is padding the domain, not raising the ceiling.

**An ion is born inside metal.** That ion splats at birth with a warning and the rest of the packet flies. A configuration error — an unknown gas, a malformed station, a mis-declared seam — aborts the run instead: configuration mistakes must never be tallied as transmission losses.

**The solver did not converge.** The iteration cap was reached above tolerance, and the field is not returned. An unconverged field that looks plausible is the most dangerous object in this toolkit, which is why you get an error instead of a result.

**The mesh on disk is not the mesh the specification hashed.** The STL preflight refuses on a SHA-256 mismatch (section 4.2): a re-exported part is a new geometry and must look like one.

**Advisories are the softer form** — printed once per run, never blocking: the point-source note, the Δt-bias and SDS-regime notes (section 7.3), the seam residual fields (section 10). An advisory you have read and can argue with is fine to fly past; one you cannot explain is the simulation telling you it is about to answer a different question than you asked.
