<!-- Generated from the ion_gym user manual and the shipped decks and notebooks. Edits made here are overwritten. -->

# 9. Reading results

A flight returns a trajectory array and a summary. The trajectory array has one row per recorded step; the column names come back from the build, so read them rather than assuming an order. Recorded per step are time, position, velocity, and the electric field components the ion actually experienced at that point. From these the interface derives further channels — speed, kinetic energy in eV, radial position, field magnitude — which can be plotted against one another for a single ion or a whole population.

The summary reports how the flight ended, and that outcome deserves attention: an ion that timed out is not the same as one that struck an electrode or left through a declared boundary, and a population statistic computed without separating those is usually meaningless.

The fate vocabulary is short and worth memorizing — every view, legend and statistic uses the same codes:

- 0 impact / exit — the ion struck metal (or left through a declared aperture in it);

- 1 boundary exit — it left the solved domain;

- 2 timeout — still in flight when the declared maximum flight time expired: not a loss, a question about your clock;

- 3 bounding plane — it crossed a declared terminating bound;

- 4 transporter max_passes — it reached a declared pass-count ceiling in a looping transporter;

- 5 station impact plane / 6 station detect — it ended on a declared station (section 7.5).

For a multi-stage instrument one more rule applies, and it is a definition rather than a computation: the only transmission figure is how many of the ions flown hit the detector. Everything else — hit metal, timed out, left the box — is a named reason an ion was not detected, and an instrument that declares no detector has flown ions but measured nothing (section 10).

**Recording is subsampled by a declared interval.** This decision is adjustable by the user and an option to save memory. Additionally, every recorded quantity carries its unit in the interface, and the units are those of section 7.4 — in particular the field channels are in V/mm, not V/m.
