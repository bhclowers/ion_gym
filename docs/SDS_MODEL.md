# SDS: statistical diffusion simulation

`collisions.model: "sds"` flies ions through a gas at pressures where individual collisions are too numerous to sample (roughly a Torr and above). It follows the method of Appelhans and Dahl (A. D. Appelhans, D. A. Dahl, *Int. J. Mass Spectrom.* **244** (2005) 1–14, Sections 2.1–2.5) and first-order kinetic theory. ion_gym's implementation (`ion_gym/physics/sds.py`) is written from the published equations. It is not a translation of any program's source code, and it ships no third-party data.

For a deck that resolves individual collisions, use `model: "hs"` (hard-sphere). The two models answer different questions, and `model: "hs"` is the reference when a deck sits between regimes.

## What happens each time step

1. **Viscous drift.** The ion's velocity relaxes toward the mobility-limited drift velocity `K_z E` (plus any declared gas flow) at the rate `gamma = |z| e / (m K_z)`. The relaxation is integrated exactly over the step, so the step only has to resolve the field, not the relaxation time.
2. **Diffusion.** The ion jumps a random distance in a uniformly random direction. The *shape* of the jump-length distribution comes from tabulated hard-sphere random walks at the ion/gas mass ratio (`sds_jump_icdf.dat`, ion_gym's own Monte Carlo of the paper's Section 2.5 procedure). The *size* is set by the Einstein relation (below).

## The mobility an ion flies with

The reduced mobility `K0` (cm²/(V s), referred to 273.15 K and 760 Torr, singly charged) is resolved per ion mass, in this order:

| Deck states | K0 used |
|---|---|
| a `collisions.sds_species` entry with `K0_cm2_per_Vs` measured in the deck's gas | that value, as given |
| an entry whose `K0_cm2_per_Vs` was measured in another gas | converted to the deck's gas (see *Gas conversion*); a WARNING advisory names every conversion |
| an entry with only `d_nm` (the ion diameter) | from the paper's air fit at that diameter, converted to the deck's gas |
| no entry for the mass | estimated from the mass: diameter `d = 0.120415405 m^(1/3)` nm (constant density), then the paper's air fit, then converted to the deck's gas |

The mobility at the deck's conditions is `K1 = K0 (760 / P_torr) (T / 273.15)`. Every advisory run (`SimSpec.advisories()`, the console and the app's status line) lists the `K0` each species flies and where it came from.

An entry looks like this:

```json
"collisions": {
  "enabled": true, "model": "sds", "gas": "N2", "T_k": 300, "P_torr": 760,
  "gas_diam_nm": 0.366,
  "sds_species": [
    {"mass_amu": 133, "gas": "N2", "charge": 1, "K0_cm2_per_Vs": 2.31},
    {"mass_amu": 622, "d_nm": 1.1},
    {"mass_amu": 922, "gas": "He", "charge": 1, "K0_cm2_per_Vs": 6.0,
     "gas_d_nm": 0.218},
    {"mass_amu": 1490, "gas": "N2", "charge": 2, "K0_cm2_per_Vs": 1.5}
  ]
}
```

Entries are matched by mass. A K0 is the measured mobility of the ion at the `charge` it was measured at, so it must state that charge. An ion of that charge flies the value as given. An ion of the same mass at another charge flies it scaled by `|z_ion| / |z_measured|` (see *Charge*). Internally a declared K0 is referred to unit charge, `K0 / |z_measured|`, and the advisory lists that value with its source.

Malformed entries are refused by name. That covers:
- no K0 and no diameter;
- a non-positive or non-finite value;
- an unknown key or gas;
- a K0 without its gas or its charge;
- a K0 from another gas without that gas's diameter;
- a mass listed twice.

No mobility table ships with ion_gym. Undeclared species are estimated, and the estimate is only as good as the constant-density assumption and the air fits. The paper's own validation used measured values, and declared values should be used wherever they exist.

## Charge

The paper's mobility formula is written for singly charged ions. ion_gym extends it as follows:

- The collision cross section is taken from the ion's **mass** (through the singly charged relations above), and the mobility scales with charge: **`K_z = |z| K1(m)`**. This is the Mason–Schamp form at a fixed cross section.
- The drift is along the force, so the **sign of z sets the direction**.
- The relaxation rate is `gamma = |z| e / (m K_z) = e / (m K1)`, the same for every charge.

**Assumption and its consequence.** Holding the cross section fixed neglects two effects. The ion-induced-dipole attraction grows with charge, which enlarges the cross section, and highly charged ions can change shape. So `K_z` is an **upper estimate** for multiply charged ions when it is scaled from a singly charged value or estimate. The error is largest for small, compact ions in polarisable gases, and it is zero for `|z| = 1`. A `K0` measured for the charge state in question, declared with that `charge`, removes it: the ion flies the measured value, and its diffusion follows from the same value through the Einstein relation.

## Diffusion

The diffusion coefficient is the Einstein (Nernst–Townsend) relation at the mobility in use:

**`D = K_z k T / (|z| e) = K1 k T / e`**, the same for every charge.

The jump scale is set so that the mean squared 3-D jump per step is exactly `6 D dt` for the tabulated jump statistics at the ion's mass ratio. Every route jumps in three dimensions. The planar route's field is 2-D, but its ions move along z as well, so each axis receives `2 D dt` there as on r-z and 3-D.

**Why not the paper's jump scale.** The paper scales jumps by a hard-sphere mean free path computed from the ion diameter. That diffusion is set by the diameter, independently of the mobility, so the two can disagree. For Cs⁺ with the paper's measured `K0` values, a hard-sphere mean-free-path jump scale gives a diffusion coefficient 1.49–1.69 times the Einstein value. The Einstein relation holds for any ion–neutral interaction in the low-field limit, which is the regime SDS assumes.

**Consequence.** For the same mobility, ion_gym's SDS spreads a packet less than a mean-free-path-scaled SDS, by the square root of that ratio: about 1.2–1.3 times narrower for Cs⁺ from diffusion alone. Widths dominated by an aperture, by space charge or by field gradients are affected less.

## Gas conversion

A `K0` is converted between gases at the ion's diameter with the paper's Mason–Schamp hard-sphere ratio:

`K0(gas) / K0(ref) = sqrt(mu_ref / mu_gas) (d_ion + d_ref)² / (d_ion + d_gas)²`

The gas diameter is `collisions.gas_diam_nm` for the deck's gas and `gas_d_nm` for the gas a species value was measured in. The air reference uses the paper's values: 28.954515 amu (dry air is 28.965 amu; the paper's air fits were made with 28.954515) and 0.366 nm. The default `gas_diam_nm` is air's 0.366 nm. A deck in another gas should state its own.

## Relation to Appelhans and Dahl

**Kept as published:**
- viscous drift at the mobility, and `K0` referred to 273.15 K and 760 Torr;
- the constant-density diameter estimate, and both air `K0`–diameter fits with their coefficients;
- the hard-sphere Mason–Schamp conversion from air to another gas, and the order in which declared and estimated quantities combine;
- the jump statistics: regenerated by ion_gym with the paper's procedure (five mass ratios from 1 to 10⁴, 100,000 ions through 100,000 collisions each) and interpolated in log mass ratio as the paper describes. `tools/regen_sds_tables.py` reproduces the shipped table with `--flux-weighted` (the gas partner is weighted by relative speed, a choice the paper leaves open) at the seed recorded in the table's header.

**Changed:**

| Element | Paper | ion_gym | Why | Consequence |
|---|---|---|---|---|
| Charge | Mobility written with `e`; charge does not enter | `K_z = \|z\| K1(m)`, drift along the force | First-order Mason–Schamp mobility is proportional to charge at fixed cross section; without it a `2e` ion drifts as if singly charged | Multiply charged ions drift `\|z\|` times faster than a singly charged ion of the same mass; an upper estimate (cross-section growth with charge neglected) |
| Diffusion scale | Jumps scaled by the hard-sphere mean free path from the ion diameter | Jumps scaled to the Einstein `D = K1 k T / e` | Einstein holds for any interaction at low field, the regime the paper's experiments were kept in. A diameter-based scale can disagree with an independently supplied mobility | For Cs⁺ at the paper's `K0` values, `D` is 0.59–0.67 of the mean-free-path value, and diffusive widths are 1.22–1.30 times narrower |
| Species data | User values assumed correct for whatever gas is simulated, at whatever charge | No table ships; declared values carry the gas and charge state they were measured at, are converted between gases with a warning, and are scaled only for a different charge | A single table applied to every gas uses each value outside the gas it was measured in; a measured mobility already contains its charge | A mobility flies unchanged only in its own gas and charge state; undeclared species are estimated |
| Drift integration | Damping applied to the acceleration; formula not given | Relaxation integrated exactly over the step | Removes the need to resolve `1/gamma` (0.401 ns for a 322 u ion in N₂ at 760 Torr) | Step length set by field and geometry; an implementation choice where the paper gives none |

The paper's agreement between simulated and measured beam widths (its Figure 13) includes SIMION's coulombic space-charge model and initial currents estimated from the measurements. It does not isolate the diffusion term, so it neither supports nor contradicts either diffusion scale.

## Limits

- **Low field.** The mobility and the diffusion coefficient do not depend on the field. High-field mobility, ion heating and longitudinal/transverse diffusion anisotropy are not modelled.
- **Relaxation within a step.** SDS assumes the drift relaxes faster than the field changes. RF confinement needs the opposite (`gamma << Omega`), so an SDS deck with RF warns when `gamma / Omega` falls below 1 and names the pressure that would reach it. Use `model: "hs"` there.
- **Uniform gas over a step.** Pressure, temperature and flow are taken at the ion's position each step.
- **No collision count.** SDS averages collisions into the two terms above, so the `n_col` channel is NaN under SDS.
- **Neutral and light ions.** `z = 0` drifts nowhere and diffuses with `K1 k T / e`. An ion lighter than the gas uses the jump statistics extrapolated below a mass ratio of 1.

## Where it is tested

- `tests/test_sds_transport.py`:
  - the per-ion numbers scale with charge as stated;
  - the jump scale gives `6 D dt`;
  - in flight on planar, r-z and 3-D, the drift velocity divided by the recorded field equals `K_z` with the sign of z (z = +1, +2, −2, +3);
  - the spread of 400 ions with no field matches `6 D t` for z = 1 and 2.
- `tests/test_sds_species.py`:
  - declared values fly as given on the three routes;
  - cross-gas conversion and its warning;
  - the diameter-only and estimated paths;
  - malformed entries are refused at every route's entry point.
- `tests/test_sds_regen_tables.py`: the shipped jump statistics are sane and no mobility table ships.
- `tests/test_regen_sds_tables_tool.py`: the regeneration tool's random walk is pinned, a small run writes a table the runtime reads, and a rejection sampler that runs out of tries stops the run by name.
- `tests/test_sds_vs_hs.py` (slow, `pytest -m slow`): SDS against hard-sphere collisions at the same cross section, on planar, r-z and 3-D. Hard-sphere collisions produce mobility and diffusion from first principles. Given the first-order Mason–Schamp mobility for that cross section:
  - SDS drift matches hard-sphere for z = 1 and 3;
  - SDS diffusion matches hard-sphere with no field;
  - both within max(5 %, 3 standard errors).

  In the full comparison (322 u in N₂, 2 Torr, 400 ions):
  - hard-sphere mobility scales with charge as `K_z = |z| K1` predicts;
  - SDS lies within 0.986–1.009 of the hard-sphere mobility for z = 1, 2, 3 and within 1.042–1.051 of its diffusion coefficient, all within one standard error.
