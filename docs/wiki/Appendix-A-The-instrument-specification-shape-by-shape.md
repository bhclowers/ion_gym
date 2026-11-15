<!-- Generated from the ion_gym user manual and the shipped decks and notebooks. Edits made here are overwritten. -->

# Appendix A. The instrument specification, shape by shape

Section 3 lists the top-level blocks of a specification; this appendix goes one level down, to the part you actually author when you build an instrument: the electrodes and their shapes. Every fragment here is taken from a deck that ships in examples/ and loads through the same SimSpec.from_json a user's own file takes.

## A.1 The skeleton

An electrode is a named conductor with a list of shapes, a DC value, and optional drive-group memberships. The geometry block holds the electrodes plus the grid declaration (extents, pitch, symmetry, and — for signed frames — origin_mm):

The geometry block below is complete and valid as written — it is the analytical quadrupole assembled in full in A.5, which builds, solves, and flies exactly as printed:

    "geometry": {
      "width_mm": 21.8,
      "height_mm": 21.8,
      "depth_mm": 0.0,
      "mm_per_gu": 0.1,
      "symmetry": {
        "coords": "xyz",
        "planes": {
          "x": "none",
          "y": "none",
          "z": "none"
        }
      },
      "electrodes": [
        {
          "name": "rod_east",
          "dc": 0.0,
          "rf_groups": [
            "RFA"
          ],
          "shapes": [
            {
              "type": "ellipse",
              "cx_mm": 17.344,
              "cy_mm": 10.9,
              "rx_mm": 3.444,
              "ry_mm": 3.444
            }
          ]
        },
        {
          "name": "rod_west",
          "dc": 0.0,
          "rf_groups": [
            "RFA"
          ],
          "shapes": [
            {
              "type": "ellipse",
              "cx_mm": 4.456,
              "cy_mm": 10.9,
              "rx_mm": 3.444,
              "ry_mm": 3.444
            }
          ]
        },
        {
          "name": "rod_north",
          "dc": 0.0,
          "rf_groups": [
            "RFB"
          ],
          "shapes": [
            {
              "type": "ellipse",
              "cx_mm": 10.9,
              "cy_mm": 17.344,
              "rx_mm": 3.444,
              "ry_mm": 3.444
            }
          ]
        },
        {
          "name": "rod_south",
          "dc": 0.0,
          "rf_groups": [
            "RFB"
          ],
          "shapes": [
            {
              "type": "ellipse",
              "cx_mm": 10.9,
              "cy_mm": 4.456,
              "rx_mm": 3.444,
              "ry_mm": 3.444
            }
          ]
        }
      ],
      "rf_groups": [
        {
          "name": "RFA",
          "frequency_hz": 1000000.0,
          "amplitude_v": 30.0,
          "phase_deg": 0.0,
          "waveform": "sin"
        },
        {
          "name": "RFB",
          "frequency_hz": 1000000.0,
          "amplitude_v": 30.0,
          "phase_deg": 180.0,
          "waveform": "sin"
        }
      ]
    }

coords is "xyz" for planar/3-D Cartesian builds and "rz" for axisymmetric ones (there x is the axis and y the radius). mm_per_gu is the grid pitch — the resolution of the physics (section 5) — and it is declared, never inferred.

## A.2 The shape primitives

Five shape types exist: rect, ellipse, polygon, cutout, and group. All coordinates are millimeters in the deck frame. Any 2-D shape may carry an extrude descriptor giving it a range along one axis in a 3-D build; without one, a 3-D build treats it as a full-depth slab, and a 2-D build ignores extrusion entirely.

**rect — an axis-aligned rectangle from its lower-left corner:**

    {"type": "rect", "x_mm": 21.0, "y_mm": 30.0,
     "width_mm": 14.0, "height_mm": 1.5}

This is the detector plate of the reflectron oa-TOF example: a 14 x 1.5 mm slab with its lower-left corner at (21, 30).

**ellipse — center plus semi-axes (a circle when rx = ry), here with an extrusion making it a 50 mm rod:**

    {"type": "ellipse", "cx_mm": 3.7152, "cy_mm": 2.145,
     "rx_mm": 1.498, "ry_mm": 1.498,
     "extrude": {"axis": "z", "lo_mm": 0.0, "hi_mm": 50.0}}

This is rod_1 of the native 3-D hexapole example: a round rod of radius 1.498 mm extruded the full 50 mm along z. The cross-section plane is the pair of axes perpendicular to the extrude axis in cyclic order (z -\> (x, y)), so cx/cy read as (x, y) here.

**polygon — an arbitrary closed solid from its vertex list. The list is implicitly closed: the last vertex connects back to the first, so the first point is not repeated (repeating it is harmless but unnecessary):**

    {"type": "polygon",
     "points_mm": [[0.0, 0.0], [4.0, 0.0], [4.0, 1.0], [0.0, 2.5]]}

The rasterizer classifies polygons reflection-exactly — a polygon and its mirror image land on identical grid cells — which is what lets mirrored geometry be authored once and unfolded (A.4).

**cutout — subtract a volume; the child shape is what gets removed:**

    {"type": "rect", "x_mm": 1.0, "y_mm": 168.0,
     "width_mm": 38.0, "height_mm": 0.9},
    {"type": "cutout", "children": [
      {"type": "ellipse", "cx_mm": 20.0, "cy_mm": 168.45,
       "rx_mm": 5.0, "ry_mm": 0.45}]}

Shapes compose in declared order: solids union in, cutouts subtract out. A plate followed by an ellipse cutout is a plate with a beam aperture. A cutout that removes nothing on the grid is refused by name — a no-op subtraction is a declaration failure, not a smaller hole. group simply nests shapes so a repeated sub-assembly can be authored once.

## A.3 Signed frames: origin_mm

By default the deck frame starts at (0, 0): node (0,0) of the grid sits at world (0, 0) and every coordinate is non-negative. origin_mm moves that anchor: it declares the world coordinate of grid node (0,0), so a device can be authored centered on the origin with signed coordinates — which is how you naturally measure a multipole. The hexapole example does exactly this:

    "width_mm": 15.6, "height_mm": 15.6,
    "origin_mm": [-7.8, -7.8],
    ...
    {"type": "ellipse", "cx_mm": 3.7152, "cy_mm": 2.145, ...}
    {"type": "ellipse", "cx_mm": -3.7152, "cy_mm": -2.145, ...}

The domain reads \[-7.8, +7.8\] mm in x and y, the six rods sit at +-R about (0, 0), the source is born at (0, 0), and the recorded trajectories come back in the same signed frame. origin_mm must sit on the pitch lattice (origin/mm_per_gu integer), and it is part of the geometry identity: two decks differing only in origin are different instruments to the solver cache.

## A.4 A symmetry example

Symmetry is declared, never assumed (section 4). Declaring a mirror plane makes the solver store and relax only the non-negative half of that axis — half the memory, roughly half the time — while flight, display and recorded output run in the full, unfolded domain with the mirror plane at coordinate 0. The SLIM 3-D transport example declares one mirror:

    "geometry": {
      "width_mm": 37.05, "height_mm": 12.35, "depth_mm": 1.6,
      "mm_per_gu": 0.05,
      "symmetry": {"coords": "xyz", "planes": {"z": "mirror"}},
      "electrodes": [
        {"name": "RF1", "shapes": [
          {"type": "rect", "x_mm": 3.277, "y_mm": 0.0,
           "width_mm": 0.406, "height_mm": 0.2}, ... ]},
      ...
    }

On the mirrored axis, coordinates mean distance from the mirror plane: the stored half is z in \[0, 0.8\] and the flight domain reads z in \[-0.8, +0.8\] with the electrode pattern imaged across z = 0 — the two facing SLIM boards authored once. Shapes on a mirrored axis must not reach below 0 (the image supplies that side), and after every mirrored build the toolkit proves, rather than assumes, that the unfolded potentials are bit-symmetric about the declared plane. Because a mirror plane sits at 0 by construction, origin_mm on a mirrored axis is refused — the two mechanisms answer different questions and compose only on distinct axes.

## A.5 A fully defined shape set: the analytical quadrupole

Everything above assembled into one complete instrument: four round rods, r0 = 3.0 mm, rod radius 1.148 r0 (the dodecapole-minimizing ratio), opposite pairs driven in antiphase at 1 MHz and 30 V. The domain is snapped to the lattice (21.8 mm = 218 cells at 0.1 mm/gu; the remainder of the physical envelope is absorbed at the outer walls, never at a symmetry plane). This deck is reproduced verbatim from a validated file: it passes validate(), builds, and a cold ion launched 0.35 mm off axis is held for the full 160 microsecond flight, oscillating about the center with the expected secular motion.

    {
      "schema_version": 1,
      "name": "Analytical quadrupole (planar cross-section)",
      "notes": "Four round rods, r0 = 3.0 mm, r_rod = 1.148 r0 (dodecapole-minimizing ratio), opposite pairs driven in antiphase. Domain snapped to the lattice: 21.8 mm = 218 cells at 0.1 mm/gu.",
      "geometry": {
        "width_mm": 21.8,
        "height_mm": 21.8,
        "depth_mm": 0.0,
        "mm_per_gu": 0.1,
        "symmetry": {
          "coords": "xyz",
          "planes": {
            "x": "none",
            "y": "none",
            "z": "none"
          }
        },
        "electrodes": [
          {
            "name": "rod_east",
            "dc": 0.0,
            "rf_groups": [
              "RFA"
            ],
            "shapes": [
              {
                "type": "ellipse",
                "cx_mm": 17.344,
                "cy_mm": 10.9,
                "rx_mm": 3.444,
                "ry_mm": 3.444
              }
            ]
          },
          {
            "name": "rod_west",
            "dc": 0.0,
            "rf_groups": [
              "RFA"
            ],
            "shapes": [
              {
                "type": "ellipse",
                "cx_mm": 4.456,
                "cy_mm": 10.9,
                "rx_mm": 3.444,
                "ry_mm": 3.444
              }
            ]
          },
          {
            "name": "rod_north",
            "dc": 0.0,
            "rf_groups": [
              "RFB"
            ],
            "shapes": [
              {
                "type": "ellipse",
                "cx_mm": 10.9,
                "cy_mm": 17.344,
                "rx_mm": 3.444,
                "ry_mm": 3.444
              }
            ]
          },
          {
            "name": "rod_south",
            "dc": 0.0,
            "rf_groups": [
              "RFB"
            ],
            "shapes": [
              {
                "type": "ellipse",
                "cx_mm": 10.9,
                "cy_mm": 4.456,
                "rx_mm": 3.444,
                "ry_mm": 3.444
              }
            ]
          }
        ],
        "rf_groups": [
          {
            "name": "RFA",
            "frequency_hz": 1000000.0,
            "amplitude_v": 30.0,
            "phase_deg": 0.0,
            "waveform": "sin"
          },
          {
            "name": "RFB",
            "frequency_hz": 1000000.0,
            "amplitude_v": 30.0,
            "phase_deg": 180.0,
            "waveform": "sin"
          }
        ]
      },
      "source": {
        "n_ions": 1,
        "distribution": "point",
        "x0_mm": 11.25,
        "y0_mm": 10.9,
        "mz_list": [
          100.0
        ],
        "charge": 1,
        "ke_lo": 0.0,
        "ke_hi": 0.0,
        "temperature_k": 0.0,
        "seed": 0,
        "tob_span_us": 0.0
      },
      "integration": {
        "dt_ns": 2.0,
        "t_max_us": 160.0,
        "rec_every": 25
      },
      "bounds": {},
      "collisions": {
        "enabled": false
      }
    }

Paste it into a .json file and load it from the app or from SimSpec.from_json to fly it yourself.
