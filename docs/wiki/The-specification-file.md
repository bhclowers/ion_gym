<!-- Generated from the ion_gym user manual and the shipped decks and notebooks. Edits made here are overwritten. -->

# 3. The specification file

A specification is JSON with a small number of top-level sections. The sections that matter most are:

| **Section** | **What it declares**                                                                                                         |
|-------------|------------------------------------------------------------------------------------------------------------------------------|
| geometry    | Grid pitch, extents, symmetry, electrodes, their voltages and RF group membership, and how the field is differenced.         |
| source      | How many ions, their *m/z*, their spatial and velocity distribution, and the random seed.                                    |
| integration | Time step, maximum flight time, and how often trajectory points are recorded.                                                |
| collisions  | Background gas species, pressure and temperature, or none for vacuum.                                                        |
| bounds      | Optional planes that terminate a flight when crossed.                                                                        |
| stations    | Declared measurement planes in the flight: detectors that log or absorb, and plates with transmission windows (section 7.5). |

Two properties of this file are worth internalizing. First, the resolution is declared, not inferred: mm_per_gu is the grid pitch in millimeters, and it sets both the accuracy and the cost of every solve. Second, defaults are omitted. A field left out of the file takes its documented default and is not written back, so two specifications that differ only in a default are byte-identical files. (Purely cosmetic fields — an electrode's display color, the recording caps — are reported once at load when left out; the decks that ship in examples/ state every field they rely on and load silently.)

## A minimal specification: the einzel lens

The deck below is `examples/einzel_round_r-z.json` with every field that equals its default removed. It loads to the same specification as the shipped file, so it solves the same field and flies the same ions. The shipped file states the remaining fields as well, so it loads without notices.

```json
{
  "schema_version": 1,
  "name": "einzel lens (r-z, three-tube) — Abdulla & Hussein 2020 geometry, decelerating mode",
  "geometry": {
    "width_mm": 190.0,
    "height_mm": 22.0,
    "mm_per_gu": 0.1,
    "symmetry": {
      "coords": "rz"
    },
    "electrodes": [
      {
        "name": "entrance",
        "shapes": [
          {
            "type": "rect",
            "x_mm": 12.0,
            "y_mm": 15.0,
            "width_mm": 25.0,
            "height_mm": 1.5
          }
        ]
      },
      {
        "name": "centre",
        "shapes": [
          {
            "type": "rect",
            "x_mm": 31.0,
            "y_mm": 12.0,
            "width_mm": 25.0,
            "height_mm": 1.5
          }
        ],
        "dc": 57.0
      },
      {
        "name": "exit",
        "shapes": [
          {
            "type": "rect",
            "x_mm": 50.0,
            "y_mm": 15.0,
            "width_mm": 25.0,
            "height_mm": 1.5
          }
        ]
      }
    ]
  },
  "source": {
    "n_ions": 12,
    "distribution": "disc",
    "x0_mm": 1.0,
    "y0_mm": 0.0,
    "z0_mm": 0.0,
    "r_mm": 3.0,
    "ke_lo": 100.0,
    "ke_hi": 100.0,
    "mz_list": [
      120.0
    ],
    "tob_span_us": 2.0
  },
  "collisions": {
    "enabled": false,
    "P_torr": 1.0
  },
  "integration": {
    "dt_ns": 2.0,
    "t_max_us": 60.0,
    "rec_every": 4
  },
  "bounds": {
    "x_max_on": true,
    "x_max": 190.0
  },
  "stations": [
    {
      "name": "detector",
      "on_hit": "pass",
      "pos_mm": 189.0,
      "window": {
        "y": [
          -15.0,
          15.0
        ],
        "z": [
          -15.0,
          15.0
        ]
      }
    }
  ]
}
```

Three kinds of field appear in a specification.

- **Required.** These set the physics, and no default is neutral for them. A deck that leaves one out is refused at load with a message naming the key. Which source keys are required depends on the birth distribution: a disc needs its radius, a box its extents, and a births file only the ion count and masses.
- **Announced.** Display settings and the recording caps. When left out they load at their defaults and the load prints one notice listing them.
- **Silent defaults.** Everything else. When left out it takes the default shown in the tables below, and a saved deck does not write it back.

## Every field and its default

The required column is evaluated for this deck. Source keys that depend on the birth distribution are marked for a disc. The entries of the electrodes, shapes, stations and drive group lists have their own tables after the sections.

### `geometry`

| field | default | kind |
|---|---|---|
| `width_mm` | (must be stated) | required |
| `height_mm` | (must be stated) | required |
| `depth_mm` | `0.0` | default |
| `metal_depth_mm` | `0.0` | default |
| `axial_extent_mm` | `null` | default |
| `mm_per_gu` | (must be stated) | required |
| `symmetry` | `{"coords": "xyz", "planes": {"x": "none", "y": "none", "z": "none"}}` | default |
| `stl_dir` | `null` | default |
| `stl_manifest` | `null` | default |
| `frame_offset_mm` | `null` | default |
| `origin_mm` | `null` | default |
| `field_method` | `"electrode_aware"` | default |
| `channel_dtype` | `"float64"` | default |
| `electrodes` | `[]` | default |
| `rf_groups` | `[]` | default |
| `dc_groups` | `[]` | default |

### `source`

| field | default | kind |
|---|---|---|
| `n_ions` | (must be stated) | required |
| `distribution` | (must be stated) | required |
| `x0_mm` | (must be stated) | required |
| `y0_mm` | (must be stated) | required |
| `z0_mm` | (must be stated) | required |
| `r_mm` | (must be stated) | required |
| `len_mm` | `0.0` | default |
| `box_mm` | `[1.0, 1.0, 1.0]` | default |
| `axis` | `"x"` | default |
| `direction` | `[1.0, 0.0, 0.0]` | default |
| `ke_lo` | (must be stated) | required |
| `ke_hi` | (must be stated) | required |
| `temperature_k` | `0.0` | default |
| `dv_fwhm_ms` | `[0.0, 0.0, 0.0]` | default |
| `fwhm_mm` | `null` | default |
| `trunc_mm` | `null` | default |
| `v_drift_mm_us` | `[0.0, 0.0, 0.0]` | default |
| `mz_list` | (must be stated) | required |
| `charge` | `1` | default |
| `tob_span_us` | (must be stated) | required |
| `births_file` | `null` | default |
| `seed` | `null` | default |

### `collisions`

| field | default | kind |
|---|---|---|
| `enabled` | `true` | default |
| `gas` | `"N2"` | default |
| `T_k` | `273.0` | default |
| `P_torr` | `null` | default |
| `P_pa` | `null` | default |
| `sigma_m2` | `2.27e-18` | default |
| `model` | `"hs"` | default |
| `gas_diam_nm` | `0.366` | default |
| `flow_mm_us` | `[0.0, 0.0, 0.0]` | default |
| `sigma_model` | `"ccs_fit"` | default |
| `ccs_fit_class` | `"all"` | default |
| `sds_species` | `[]` | default |

### `integration`

| field | default | kind |
|---|---|---|
| `dt_ns` | (must be stated) | required |
| `t_max_us` | (must be stated) | required |
| `rec_every` | `80` | announced |
| `max_records` | `100000` | announced |
| `record_channels` | `["speed", "ke_ev", "e_field"]` | default |

### `bounds`

| field | default | kind |
|---|---|---|
| `x_min_on` | `false` | default |
| `x_max_on` | `false` | default |
| `y_min_on` | `false` | default |
| `y_max_on` | `false` | default |
| `z_min_on` | `false` | default |
| `z_max_on` | `false` | default |
| `x_min` | `0.0` | default |
| `x_max` | `100.0` | default |
| `y_min` | `-50.0` | default |
| `y_max` | `50.0` | default |
| `z_min` | `-50.0` | default |
| `z_max` | `50.0` | default |

### `stations` (each entry)

| field | default | kind |
|---|---|---|
| `name` | `"station"` | default |
| `kind` | `"detect"` | default |
| `on_hit` | `null` | default |
| `axis` | `"x"` | default |
| `pos_mm` | `0.0` | default |
| `window` | `{}` | default |

### `transporter`

| field | default | kind |
|---|---|---|
| `axis` | (must be stated) | required |
| `accept_mm` | (must be stated) | required |
| `emit_mm` | (must be stated) | required |
| `direction` | (must be stated) | required |
| `max_passes` | `1000000000` | default |

### `optic_axis`

| field | default | kind |
|---|---|---|
| `axis` | (must be stated) | required |
| `through_mm` | (must be stated) | required |

### `view`

| field | default | kind |
|---|---|---|
| `mode` | `"2d"` | announced |
| `planes` | `["xy"]` | announced |
| `equipotential_lines` | `16` | announced |
| `show_field` | `false` | announced |
| `color_by` | `"fate"` | announced |
| `decimate` | `4` | announced |

### `geometry.symmetry`

| field | default | kind |
|---|---|---|
| `coords` | `"xyz"` | default |
| `planes` | `{"x": "none", "y": "none", "z": "none"}` | default |
| `plane_mm` | `{}` | default |

### `geometry.electrodes[]`

| field | default | kind |
|---|---|---|
| `name` | (must be stated) | required |
| `shapes` | `[]` | default |
| `stl` | `null` | default |
| `is_grid` | `false` | default |
| `dc` | `0.0` | default |
| `rf_groups` | `[]` | default |
| `dc_group` | `null` | default |
| `dc_index` | `null` | default |
| `dc_weight` | `null` | default |
| `color` | `[200, 60, 60]` | announced |
| `basis` | `null` | default |
| `metal_depth_mm` | `null` | default |

### `geometry.rf_groups[]`

| field | default | kind |
|---|---|---|
| `name` | (must be stated) | required |
| `frequency_hz` | (must be stated) | required |
| `amplitude_v` | (must be stated) | required |
| `phase_deg` | `0.0` | default |
| `duty` | `0.5` | default |
| `offset_v` | `0.0` | default |
| `waveform` | `"sin"` | default |
| `table_t_us` | `[]` | default |
| `table_v` | `[]` | default |
| `interp` | `"hold"` | default |
| `pe_mode` | `null` | default |

### `geometry.dc_groups[]`

| field | default | kind |
|---|---|---|
| `name` | `"LADDER"` | default |
| `v_in` | `0.0` | default |
| `v_out` | `0.0` | default |
| `interp` | `"linear"` | default |
| `mode` | `"set"` | default |
| `uniform` | `false` | default |

### `geometry.electrodes[].shapes[]`

Each shape states its `type` and the keys that type reads (a `rect` needs `x_mm`, `y_mm`, `width_mm` and `height_mm`, for example). Appendix A lists every shape type and its keys.
