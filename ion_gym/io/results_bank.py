"""Bank campaign ensemble results as self-describing .npz archives.

One bank = named per-ion arrays (tofs, slips, tob, ...) plus a JSON
meta block carrying the operating point and every parameter needed to
re-plot the result later without the session that made it. The point
is post-processing: fly once, re-render as many times as the figures
need.

Contract (mirrors field_cycle_io's one-self-describing-npz pattern):
- meta MUST carry 'operating_point' — a banked number without its
  operating point is not a result (correctness invariant).
- An existing path is REFUSED unless overwrite=True: a silently
  replaced bank is lost data.
- Writes are atomic (tmp + os.replace): a crashed write never leaves
  a half-bank behind under the final name.
- No pickled objects: arrays are plain ndarrays, meta is JSON, and
  load uses allow_pickle=False, so a bank is loadable anywhere.
"""

# The public names (docs/PUBLIC_API.md); everything else is internal.
__all__ = ["bank_run", "load_bank", "list_banks", "latest_bank",
           "bank_trajectories", "trajs_from_bank", "bank_flight",
           "load_flight_bank", "BankError"]

import glob
import json
import os
import re
import time

import numpy as np

import ion_gym

SCHEMA = 1
_META_KEY = "__meta_json__"


class BankError(RuntimeError):
    """A results-bank contract violation (named, never swallowed)."""


def bank_run(path, *, meta, arrays, overwrite=False, spec=None):
    """Write one bank. path: file path ('.npz' appended if absent).
    meta: JSON-serializable dict, MUST include 'operating_point'.
    arrays: {name: ndarray}. spec: the SimSpec that flew, so the bank
    records the seed the run used (meta 'seed' / 'seed_mode'). A bank
    without a spec, and without a seed already in meta, records
    seed_mode "not recorded" and says so once: the run cannot be flown
    again exactly. Returns the written path."""
    if "operating_point" not in meta:
        raise BankError(
            "bank_run: meta lacks 'operating_point' — a banked number "
            "without its operating point is not a result")
    if not arrays:
        raise BankError("bank_run: no arrays — an empty bank is a "
                        "bookkeeping bug, not a result")
    if _META_KEY in arrays:
        raise BankError(f"bank_run: array name {_META_KEY!r} is "
                        f"reserved for the meta block")
    p = str(path)
    if not p.endswith(".npz"):
        p += ".npz"
    if os.path.exists(p) and not overwrite:
        raise BankError(
            f"bank_run: {p!r} exists — pass overwrite=True or choose "
            f"a new name; silent replacement is lost data")
    from ion_gym.io.run_provenance import run_seed, versions
    full = dict(meta)
    if spec is not None:
        full["seed"], full["seed_mode"] = run_seed(spec)
    elif "seed" not in full:
        full["seed"], full["seed_mode"] = None, "not recorded"
        print(f"bank_run: {os.path.basename(p)} records no seed (no spec= "
              f"given), so this run cannot be flown again exactly; pass "
              f"spec=<the SimSpec that flew>", flush=True)
    full["schema"] = SCHEMA
    full["ion_gym_version"] = ion_gym.__version__
    # the full stack (checkout commit, numpy, numba, python): the release
    # number alone does not pin a development checkout
    full["versions"] = versions()
    full["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime())
    try:
        meta_json = json.dumps(full)
    except TypeError as e:
        raise BankError(
            f"bank_run: meta is not JSON-serializable ({e}) — pass "
            f"plain numbers/strings/lists, not arrays or objects"
        ) from e
    payload = {_META_KEY: np.array(meta_json)}
    for name, arr in arrays.items():
        payload[name] = np.asarray(arr)
    tmp = p + ".tmp"
    # write-then-rename so a reader never sees a partial bank; a failed
    # write REMOVES its temp file and re-raises (it was left behind)
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **payload)
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return p


def load_bank(path):
    """Read one bank -> (meta_dict, {name: ndarray}). Refuses a file
    without the meta block: that is not a bank."""
    p = str(path)
    with np.load(p, allow_pickle=False) as z:
        if _META_KEY not in z.files:
            raise BankError(f"load_bank: {p!r} has no {_META_KEY!r} "
                            f"block — not a results bank")
        meta = json.loads(str(z[_META_KEY][()]))
        arrays = {k: z[k] for k in z.files if k != _META_KEY}
    return meta, arrays


def list_banks(dir_path):
    """Print name, created stamp, and operating point for every .npz
    bank under dir_path (sorted); returns the list of paths. A file
    that fails to load is reported with its error, never skipped
    silently. A missing directory is an ERROR, not an empty success."""
    d = str(dir_path)
    if not os.path.isdir(d):
        raise BankError(f"list_banks: {d!r} is not a directory")
    paths = sorted(glob.glob(os.path.join(d, "*.npz")))
    for p in paths:
        try:
            meta, _ = load_bank(p)
        except (BankError, OSError, ValueError, KeyError) as e:
            print(f"{os.path.basename(p):44s} UNREADABLE: {e}")
            continue
        print(f"{os.path.basename(p):44s} "
              f"{meta.get('created_utc', '?'):21s} "
              f"{meta.get('operating_point', '?')}")
    if not paths:
        print(f"(no banks in {d})")
    return paths


def _natural_key(path):
    """Sort key that compares runs of digits as numbers, so run_10
    follows run_9; fixed-width timestamps order as before."""
    return [(0, int(t), "") if t.isdigit() else (1, 0, t)
            for t in re.split(r"(\d+)", os.path.basename(path))]


def latest_bank(dir_path, prefix):
    """Last bank under dir_path whose basename starts with prefix, in
    name order with digit runs compared as numbers (run_10 follows
    run_9; timestamped names sort chronologically). File times are not
    used: a copy or checkout resets them. REFUSES if none match,
    listing what is on hand — an empty match is a missing input, not
    a None to limp on with."""
    d = str(dir_path)
    if not os.path.isdir(d):
        raise BankError(f"latest_bank: {d!r} is not a directory")
    hits = sorted((p for p in glob.glob(os.path.join(d, "*.npz"))
                   if os.path.basename(p).startswith(prefix)),
                  key=_natural_key)
    if not hits:
        have = sorted(os.path.basename(p)
                      for p in glob.glob(os.path.join(d, "*.npz")))
        raise BankError(
            f"latest_bank: no bank matching {prefix!r}* in {d!r} — "
            f"on hand: {have if have else '(none)'}")
    return hits[-1]


# ---------------------------------------------------------------------
# Trajectory banks: ragged per-ion records stored as one concatenated
# array per channel plus an 'offsets' index (ion i occupies rows
# offsets[i]:offsets[i+1]). float32 by default: us/mm at these scales
# carry ~1e-3 precision at float32, ample for slip counting, and it
# halves the file. Pass dtype='float64' to keep solver precision.
# ---------------------------------------------------------------------

_TRAJ_MARK = "trajectory_bank_v1"


def bank_trajectories(path, *, results, cols, meta,
                      channels=("t", "x", "wrap_passes", "y", "z"),
                      stride=1, dtype="float32", overwrite=False,
                      spec=None):
    """Bank per-ion trajectory columns for later reprocessing.
    results: iterable with .traj/.index/.summary (driver IonResults).
    cols: the run's column names — every requested channel must be
    present (REFUSED otherwise; a silently absent channel would make
    every later analysis wrong). stride subsamples records (>=1).
    Ions without a usable trajectory are skipped AND reported, with
    their indices stored in meta['skipped']. Also stores per-ion
    'ion_index' and 'tof_us' (from summary) alongside the ragged
    channels. spec: the SimSpec that flew, for the seed (see bank_run).
    Returns the written path."""
    if int(stride) < 1:
        raise BankError(f"bank_trajectories: stride={stride!r} — "
                        f"must be >= 1")
    missing = [c for c in channels if c not in cols]
    if missing:
        raise BankError(
            f"bank_trajectories: channel(s) {missing} not in cols "
            f"{list(cols)!r} — record them at fly time or drop them "
            f"from `channels`")
    idx = {c: cols.index(c) for c in channels}
    chunks = {c: [] for c in channels}
    offsets, ion_index, tofs, skipped = [0], [], [], []
    for r in results:
        if r.traj is None or len(r.traj) < 2:
            skipped.append(int(r.index))
            continue
        tr = r.traj[::int(stride)]
        for c in channels:
            chunks[c].append(np.asarray(tr[:, idx[c]], dtype=dtype))
        offsets.append(offsets[-1] + len(tr))
        ion_index.append(int(r.index))
        tofs.append(float(r.summary["tof"]))
    if skipped:
        print(f"bank_trajectories: skipped {len(skipped)} ion(s) "
              f"with no usable trajectory: {skipped[:20]}"
              + (" ..." if len(skipped) > 20 else ""))
    if not ion_index:
        raise BankError("bank_trajectories: no usable trajectories — "
                        "nothing to bank")
    full_meta = dict(meta, kind=_TRAJ_MARK,
                     channels=list(channels), stride=int(stride),
                     dtype=str(dtype), n_ions=len(ion_index),
                     skipped=skipped)
    arrays = {c: np.concatenate(chunks[c]) for c in channels}
    arrays["offsets"] = np.asarray(offsets, dtype=np.int64)
    arrays["ion_index"] = np.asarray(ion_index, dtype=np.int64)
    arrays["tof_us"] = np.asarray(tofs, dtype=np.float64)
    return bank_run(path, meta=full_meta, arrays=arrays,
                    overwrite=overwrite, spec=spec)


def trajs_from_bank(path_or_loaded):
    """(meta, trajs, tof_us) from a trajectory bank. trajs is a list
    of per-ion dicts with keys 't', 'x' and, when banked, 'wrap'
    (from 'wrap_passes'), 'y', 'z' — the shape the ensemble_figs
    trajectory analyses consume. Views, not copies. Refuses a bank
    that is not a trajectory bank."""
    if isinstance(path_or_loaded, tuple):
        meta, arrays = path_or_loaded
    else:
        meta, arrays = load_bank(path_or_loaded)
    if meta.get("kind") != _TRAJ_MARK:
        raise BankError(
            f"trajs_from_bank: kind={meta.get('kind')!r} — not a "
            f"{_TRAJ_MARK} bank")
    off = arrays["offsets"]
    keymap = {"wrap_passes": "wrap"}
    chans = [c for c in meta["channels"]]
    trajs = []
    for i in range(len(off) - 1):
        a, b = int(off[i]), int(off[i + 1])
        trajs.append({keymap.get(c, c): arrays[c][a:b] for c in chans})
    return meta, trajs, arrays["tof_us"]


# ---------------------------------------------------------------------
# Flight banks: one flown deck, enough to re-render its run figure
# (viz.run_figures.publication_run_figure) without flying again. The
# deck itself (SimSpec.to_dict) rides in meta, so the bank names the
# exact solver input; the impact arrivals are plain arrays; a few full
# trajectories (the ones a figure draws) are stored ragged, as in the
# trajectory banks above.
# ---------------------------------------------------------------------

_FLIGHT_MARK = "flight_bank_v1"
_HIT_FIELDS = ("x", "y", "z", "t_us", "mz")


def bank_flight(path, *, spec, n_flown, cols, hits, trajs, operating_point,
                overwrite=False):
    """Write one flight bank. spec: the SimSpec that flew. n_flown: ions
    born. cols: the record column names. hits: list from
    physics.stations.impact_hits. trajs: the full record arrays to keep
    (the drawn subset). Returns the written path. Refuses a flight with
    no arrivals or no kept trajectories (nothing a run figure could
    draw)."""
    if not hits:
        raise BankError("bank_flight: no impact arrivals in this flight; "
                        "declare a station or enable a bound the ions "
                        "reach, or there is nothing to plot")
    if not trajs:
        raise BankError("bank_flight: no trajectories to keep")
    surfaces = sorted({str(h["station"]) for h in hits})
    sidx = {s: i for i, s in enumerate(surfaces)}
    arrays = {f"hit_{k}": np.array([np.nan if h.get(k) is None
                                    else float(h[k]) for h in hits])
              for k in _HIT_FIELDS}
    arrays["hit_surface"] = np.array([sidx[str(h["station"])]
                                      for h in hits], dtype=np.int64)
    tr = [np.asarray(t, dtype=np.float64) for t in trajs]
    widths = {t.shape[1] for t in tr}
    if widths != {len(cols)}:
        raise BankError(f"bank_flight: trajectory widths {sorted(widths)} "
                        f"do not match the {len(cols)} record columns")
    arrays["traj_rows"] = np.concatenate(tr)
    arrays["traj_offsets"] = np.cumsum([0] + [len(t) for t in tr])
    from ion_gym.io.run_provenance import (deck_location,
                                           portable_spec_dict, run_seed)
    # the seed rides beside the deck: a random-mode deck stores seed null,
    # and without the drawn value the flight cannot be repeated
    seed, seed_mode = run_seed(spec)
    # and the deck's location: a relative STL folder or births file
    # resolves against it, and to_dict() does not carry it (a banked STL
    # flight could not be rebuilt from anywhere else)
    meta = dict(kind=_FLIGHT_MARK, operating_point=operating_point,
                spec=portable_spec_dict(spec),
                spec_loaded_from=deck_location(spec), n_flown=int(n_flown),
                cols=list(cols), surfaces=surfaces,
                seed=seed, seed_mode=seed_mode)
    return bank_run(path, meta=meta, arrays=arrays, overwrite=overwrite)


def load_flight_bank(path):
    """(spec, n_flown, cols, hits, trajs, meta) from a flight bank, with
    hits in the impact_hits dict shape and spec rebuilt from the banked
    deck. Refuses a bank of another kind."""
    from ion_gym.io.sim_spec import SimSpec
    meta, arrays = load_bank(path)
    if meta.get("kind") != _FLIGHT_MARK:
        raise BankError(f"load_flight_bank: kind={meta.get('kind')!r} is "
                        f"not a {_FLIGHT_MARK} bank")
    surfaces = meta["surfaces"]
    n = len(arrays["hit_surface"])
    hits = []
    for i in range(n):
        h = {"station": surfaces[int(arrays["hit_surface"][i])]}
        for k in _HIT_FIELDS:
            v = float(arrays[f"hit_{k}"][i])
            h[k] = None if np.isnan(v) else v
        hits.append(h)
    off = arrays["traj_offsets"]
    rows = arrays["traj_rows"]
    trajs = [rows[int(off[i]):int(off[i + 1])] for i in range(len(off) - 1)]
    from ion_gym.io.run_provenance import with_location
    spec = with_location(SimSpec.from_dict(meta["spec"]),
                         meta.get("spec_loaded_from"))
    return spec, int(meta["n_flown"]), list(meta["cols"]), hits, trajs, meta
