"""
stl_upload.py — user-facing STL import for ion_gym.

Charter decisions implemented (CHARTER, "decided this cycle v52"):

(a) The user sets the voxel pitch. ion_gym makes NO assumption about the
    shape or type of optic imported — no feature detection, no pitch
    proposal. What it DOES provide is the COST of the chosen pitch: domain
    (union AABB), grid dims, memory (n_electrode float32 bases), and an
    estimated solve/calc time — so the user picks pitch against real cost,
    plus a units sanity check (extent only).
(b) A coarse preview voxelization catches unit (mm-vs-inch/m) and
    registration errors for pennies before the real solve.
(c) The voltage table rows ARE the fast-adjust voltage vector: row order =
    basis index = the quad_N.stl convention shared with the external export
    path.
(d) Persistence bundle = sim.json + bases/*.npz + geometry/*.stl +
    manifest.json carrying a sha256 content hash over (geometry bytes,
    pitch, domain, symmetry, solver version). Hash mismatch (including a
    solver-version bump) => re-solve; never silently reuse stale bases.

The core (sizing / hash / bundle) is Panel-free and unit-tested in
test_stl_upload.py; StlUploadPanel is the Panel front end, wired into
sim_app as the "STL upload" tab with an on_commit(spec) callback.
"""
from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:                    # ruff F821: the "SimSpec" return
    from ion_gym.io.sim_spec import SimSpec   # annotation's name source
    from ion_gym.physics.sizing import SizingProposal  # annotations only
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

SOLVER_VERSION = "solver3d/nz1-v51"      # bump on any solver change


# mesh ingest has ONE owner (L-546): STL bytes are parsed through
# stl_resolve.parse_stl_bytes, a deck's files with their declared placement
# through stl_resolve.load_mesh, and trimesh's import through
# stl_resolve._require_trimesh. This module used to call trimesh.load
# itself in three places and carry its own _require_trimesh.
from ion_gym.io.stl_resolve import parse_stl_bytes        # noqa: E402


# ------------------------------------------------------------- mesh info
@dataclass
class MeshInfo:
    name: str                 # electrode label (defaults to file stem)
    filename: str
    data: bytes               # raw STL bytes (hash + bundle source)
    n_faces: int
    aabb_min: np.ndarray      # (3,) mm (as-authored units, assumed mm)
    aabb_max: np.ndarray

    @property
    def extent(self):
        return self.aabb_max - self.aabb_min


def load_mesh_bytes(filename: str, data: bytes) -> MeshInfo:
    m = parse_stl_bytes(data)
    lo, hi = np.asarray(m.bounds[0], float), np.asarray(m.bounds[1], float)
    return MeshInfo(name=Path(filename).stem, filename=filename, data=data,
                    n_faces=len(m.faces), aabb_min=lo, aabb_max=hi)


# --------------------------------------------------------- sizing proposal
# ONE cost model: stl_upload used to carry its OWN
# SizingProposal + propose_sizing with a "rough SOR estimate" that (a)
# used the wrong solver's constants and (b) carried a spurious extra
# `/ 1e3` factor — so an STL import estimated 1000x too FAST (a
# 274 M-voxel import read 2.8 min here while the real solve status bar
# read ~48 h from sizing.py's estimator). Two cost models is one too
# many; the duplicate is deleted and this delegates to
# sizing.propose_sizing_from_aabbs, the SAME authority sizing_for /
# build_run's status bar use. The STL readout and the solve estimate now
# agree by construction, on the corrected multigrid throughput.


def propose_sizing(meshes: List[MeshInfo], *, pitch: float = 0.5,
                   margin_mm: float = 2.0) -> SizingProposal:
    """Domain from the union AABB + margin, and the SOLVE-TIME / memory
    COST at the user's chosen pitch, via the one shared estimator
    (sizing.propose_sizing_from_aabbs). Makes NO assumption about the
    optic — no feature detection, no pitch proposal. The user sets the
    pitch; this only reports what it will cost."""
    if not meshes:
        raise ValueError("no meshes loaded")
    from ion_gym.physics.sizing import propose_sizing_from_aabbs \
        # deferred to call time (io stands below physics)
    return propose_sizing_from_aabbs(
        [m.aabb_min for m in meshes], [m.aabb_max for m in meshes],
        len(meshes), pitch=float(pitch), margin_mm=margin_mm)


# --------------------------------------------------------------- manifest
def manifest_hash(meshes: List[MeshInfo], pitch: float,
                  domain_min, domain_max, symmetry: Optional[dict] = None,
                  solver_version: str = SOLVER_VERSION) -> str:
    """sha256 over everything that invalidates a solved basis set.
    Meshes are sorted by name so upload order never changes the key."""
    h = hashlib.sha256()
    for m in sorted(meshes, key=lambda m: m.name):
        h.update(m.name.encode())
        h.update(m.data)
    h.update(f"{pitch:.9g}".encode())
    h.update(np.asarray(domain_min, float).tobytes())
    h.update(np.asarray(domain_max, float).tobytes())
    h.update(json.dumps(symmetry or {}, sort_keys=True).encode())
    h.update(solver_version.encode())
    return h.hexdigest()


def save_bundle(dirpath, spec_json: str, meshes: List[MeshInfo],
                bases: Dict[str, np.ndarray], sizing: SizingProposal,
                symmetry: Optional[dict] = None) -> Path:
    """Write the portable unit: sim.json + geometry/ + bases/ + manifest.
    Move the folder (or zip it) and ion_gym elsewhere reproduces the run
    without re-solving — the manifest hash certifies the bases match."""
    d = Path(dirpath)
    (d / "geometry").mkdir(parents=True, exist_ok=True)
    (d / "bases").mkdir(exist_ok=True)
    (d / "sim.json").write_text(spec_json)
    for m in meshes:
        (d / "geometry" / m.filename).write_bytes(m.data)
    for name, arr in bases.items():
        np.savez_compressed(d / "bases" / f"{name}.npz",
                            basis=arr.astype(np.float32))
    man = dict(
        hash=manifest_hash(meshes, sizing.pitch, sizing.domain_min,
                           sizing.domain_max, symmetry),
        solver_version=SOLVER_VERSION, pitch=sizing.pitch,
        domain_min=list(map(float, sizing.domain_min)),
        domain_max=list(map(float, sizing.domain_max)),
        dims=list(sizing.dims), symmetry=symmetry or {},
        electrodes=[m.name for m in sorted(meshes, key=lambda m: m.name)],
        created=time.strftime("%Y-%m-%d %H:%M:%S"))
    (d / "manifest.json").write_text(json.dumps(man, indent=2))
    return d


def load_bundle(dirpath):
    """Load a bundle; returns (manifest, meshes, bases, fresh: bool).
    fresh=False (hash mismatch — geometry/pitch/domain/symmetry/solver
    version changed) means the bases are STALE: re-solve, never reuse."""
    d = Path(dirpath)
    man = json.loads((d / "manifest.json").read_text())
    meshes = [load_mesh_bytes(p.name, p.read_bytes())
              for p in sorted((d / "geometry").glob("*.stl"))]
    bases = {p.stem: np.load(p)["basis"]
             for p in sorted((d / "bases").glob("*.npz"))}
    now = manifest_hash(meshes, man["pitch"], man["domain_min"],
                        man["domain_max"], man.get("symmetry") or {},
                        man.get("solver_version", ""))
    fresh = (now == man["hash"]
             and man.get("solver_version") == SOLVER_VERSION)
    return man, meshes, bases, fresh


# ------------------------------------------------------------- validation
def check_mask_overlaps(spec):
    """Detect interpenetrating electrodes. The voxelizer itself REFUSES
    overlapping solids (correct-physics-over-flying-wrong), raising
    ValueError('electrode i overlaps j at node ...'); this wraps that
    refusal into a report the UI can show. Returns [] when clean, else
    [(message,)] describing the collision. The classic cause: CAD parts
    exported to STL individually land in PART frames, not the assembly
    frame. Any OTHER failure of the check propagates: a check that did
    not run is not a clean result (L-529). The overlap is recognised by
    its TYPE, voxelize.ElectrodeOverlap -- this matched the word
    'overlaps' in the message."""
    from ion_gym.physics.build_stl import stl_masks_2d
    from ion_gym.physics.voxelize import ElectrodeOverlap
    try:
        stl_masks_2d(spec)
        return []
    except ElectrodeOverlap as e:
        return [(str(e),)]


def spec_from_upload(meshes: List[MeshInfo], voltages: Dict[str, float],
                     sizing: SizingProposal, workdir,
                     rf_groups: Optional[Dict[str, str]] = None,
                     solve_3d: bool = False,
                     name="stl_upload") -> "SimSpec":
    """Write build-frame STLs to workdir and return a SimSpec on the
    validated build_stl path (ElectrodeSpec.stl + geometry.stl_dir;
    build_run dispatches on stl presence at depth_mm=0).

    Frame convention (pinned from stl_masks_2d): the solver grid spans
    (0..width_mm, 0..height_mm); the CAD->build translation of -domain_min
    is DECLARED in geometry.frame_offset_mm and applied at the single
    mesh-ingest point, stl_resolve.load_mesh. The working-dir
    files are byte-for-byte the uploaded meshes -- no translated build
    artifacts -- so the manifest hash over as-uploaded bytes describes
    exactly the files on disk (the copies were once translated
    while the hash covered the originals: two byte-states, one hash).
    Row/basis order = sorted electrode names = the quad_N.stl convention.
    """
    from ion_gym.io.sim_spec import (SimSpec, ElectrodeSpec, GeometrySpec, SourceSpec,
                          RFGroupSpec)
    from ion_gym.io.stl_resolve import load_mesh
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    off = -np.asarray(sizing.domain_min, float)
    rf_groups = rf_groups or {}
    electrodes = []
    for m in sorted(meshes, key=lambda m: m.name):
        # AS-UPLOADED BYTES ON DISK: no translation is baked into
        # the copies; placement rides in the declaration below.
        (wd / m.filename).write_bytes(m.data)
        grp = rf_groups.get(m.name) or None
        electrodes.append(ElectrodeSpec(
            name=m.name, stl=m.filename,
            dc=float(voltages.get(m.name, 0.0)), rf_groups=([grp] if grp else [])))
    # Every distinct group name becomes a real RFGroupSpec so the Voltages
    # tab (which builds its options from geometry.rf_groups) can render and
    # edit it, and _sync_spec round-trips instead of erasing. Amplitude/
    # frequency/phase start at 0 — the USER sets the drive (no assumptions
    # about the optic).
    group_specs = [RFGroupSpec(name=g, amplitude_v=0.0, frequency_hz=0.0,
                               phase_deg=0.0)
                   for g in sorted({g for g in rf_groups.values() if g})]
    ext = np.asarray(sizing.domain_max, float) - np.asarray(
        sizing.domain_min, float)
    geo = GeometrySpec(
        width_mm=float(ext[0]), height_mm=float(ext[1]),
        depth_mm=float(ext[2]) if solve_3d else 0.0,
        # axial_extent_mm is set below from the meshes as INGESTED
        mm_per_gu=float(sizing.pitch), stl_dir=str(wd),
        # DECLARED (2a.3): the measured CAD->build translation as
        # first-class data (was notes prose only). build = uploaded +
        # frame_offset_mm; plane_mm declarations on this spec are in the
        # BUILD frame.
        frame_offset_mm=[float(v) for v in off],
        electrodes=electrodes, rf_groups=group_specs)
    spec = SimSpec(name=name, geometry=geo, source=SourceSpec())
    if not solve_3d:
        # measured ONCE at spec creation and DECLARED into the JSON (the
        # draw path reads only the declaration): the bodies' BUILD-frame
        # z-span, so 2-D transport views draw finite electrodes; the user
        # can edit it like any other config value. Read through THE mesh
        # ingest point, which applies the declared placement (L-546: the
        # offset used to be re-added to native bounds by hand here).
        zs = [load_mesh(spec, el, stl_dir=wd).bounds[:, 2]
              for el in electrodes]
        geo.axial_extent_mm = [float(min(z[0] for z in zs)),
                               float(max(z[1] for z in zs))]
    spec.notes = (f"STL upload: build = uploaded + {off.tolist()} mm "
                  f"(domain_min -> origin), declared in geometry."
                  f"frame_offset_mm and applied at mesh ingest; files on "
                  f"disk ARE the uploaded bytes (manifest hash matches).")
    return spec


# --------------------------------------------------------------- Panel UI
class StlUploadPanel:
    """The "STL upload" tab. Flow: drop STLs -> table + live sizing ->
    (optional) coarse preview -> Commit builds a SimSpec and hands it to
    on_commit(spec); the app then shows geometry immediately and solves
    only on Recompute/Fly (matching _rebuild_for_new_spec discipline)."""

    def __init__(self, on_commit: Callable, on_clear: Callable = None,
                 workdir=None):
        import panel as pn
        self._pn = pn
        self.on_commit = on_commit
        self.on_clear = on_clear
        if workdir is None:
            from ion_gym.io.paths import uploads_dir
            # no /tmp absolutes: uploads persist in the registered
            # uploads home; an explicit workdir is honoured unchanged
            # (every gate passes one).
            workdir = uploads_dir("stl")
        self.workdir = Path(workdir)
        self.meshes: List[MeshInfo] = []
        self.sizing: Optional[SizingProposal] = None
        # dropped files that did not parse as STL, kept until the next
        # drop: every resize shows them and Commit refuses while any
        # remain, so a deck cannot commit with an electrode missing (L-529)
        self.skipped: List[str] = []

        # NB: do NOT set accepted_filetypes here. FilePond's MIME detection
        # rejects .stl (no consistent MIME type) and the drop silently fails
        # — the value never changes so the watcher never fires. Accept
        # everything and validate by trying to parse as STL in _on_files.
        self.dropper = pn.widgets.FileDropper(
            multiple=True, max_file_size="200MB", height=110,
            layout="integrated")
        self.table = pn.widgets.Tabulator(
            value=self._df([]), show_index=False, height=180,
            editors={"dc [V]": {"type": "number"},
                     "rf_group": {"type": "input"},
                     "electrode": None, "mesh faces": None})
        self.pitch = pn.widgets.FloatInput(name="pitch mm/gu (you set)",
                                           value=0.5, step=0.05, width=150)
        self.sizing_md = pn.pane.Markdown("_drop one STL per electrode; set "
                                          "a pitch to see the solve cost_",
                                          sizing_mode="stretch_width")
        self.preview_btn = pn.widgets.Button(
            name="⬛ Preview voxelization (coarse)", button_type="primary",
            width=230)
        self.commit_btn = pn.widgets.Button(name="Commit geometry",
                                            button_type="success",
                                            width=170, disabled=True)
        self.w_3d = pn.widgets.Checkbox(
            name="solve full 3-D (slower; otherwise 2-D mid-z slice)",
            value=False)
        self.clear_btn = pn.widgets.Button(
            name="Clear loaded STLs + cache", button_type="warning",
            width=210)
        self.preview_pane = pn.pane.Plotly(height=720,
                                           sizing_mode="stretch_width")

        self.dropper.param.watch(self._on_files, "value")
        self.pitch.param.watch(lambda e: self._resize(), "value")
        self.preview_btn.on_click(self._on_preview)
        self.commit_btn.on_click(self._on_commit)
        self.clear_btn.on_click(self._on_clear)

    def _on_clear(self, _=None):
        """Clear staged STLs from memory, wipe the upload workdir, and
        purge the solved-basis cache. Then notify the app (on_clear) so it
        can drop the committed geometry."""
        from ion_gym.io.fa_cache import (clear_all, CachePurgeIncomplete,
                                         remove_tree)
        self.meshes = []
        self.skipped = []
        self.sizing = None
        self.table.value = self._df([])
        self.preview_pane.object = None
        self.commit_btn.disabled = True
        # every failure is collected and SHOWN (L-543): this was
        # rmtree(ignore_errors=True) plus "except Exception: 0 purged",
        # so a purge that failed said it succeeded
        failed = [f"{p} ({why})"
                  for p, why in remove_tree(self.workdir).items()]
        try:
            n, freed = clear_all()
            cache_line = f"{n} cache entries purged ({freed/1e6:.1f} MB freed)"
        except CachePurgeIncomplete as e:
            cache_line = (f"cache purge INCOMPLETE: {e.n_removed} entries "
                          f"removed ({e.bytes_freed/1e6:.1f} MB), "
                          f"{len(e.failures)} NOT removed")
            failed += [f"{p} ({why})" for p, why in e.failures.items()]
        if failed:
            self.sizing_md.object = (
                f"⚠️ **clear incomplete** — staged STLs dropped; "
                f"{cache_line}. Could not delete: " + "; ".join(failed))
        else:
            self.sizing_md.object = (f"**cleared** — staged STLs dropped, "
                                     f"workdir wiped, {cache_line}.")
        if self.on_clear:
            self.on_clear()

    # ---- helpers
    def _df(self, meshes):
        import pandas as pd
        return pd.DataFrame(
            [{"electrode": m.name, "dc [V]": 0.0, "rf_group": "",
              "mesh faces": m.n_faces} for m in meshes],
            columns=["electrode", "dc [V]", "rf_group", "mesh faces"])

    def _on_files(self, event):
        import base64
        import binascii
        files = event.new or {}
        loaded, skipped = [], []
        for fn, data in files.items():
            if isinstance(data, str):
                # FileDropper may hand back text (ASCII STL) or base64
                try:
                    data = base64.b64decode(data, validate=True)
                except (binascii.Error, ValueError):
                    data = data.encode("utf-8", "replace")
            try:
                loaded.append(load_mesh_bytes(fn, data))
            except Exception as e:   # noqa: BLE001 -- trimesh raises many
                # types on malformed bytes; each is REPORTED by file name
                # and blocks Commit (self.skipped), never dropped
                skipped.append(f"{fn} ({type(e).__name__}: {e})")
        self.meshes = loaded
        self.skipped = skipped
        self.table.value = self._df(self.meshes)
        self._resize()

    def _skipped_md(self):
        return ("⚠️ **could not parse as STL** (Commit is disabled until "
                "these are fixed or removed from the drop): "
                + "; ".join(self.skipped)) if self.skipped else ""

    def _resize(self):
        if not self.meshes:
            self.sizing = None
            self.commit_btn.disabled = True
            if self.skipped:
                self.sizing_md.object = self._skipped_md()
            return
        self.sizing = propose_sizing(self.meshes, pitch=self.pitch.value)
        # the parse-failure list SURVIVES every resize (L-529: the sizing
        # readout overwrote it as soon as one mesh had loaded)
        self.sizing_md.object = "\n\n".join(
            s for s in (self._skipped_md(), self.sizing.markdown()) if s)
        self.commit_btn.disabled = bool(self.skipped)

    def _voltages(self):
        df = self.table.value
        dc = {r["electrode"]: float(r["dc [V]"]) for _, r in df.iterrows()}
        rf = {r["electrode"]: str(r["rf_group"]).strip()
              for _, r in df.iterrows() if str(r["rf_group"]).strip()}
        return dc, rf

    def _on_preview(self, _=None):
        """3-D preview of the uploaded STLs IN PLACE: one colour per
        electrode, name on hover. This is the registration/units check —
        you should see your assembly, correctly arranged in 3-D."""
        if not self.meshes:
            return
        # the figure is drawn by THE viz_core mesh preview (L-548)
        from ion_gym.viz.viz_core import mesh_preview_figure
        self.preview_pane.object = mesh_preview_figure(
            [(m.name, parse_stl_bytes(m.data), i)
             for i, m in enumerate(self.meshes)],
            height=700,
            title="3-D preview — colour = electrode, hover for name; check "
                  "units & placement")

    def _on_commit(self, _=None):
        """Commit only a deck that is whole and checked (L-529): every
        dropped file parsed, and the overlap check RAN and found nothing.
        It used to commit after 'except Exception: overlaps = []' -- a
        check that failed read as a clean one -- and committed even when
        it had just reported an interpenetration."""
        if not (self.meshes and self.sizing):
            return
        if self.skipped:
            self.sizing_md.object = (self._skipped_md() + "\n\n"
                                     + self.sizing.markdown())
            return
        dc, rf = self._voltages()
        spec = spec_from_upload(self.meshes, dc, self.sizing,
                                self.workdir, rf_groups=rf,
                                solve_3d=self.w_3d.value)
        try:
            overlaps = check_mask_overlaps(spec)
        except (ValueError, OSError, ImportError) as e:
            self.sizing_md.object = (
                self.sizing.markdown()
                + f"\n\n⚠️ **not committed** — the electrode-overlap check "
                  f"could not run ({type(e).__name__}: {e}), so the "
                  f"geometry is unchecked.")
            return
        if overlaps:
            msg = "; ".join(o[0] for o in overlaps)
            self.sizing_md.object = (
                self.sizing.markdown()
                + f"\n\n⚠️ **not committed: electrodes interpenetrate** — "
                  f"the voxelizer refuses to build this ({msg}). This "
                  "usually means the STLs were exported in PART frames "
                  "(assembly placement lost). Fix placement in CAD and "
                  "re-export.")
            return
        self.on_commit(spec)

    def panel(self):
        pn = self._pn
        # LAYOUT: the destructive "Clear" used to share
        # a row with the action buttons directly above the 720 px preview
        # plot, where an unwidth'd button could render over the plot. Each
        # button now carries an explicit width, the primary actions
        # (Preview / Commit) sit in their own row, and Clear is pushed to
        # its OWN row under a divider, right-aligned away from the plot —
        # it can neither crowd the actions nor overlap the canvas.
        return pn.Column(
            self.dropper,
            self.table,
            pn.Row(self.pitch, self.preview_btn, self.commit_btn),
            self.w_3d,
            self.sizing_md,
            pn.layout.Divider(),
            pn.Row(pn.layout.HSpacer(), self.clear_btn),
            pn.layout.Divider(),
            self.preview_pane,
            sizing_mode="stretch_width")
