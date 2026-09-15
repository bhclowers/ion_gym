"""Reusable CadQuery script emission — the general machinery, factored out
of a project-specific geometry module (the CadQuery
architecture belongs in its own set of functions).

A geometry module (Q3, SLIM, reflectron, ...) supplies its OWN parametric
solid-building code as text; this module owns everything that is NOT
geometry-specific and was previously duplicated per project:

  * clean_num / fmt_params: emit derived numbers without binary float
    noise (5.449999999999999 has no place in a CAD script) and a named
    parameter block, so "every number is derived, none is typed twice".
  * ELECTRODE-NUMBERING DOCTRINE, in one place: the index in the filename
    IS the basis index IS the int16 voxel label. If two conductors land
    in one STL they solve as ONE electrode and every downstream voltage
    is wrong. The emitted export_stls() enforces one-file-per-electrode.
  * the assembly + cadDict registry boilerplate, the STL exporter, and
    the STEP save line — identical across projects.

The emitter returns a runnable .py string (CadQuery is an OPTIONAL,
author-time dependency — ion_gym never imports cadquery at runtime, so
this module only builds TEXT and must not import cadquery itself).

Design: CadEmitter is a small builder. A project calls add_electrode(
number, name, build_expr, color) with a CadQuery expression string that
evaluates to the solid, plus header/param blocks, and render() returns
the script. No project-specific constant appears here.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


def clean_num(v, ndigits: int = 6) -> str:
    """Format a float as a clean CAD literal: round off binary noise
    (1e-6 mm is far below any electrode tolerance) and drop trailing
    zeros. round(5.449999999999999,6) -> '5.45'."""
    return f"{round(float(v), ndigits):g}"


def fmt_params(params, ndigits: int = 6) -> str:
    """A named parameter block from (NAME, value, comment) triples. Values
    are clean_num'd (numbers) or repr'd (str/other). One assignment per
    line, comments aligned after the value."""
    lines = []
    width = max((len(p[0]) for p in params), default=0)
    for name, value, *rest in params:
        comment = rest[0] if rest else ""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            lit = clean_num(value, ndigits)
        else:
            lit = repr(value)
        line = f"{name:<{width}} = {lit}"
        if comment:
            line += f"    # {comment}"
        lines.append(line)
    return "\n".join(lines)


@dataclass
class _Electrode:
    number: int
    name: str
    build_expr: str          # CadQuery expr text evaluating to the solid
    color: str = "gray"
    label: Optional[str] = None   # display label (defaults to name)


# The doctrine banner emitted into every script's export function — the
# single source of the numbering rule that q3 stated inline.
_EXPORT_DOC = '''# ONE FILE PER ELECTRODE. The index in the filename IS the basis index
# and IS the int16 label the voxeliser assigns. If two conductors land in
# one STL they solve as ONE electrode and every voltage downstream is
# wrong. Tessellate finely (linearDeflection <= 0.01 mm): a cylinder
# becomes a faceted prism in STL and then a staircase on the solver grid.'''


@dataclass
class CadEmitter:
    """Builds a runnable CadQuery script for a numbered-electrode assembly.

    title/preamble:  the docstring at the top (physics/provenance notes).
    params:          (NAME, value, comment) triples -> a named block.
    helpers:         extra CadQuery def/text a project needs before the
                     electrode expressions reference it (e.g. a BOX_SOLID
                     trim solid, a make_rod_pair helper).
    Each add_electrode(number, name, build_expr, ...) registers a solid;
    render() emits the assembly + cadDict + export_stls() + STEP save.
    """
    title: str
    params: List[Tuple] = field(default_factory=list)
    helpers: str = ""
    preamble: str = ""
    electrodes: List[_Electrode] = field(default_factory=list)
    step_name: str = "assembly.step"
    ndigits: int = 6

    def add_electrode(self, number, name, build_expr, color="gray",
                      label=None):
        for e in self.electrodes:
            if e.number == number:
                raise ValueError(
                    f"electrode number {number} already used by "
                    f"{e.name!r} — numbers are the basis index and must "
                    f"be unique (one conductor per number)")
        self.electrodes.append(_Electrode(number, name, build_expr,
                                          color, label))
        return self

    def render(self) -> str:
        if not self.electrodes:
            raise ValueError("CadEmitter.render: no electrodes added — a "
                             "script with an empty assembly is a mistake")
        nums = sorted(e.number for e in self.electrodes)
        pblock = (fmt_params(self.params, self.ndigits)
                  if self.params else "")
        adds = []
        for e in sorted(self.electrodes, key=lambda x: x.number):
            lbl = e.label or e.name
            adds.append(
                f'_solid = {e.build_expr}\n'
                f'assy.add(_solid, name="{e.number}", '
                f'color=cq.Color("{e.color}"))\n'
                f'cadDict["{e.number}"] = [_solid, "{lbl}"]')
        body = "\n\n".join(adds)
        return f'''"""
{self.title}
"""
import cadquery as cq

{("# ---- parameters (derived; regenerate, never hand-edit) ----" + chr(10) + pblock + chr(10)) if pblock else ""}
{(self.preamble + chr(10)) if self.preamble else ""}\
{(self.helpers + chr(10)) if self.helpers else ""}
cadDict = {{}}
assy = cq.Assembly()

{body}


{_EXPORT_DOC}
def export_stls(outdir="stl_out", tol=0.01, ang=0.1):
    import os
    os.makedirs(outdir, exist_ok=True)
    for idx, (solid, label) in cadDict.items():
        nm = label.split()[0]
        path = os.path.join(outdir, "%03d_%s.stl" % (int(idx), nm))
        cq.exporters.export(solid, path, tolerance=tol,
                            angularTolerance=ang)
    print("wrote %d STLs to %s/ (tol %.3f mm)" % (len(cadDict), outdir, tol))


# electrode numbers present: {nums}
# export_stls()
# assy.save("{self.step_name}")
'''
