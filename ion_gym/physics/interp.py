"""interp.py -- generic grid interpolation. No device, no geometry.

PLACEMENT: this function operates on a CLASS of things --
any regularly-gridded 2-D array -- so it is general and lives in the
module for that class. It previously lived in `funnel_data`, which meant
every tracer that needed to interpolate a field imported a dependency on
one device's imported data pack: the arrow pointed core -> device data.
Nothing about bilinear interpolation is funnel-specific.

MEASURED BEHAVIOUR, not assumed: `_bilin` CLAMPS THE CELL INDEX to the
array interior but does NOT clamp the fractional part, so a query outside
the grid EXTRAPOLATES linearly off the edge cell rather than returning the
edge value. Bounds checking belongs in the caller (the tracer knows what
"outside the instrument" means; this function does not).
"""
from numba import njit


@njit(cache=True, nogil=True)
def _bilin(F, gx, gu, nx, nu):
    i = int(gx); j = int(gu)
    if i < 0: i = 0
    if j < 0: j = 0
    if i > nx - 2: i = nx - 2
    if j > nu - 2: j = nu - 2
    # Fraction CLAMPED. The index clamp alone turned any
    # out-of-grid sample into a linear EXTRAPOLATION of the last two
    # columns (fx grows past 1 without bound) -- fabricated field, not
    # solved field. Clamped, an out-of-grid sample holds the edge value.
    fx = gx - i; fu = gu - j
    if fx < 0.0: fx = 0.0
    if fx > 1.0: fx = 1.0
    if fu < 0.0: fu = 0.0
    if fu > 1.0: fu = 1.0
    return (F[i, j] * (1 - fx) * (1 - fu) + F[i + 1, j] * fx * (1 - fu)
            + F[i, j + 1] * (1 - fx) * fu + F[i + 1, j + 1] * fx * fu)
