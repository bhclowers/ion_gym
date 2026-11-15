"""
When the PE surface's 'barrier' electrode plateaus cannot be measured,
the figure itself says the heights are raw DC.

The fallback plots q*V_DC instead of the measured barrier -- a different
quantity on the same axis -- and said so only on the console. The figure
now carries it in a footer line and in layout.meta["pe_fallback"], which
the PE tab repeats on its status pane.
"""
import numpy as np
import scipy.ndimage

from ion_gym.viz import pe_view


def _surface():
    x = np.linspace(0.0, 10.0, 21)
    y = np.linspace(0.0, 6.0, 13)
    X, Y = np.meshgrid(x, y, indexing="ij")
    PE = 0.1 * (X - 5.0) ** 2
    ele = np.zeros(PE.shape, np.int16)
    ele[:3, :] = 1
    ele[-3:, :] = 2
    return x, y, PE, ele


def _footers(fig):
    return " ".join(str(a.text) for a in (fig.layout.annotations or ()))


def test_a_failed_measurement_labels_the_figure(monkeypatch):
    real = scipy.ndimage.binary_dilation

    def _fail_on_the_ring(a, iterations=1, **kw):
        if iterations == 2 + 1:              # the plateau-ring dilation
            raise ValueError("ring dilation failed")
        return real(a, iterations=iterations, **kw)
    monkeypatch.setattr(scipy.ndimage, "binary_dilation", _fail_on_the_ring)
    fig = pe_view.pe_figure_3d(surface=_surface(), metal_mode="barrier",
                               trust_cells=2, electrode_dc={1: 5.0, 2: 0.0})
    note = fig.layout.meta["pe_fallback"]
    assert "ValueError" in note and "ring dilation failed" in note
    assert "RAW DC" in _footers(fig)


def test_a_measured_barrier_carries_no_label():
    fig = pe_view.pe_figure_3d(surface=_surface(), metal_mode="barrier",
                               trust_cells=2, electrode_dc={1: 5.0, 2: 0.0})
    meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
    assert "pe_fallback" not in meta
    assert "RAW DC" not in _footers(fig)
