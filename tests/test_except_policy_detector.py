"""
The except-policy gate's detector (tools/except_policy.scan_source, which
tests/test_except_policy.py runs) sees every way review F10 found to
slip a swallow past the gate.

Each snippet below passed the old gate as clean.
"""
import pytest

from tools.except_policy import scan_source

SWALLOWS = {
    "tuple": "try:\n    f()\nexcept (ValueError, Exception):\n    pass\n",
    "one_tuple": "try:\n    f()\nexcept (Exception,):\n    pass\n",
    "alias": "B = Exception\ntry:\n    f()\nexcept B:\n    pass\n",
    "builtins": ("import builtins\ntry:\n    f()\n"
                 "except builtins.Exception:\n    pass\n"),
    "dead_raise": ("try:\n    f()\nexcept Exception:\n    if 0:\n"
                   "        raise\n    x = sor()\n"),
    "nested_def_raise": ("try:\n    f()\nexcept Exception:\n"
                         "    def g():\n        raise\n    x = sor()\n"),
    "caught_raise": ("try:\n    f()\nexcept Exception:\n    try:\n"
                     "        raise\n    except Exception as e2:\n"
                     "        print(e2)\n    x = sor()\n"),
}


@pytest.mark.parametrize("src", list(SWALLOWS.values()), ids=list(SWALLOWS))
def test_a_broad_swallow_is_counted_and_refused(src):
    outer = 1 + next(i for i, ln in enumerate(src.splitlines())
                     if ln.startswith("except"))
    broad, problems, _silent = scan_source(src, "m.py")
    assert outer in [ln for ln, _t in broad], broad
    assert any(p.startswith(f"line {outer}:") for p in problems), \
        "a broad handler that does not report must be refused"


def test_a_narrow_swallow_is_seen():
    src = ("try:\n    f()\nexcept (ValueError, TypeError, KeyError, "
           "IndexError, OSError, AttributeError, LookupError):\n    pass\n")
    broad, problems, silent = scan_source(src, "m.py")
    assert broad == [] and problems == []
    assert silent and silent[0][1][0] == "m.py"
    assert silent[0][1][1] == "<module>"


def test_a_reporting_broad_handler_counts_but_passes():
    src = ("def cb(self):\n    try:\n        f()\n    except Exception as e:"
           "\n        self.status.object = f'failed: {e}'\n")
    broad, problems, silent = scan_source(src, "m.py")
    assert len(broad) == 1 and problems == [] and silent == []


def test_a_real_reraise_is_not_counted():
    src = ("try:\n    f()\nexcept Exception as e:\n"
           "    raise RuntimeError('x') from e\n")
    assert scan_source(src, "m.py") == ([], [], [])


def test_silent_handlers_are_keyed_by_function_and_types():
    src = ("class A:\n    def m(self):\n        try:\n            f()\n"
           "        except (TypeError, KeyError):\n            return None\n")
    _b, _p, silent = scan_source(src, "m.py")
    assert silent[0][1] == ("m.py", "A.m", ("KeyError", "TypeError"))
