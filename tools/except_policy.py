"""The exception-policy scanner behind tests/test_except_policy.py.

scan_source(src, fname) classifies every handler in one module's
source: broad handlers without a reachable raise (and whether each
reports the exception), and silent handlers keyed by enclosing
function and caught types. It lives here, not in tests/, because the
gate and its own detector test both use it and no test may import
from another test (dependency audit K9).
"""
import ast

BROAD_NAMES = {"Exception", "BaseException"}

# Why this is stricter than a name match: the gate once counted only a
# handler spelled exactly `except Exception` / `BaseException` / bare, took
# any `raise` anywhere in the handler (a dead `if 0:`, a nested def, a raise
# caught again) as a re-raise, never looked at narrow handlers, and never
# checked that an allowed handler reports. Each of those let a swallow
# through; scan_source closes them, and tests/test_except_policy_detector.py
# feeds it every snippet a review used to show it.


def _leaf(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _aliases(tree):
    """Module-level names bound to a broad class (B = Exception)."""
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and _leaf(n.value) in BROAD_NAMES:
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = _leaf(n.value)
    return out


def _types(h, aliases):
    if h.type is None:
        return ("BARE",)
    elts = h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
    out = []
    for e in elts:
        leaf = _leaf(e)
        out.append(aliases.get(leaf, leaf) if leaf else ast.unparse(e))
    return tuple(out)


def _reachable_raise(stmts):
    """A raise the handler can actually execute: not in a nested def or
    class, not under a constant-false `if`, not in a nested try body (its
    own handlers may catch it)."""
    for s in stmts:
        if isinstance(s, ast.Raise):
            return True
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef)):
            continue
        if isinstance(s, ast.If):
            dead = isinstance(s.test, ast.Constant) and not s.test.value
            if (not dead and _reachable_raise(s.body)) \
                    or _reachable_raise(s.orelse):
                return True
            continue
        if isinstance(s, ast.Try):
            if (any(_reachable_raise(x.body) for x in s.handlers)
                    or _reachable_raise(s.orelse)
                    or _reachable_raise(s.finalbody)):
                return True
            continue
        for field in ("body", "orelse"):
            if _reachable_raise(getattr(s, field, []) or []):
                return True
    return False


def _uses_exception(h):
    return h.name is not None and any(
        isinstance(n, ast.Name) and n.id == h.name
        for b in h.body for n in ast.walk(b))


def _silent(h):
    return not _uses_exception(h) and all(
        isinstance(s, (ast.Pass, ast.Continue))
        or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
        or (isinstance(s, ast.Return)
            and (s.value is None or isinstance(s.value, ast.Constant)))
        for s in h.body)


def _enclosing(tree):
    """handler -> dotted name of the def/class chain around it."""
    out = {}

    def walk(node, chain):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
                walk(ch, chain + [ch.name])
            else:
                if isinstance(ch, ast.ExceptHandler):
                    out[id(ch)] = ".".join(chain) or "<module>"
                walk(ch, chain)
    walk(tree, [])
    return out


def scan_source(src, fname):
    """(broad, problems, silent) for one module's source.
    broad    [(line, types)] broad handlers without a reachable raise
             (each counts against ALLOW);
    problems [str] broad handlers that do not report the exception;
    silent   [(line, (fname, function, sorted types))] silent handlers."""
    tree = ast.parse(src)
    aliases = _aliases(tree)
    where = _enclosing(tree)
    broad, problems, silent = [], [], []
    for h in ast.walk(tree):
        if not isinstance(h, ast.ExceptHandler):
            continue
        types = _types(h, aliases)
        is_broad = any(t in BROAD_NAMES or t == "BARE" for t in types)
        if is_broad and not _reachable_raise(h.body):
            broad.append((h.lineno, types))
            if not _uses_exception(h):
                problems.append(
                    f"line {h.lineno}: except {', '.join(types)} neither "
                    f"re-raises nor reports the exception (bind it and "
                    f"report it, or narrow it)")
        if _silent(h):
            silent.append((h.lineno,
                           (fname, where[id(h)], tuple(sorted(types)))))
    return broad, problems, silent
