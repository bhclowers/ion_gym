"""test_shipped_text.py -- shipped files carry technical reasoning only.

Every file the public cut includes is documentation a stranger will read.
It must explain what the code does and why; it must not carry the
development apparatus: tracking-row ids, dated decisions, people,
version histories, paths into the development-only tree, the names of
its design documents, or the project's former name. That
provenance lives in the development tree, where it is kept as records
that point AT the code; the code does not point back.

Without a gate this drifts every working week, and the drift is only
discovered when a public cut is prepared, as a scrub of hundreds of
lines. This gate moves the discovery to the edit that introduced it.

WHAT IS SCANNED. Exactly the public manifest, taken from the release
packer itself, so the gate and the cut cannot disagree about which
files ship. Python is tokenized and only comments and string literals
are read (an identifier such as a pi constant is not prose); notebooks
are read cell by cell; every other text format is read line by line.
A file that cannot be read this way (not UTF-8, not tokenizable, a
notebook without cells) stops the gate with its name: a file the gate
cannot read is a file it cannot vouch for, so it is never skipped.

RATCHET. Existing debt is recorded per file and per pattern in a
baseline kept in the development tree. A count may fall, never rise:
a new hit fails this gate naming the file, line and text, even when an
unrelated cleanup elsewhere removed more than it added. When counts
fall, `--tighten` lowers the baseline to match; it never raises one.
A hit that is genuinely required in shipped text (an attribution line,
an executable string whose change would alter behaviour) belongs in
ALLOW below with its reason, not in the baseline.

In a public checkout the development tree is absent; there is nothing
to ratchet against (that tree's output IS what you are holding), so the
gate says so and exits cleanly.
"""
import _bootstrap  # noqa: F401
import io
import json
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "internal" / "tools"
BASELINE = TOOLS / "shipped_text_baseline.json"

# What the development apparatus looks like in prose. Deliberately
# narrow: each pattern is near-certain provenance when it appears in a
# shipped comment or string, so a failure is a real finding.
PATTERNS = {
    "tracking_id": r"\bL-\d{2,4}[a-z]?\b",
    # (not JavaScript's Math.PI, in the editor's embedded viewer code)
    "pi_reference": r"(?<!Math\.)\bPI\b",
    "person": r"\b(?:Brian|Clowers)\b",
    "iso_date": r"\b20\d\d-\d\d-\d\d\b",
    "dev_tree_path": r"\binternal/",
    "version_tag": r"\bv\d{3}\b",
    # design documents of the development tree, cited by name
    "charter": r"\b[Cc]harters?\b|CHARTER",
    # the project's earlier name
    "old_name": r"\bion_playground\b",
}

# Permanent, reasoned exceptions: (path, regex the LINE must match).
ALLOW = [
    # attribution is required in what ships
    ("README.md", r"^Copyright \(C\) 2026 Brian Clowers"),
    ("README.md", r"^Developed in the Clowers Research Group, Washington "),
    ("ion_gym/__init__.py", r"^Copyright \(C\) 2026 Brian Clowers"),
    ("pyproject.toml", r'^authors = \[\{name = "Brian Clowers"\}\]'),
    # this file names the patterns it enforces
    ("tests/test_shipped_text.py", r"."),
    # an ignore rule the gate asserts the public .gitignore carries; it is
    # executable data, and changing it would change what the gate checks.
    # Python strings are read token by token, so the match is the token.
    ("tests/test_ignore_rules.py", r'^"internal/"$'),
    # lint excludes for development-tree folders: executable config that
    # keeps a development checkout's lint run off retired code; in the
    # public cut the folders are absent and the lines do nothing
    ("ruff.toml", r'^"internal/[\w/]+",$'),
    # a salt hashed into the 3-D STL field cache key: changing the bytes
    # would re-key every cached 3-D STL field
    ("ion_gym/physics/build_stl3d.py", r'^b"mirror-bc-fix-2026-07-19"$'),
]

TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".txt", ".cfg", ".yml",
                 ".yaml", ".ipynb", ".js", ".html", ".css"}


class UnreadableShippedFile(Exception):
    """A shipped text file the gate cannot read as text it can scan."""


def _py_prose(src, where):
    """(line_no, text) for every comment and string-literal line."""
    out = []
    keep = {tokenize.COMMENT, tokenize.STRING}
    fmid = getattr(tokenize, "FSTRING_MIDDLE", None)
    if fmid is not None:
        keep.add(fmid)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, SyntaxError) as e:
        raise UnreadableShippedFile(
            f"{where}: cannot be tokenized as Python ({type(e).__name__}: "
            f"{e}), so its comments and strings cannot be told apart from "
            f"code. Fix the source; the gate does not guess.") from e
    for tok in toks:
        if tok.type in keep:
            for k, ln in enumerate(tok.string.splitlines()):
                out.append((tok.start[0] + k, ln))
    return out


def _prose(path, rel):
    try:
        src = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise UnreadableShippedFile(
            f"{rel}: ships with a text suffix but is not UTF-8 ({e}). "
            f"Re-encode it, or give it a suffix the gate does not read as "
            f"text if it is binary.") from e
    if path.suffix == ".py":
        return _py_prose(src, rel)
    if path.suffix == ".ipynb":
        nb = json.loads(src)
        if "cells" not in nb:
            raise UnreadableShippedFile(
                f"{rel}: notebook JSON has no 'cells' list, so there is "
                f"nothing the gate can read; a notebook without cells is "
                f"malformed.")
        out = []
        for ci, cell in enumerate(nb["cells"]):
            text = "".join(cell["source"])
            pairs = (_py_prose(text, f"{rel} cell {ci}")
                     if cell["cell_type"] == "code"
                     else list(enumerate(text.splitlines(), 1)))
            out += [(f"cell {ci} line {n}", ln) for n, ln in pairs]
        return out
    return list(enumerate(src.splitlines(), 1))


def _allowed(rel, line):
    return any(rel == p and re.search(rx, line.strip()) for p, rx in ALLOW)


def scan(root=ROOT):
    """{rel: {pattern: [(where, text), ...]}} over the public manifest."""
    sys.path.insert(0, str(TOOLS))
    import pack_release
    keep, _dropped, _over = pack_release.collect(
        root, keep_study_data=False, public=True)
    found = {}
    for rel, _size in keep:
        path = root / rel
        r = rel.as_posix()
        # vendor/ holds third-party code shipped verbatim (three.js); its
        # text is not ours to rewrite, so it is outside this gate's scope
        if path.suffix not in TEXT_SUFFIXES or "/vendor/" in f"/{r}":
            continue
        for where, line in _prose(path, r):
            if _allowed(r, line):
                continue
            for name, rx in PATTERNS.items():
                if re.search(rx, line):
                    found.setdefault(r, {}).setdefault(name, []).append(
                        (where, line.strip()[:160]))
    return found


def counts(found):
    return {f: {p: len(h) for p, h in pats.items()}
            for f, pats in found.items()}


def main(argv):
    if not (TOOLS / "pack_release.py").is_file():
        print("shipped-text gate: SKIPPED -- no development tree here. "
              "This checkout is the public cut, which is this gate's "
              "output, so there is nothing to ratchet against.")
        return 0
    try:
        found = scan()
    except UnreadableShippedFile as e:
        print(f"SHIPPED-TEXT GATE: FAIL -- {e}")
        return 1
    now = counts(found)
    if "--init" in argv:
        if BASELINE.exists():
            print(f"REFUSED: {BASELINE} exists; --init would overwrite "
                  f"the ratchet. Use --tighten.")
            return 1
        BASELINE.write_text(json.dumps(now, indent=1, sort_keys=True) + "\n")
        print(f"baseline written: {sum(map(sum, (c.values() for c in now.values())))} "
              f"hits in {len(now)} files -> {BASELINE.name}")
        return 0
    if not BASELINE.is_file():
        print(f"REFUSED: no baseline at {BASELINE}. Create it once with "
              f"--init; after that it may only be tightened.")
        return 1
    base = json.loads(BASELINE.read_text())
    over, under = [], 0
    for f in sorted(set(now) | set(base)):
        for p in sorted(set(now.get(f, {})) | set(base.get(f, {}))):
            n, b = now.get(f, {}).get(p, 0), base.get(f, {}).get(p, 0)
            if n > b:
                over.append((f, p, n, b))
            elif n < b:
                under += b - n
    if "--tighten" in argv:
        if over:
            print("REFUSED: --tighten cannot run while counts are OVER the "
                  "baseline; fix those first (it never raises a count).")
        else:
            tight = {f: {p: n for p, n in pats.items() if n}
                     for f, pats in now.items()}
            tight = {f: pats for f, pats in tight.items() if pats}
            BASELINE.write_text(json.dumps(tight, indent=1, sort_keys=True)
                                + "\n")
            print(f"baseline tightened by {under} hits")
            return 0
    total = sum(sum(c.values()) for c in now.values())
    if over:
        print(f"SHIPPED-TEXT GATE: FAIL -- {len(over)} file/pattern counts "
              f"rose above the baseline:")
        for f, p, n, b in over:
            # the baseline holds counts, not lines, so the new hit cannot
            # be singled out; every hit of the pattern in that file is
            # listed and the new one is among them
            print(f"  {f} [{p}] {n} > baseline {b}; all {n} hits:")
            for where, text in found[f][p]:
                print(f"      {where}: {text}")
        print("Shipped text carries technical reasoning only; provenance "
              "belongs in the development tree. Rewrite the line, or, if "
              "it is genuinely required, add it to ALLOW with a reason.")
        return 1
    print(f"SHIPPED-TEXT GATE: PASS -- {total} hits, none above the "
          f"baseline" + (f"; {under} fewer than recorded (run --tighten "
                         f"to lock that in)" if under else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
