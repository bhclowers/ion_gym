"""
IGNORE RULES: build metadata is never tracked, and the public repository's
ignore file keeps the development-only trees out.

`pip install` writes ion_gym.egg-info/ into the source tree. Once that
folder is tracked, every install shows it as modified, and a stale copy
(its SOURCES.txt, requires.txt and version) is committed and shipped as if
it described the package. The public cut writes its own root .gitignore
from a template kept in the development tree; it must ignore the
development-only folders, because copying a development tree over a
public checkout and running `git add -A` would otherwise commit them
without a word.

Checked:
  * no tracked path lies inside a *.egg-info folder (needs git and a
    repository; otherwise that check is reported as not run);
  * the root .gitignore ignores *.egg-info/;
  * the public .gitignore template, when this is a development tree,
    ignores *.egg-info/ and both development-only folders
    (PUBLIC_RULES).

Run: python tests/test_ignore_rules.py [repo_root]
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The public .gitignore template and the rules it must carry.
PUBLIC_TEMPLATE = ("internal", "tools", "public.gitignore")
PUBLIC_RULES = ("*.egg-info/", "internal/", ".claude/")


def _rules(path):
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")}


def check(root):
    fails, notes = [], []
    if shutil.which("git") is None or not (root / ".git").exists():
        notes.append("tracked-file check NOT RUN: no git executable or no "
                     "repository at this root")
    else:
        r = subprocess.run(["git", "-C", str(root), "ls-files"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            fails.append(f"git ls-files failed: {r.stderr.strip()}")
        else:
            meta = [p for p in r.stdout.splitlines()
                    if any(part.endswith(".egg-info") for part in p.split("/"))]
            for p in meta:
                fails.append(f"build metadata is tracked: {p}")
    dev = root / ".gitignore"
    if not dev.is_file():
        fails.append("no root .gitignore")
    elif "*.egg-info/" not in _rules(dev):
        fails.append(".gitignore does not ignore *.egg-info/")
    pub = root.joinpath(*PUBLIC_TEMPLATE)
    if pub.is_file():
        rel = pub.relative_to(root).as_posix()
        for r in PUBLIC_RULES:
            if r not in _rules(pub):
                fails.append(f"{rel} does not ignore {r}")
    else:
        notes.append("public ignore-file check NOT RUN: no development tree "
                     "(this checkout is the public cut)")
    return fails, notes


def main(argv):
    root = Path(argv[0]).resolve() if argv else HERE.parent
    fails, notes = check(root)
    for msg in notes:
        print(f"  NOTE {msg}")
    for msg in fails:
        print(f"  FAIL {msg}")
    print(f"IGNORE RULES ({root}): "
          + ("PASS" if not fails else f"FAIL, {len(fails)} problem(s)"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
