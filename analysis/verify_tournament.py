#!/usr/bin/env python3
"""Check our see/tournament/ against the module the instructor released.

Background. `scripts/evaluate.py` and `scripts/eval_submission.py` both
import `see.tournament.runner`, and that package is absent from the working
tree of the handed-out repository, so both scripts die with ImportError. We
first rebuilt it from the protocol in docs/DESIGN.md Sec. 10, then noticed
the handout is a *git* repository in which the files had only been deleted
from the working tree, and restored the originals with

    git checkout HEAD -- see/tournament/ benchmarks/

On 2026-08-09 the instructor posted a `tournament/` folder to the class
group as the official fix. This script compares that folder against what we
have been running all along.

Line endings are normalised before comparison: our copy came out of a
Windows git checkout (CRLF), the posted zip was made on macOS (LF). That
difference changes every byte of every line and none of the behaviour, so
comparing raw bytes would report "completely different" for files that are
in fact the same.

    python analysis/verify_tournament.py <path to extracted tournament dir>
"""
import argparse
import hashlib
import io
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = ("__init__.py", "runner.py", "leaderboard.py")


def norm(p):
    return p.read_bytes().replace(b"\r\n", b"\n").rstrip() + b"\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("official", help="extracted tournament/ directory")
    ap.add_argument("--out", default=str(ROOT / "analysis" /
                                         "verify_tournament.txt"))
    a = ap.parse_args()

    off = pathlib.Path(a.official)
    ours = ROOT / "see" / "tournament"
    out = ["see/tournament/ : ours vs the module released 2026-08-09", ""]
    out.append("%-16s %18s %18s   %s"
               % ("file", "official (norm)", "ours (norm)", "verdict"))
    same = True
    for n in FILES:
        po, pu = off / n, ours / n
        if not po.exists() or not pu.exists():
            out.append("%-16s %s" % (n, "MISSING on one side"))
            same = False
            continue
        ho = hashlib.sha256(norm(po)).hexdigest()
        hu = hashlib.sha256(norm(pu)).hexdigest()
        ok = ho == hu
        same &= ok
        out.append("%-16s %18s %18s   %s"
                   % (n, ho[:16], hu[:16], "IDENTICAL" if ok else "DIFFERS"))

    out.append("")
    if same:
        out.append("All three files are identical once line endings are")
        out.append("normalised. Every number in this project was therefore")
        out.append("already produced by the official runner -- which is also")
        out.append("why the four exactly-known board references reproduce to")
        out.append("the decimal on all four of our uploads.")
        out.append("")
        out.append("We deliberately keep our CRLF copy rather than dropping")
        out.append("the LF files in. The two are functionally identical, and")
        out.append("`git status --porcelain see/` returning empty is our")
        out.append("evidence for instruction 5.3 (the engine is untouched).")
        out.append("Overwriting the files would break that check to no end.")
    else:
        out.append("MISMATCH -- results computed here may not match the board.")

    text = "\n".join(out)
    io.open(a.out, "w", encoding="utf-8").write(text + "\n")
    sys.stdout.write(text + "\n")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
