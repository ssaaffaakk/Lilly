#!/usr/bin/env python3
"""Is everything actually on GitHub, or does it only look finished from here?

    python3 scripts/state.py           # report, exit 1 if anything is missing
    python3 scripts/state.py --quiet   # exit code only

Run this before saying a step is done, and always before launching anything that
clones the repository.

Three things nearly cost this project a night's work on 27 August, all the same
shape: work that existed on this machine and not on GitHub.

  - The speech downloader had been rewritten and never committed. The Kaggle
    notebook clones from GitHub, so the training run that had just been launched
    pulled a version without the arguments the notebook passes it.
  - training/speech_bench.py was untracked. It computes the second of the two
    pre-registered thresholds the run about to finish would be judged against.
  - The photograph harvester, which collected everything the reader's 36% score
    is measured on, had never been in the repository at all.

`git status` shows the first two. It does not tell you whether the push landed,
and this checks that by asking GitHub for the commit rather than trusting an
exit code — raw.githubusercontent.com serves `/main/` from a CDN that handed
back a file two commits stale, which is exactly the kind of thing that reads as
success.
"""
import argparse
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE = "https://github.com/ssaaffaakk/Lilly"
# Working files that are meant to stay local. Anything untracked and outside
# this list is reported, because "it is only a scratch file" is a judgement the
# person reading the report should make, not one this script should make for
# them.
LOCAL_ONLY = (".log", ".pyc", ".swp")


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                          capture_output=True, text=True).stdout.strip()


def on_github(sha: str) -> bool:
    """Does GitHub have this commit? Asked by SHA, never by branch name."""
    request = urllib.request.Request(
        f"{REMOTE}/commit/{sha}", method="HEAD",
        headers={"User-Agent": "Lilly/state.py"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status == 200
    except urllib.error.HTTPError:
        return False
    except Exception as exc:                       # offline, DNS, timeout
        print(f"  (could not reach GitHub: {exc})", file=sys.stderr)
        return True                                # unknown, not "missing"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    modified = [l for l in git("status", "--porcelain").splitlines()
                if not l.startswith("??")]
    untracked = [l[3:] for l in git("status", "--porcelain").splitlines()
                 if l.startswith("??")]
    untracked = [f for f in untracked if not f.endswith(LOCAL_ONLY)]
    ahead = git("log", "--oneline", "@{u}..HEAD").splitlines()
    head = git("rev-parse", "HEAD")
    pushed = on_github(head) if head else False

    problems = []
    if modified:
        problems.append(f"{len(modified)} file(s) changed and not committed")
    if untracked:
        problems.append(f"{len(untracked)} untracked path(s) that are not logs")
    if ahead:
        problems.append(f"{len(ahead)} commit(s) not pushed")
    if not pushed:
        problems.append("GitHub does not have HEAD")

    if args.quiet:
        return 1 if problems else 0

    if not problems:
        print(f"clean — GitHub has {head[:8]}")
        return 0

    print("NOT FINISHED:")
    for p in problems:
        print(f"  - {p}")
    for group, rows in (("changed, uncommitted", modified),
                        ("untracked", untracked),
                        ("committed, unpushed", ahead)):
        if rows:
            print(f"\n{group}:")
            for row in rows[:20]:
                print(f"  {row}")
            if len(rows) > 20:
                print(f"  ... and {len(rows) - 20} more")
    print("\nUntracked paths are listed even when they look like working files: "
          "\nthe harvester that collected every photograph behind the reader's "
          "score \nlooked like a working file for a week.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
