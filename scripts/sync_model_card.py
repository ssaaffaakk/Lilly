#!/usr/bin/env python3
"""Write the Hugging Face model card from the GitHub README.

    python3 scripts/sync_model_card.py            # rewrites models/lilly/README.md
    python3 scripts/sync_model_card.py --check    # exit 1 if the card is not what README.md would give

The owner's rule (8 September 2026): the card on Hugging Face says what the
README on GitHub says. So the card is generated, not maintained twice:

    YAML header      kept from the existing card (license, base models, tags)
    README.md body   with relative links and images pointed at GitHub
    card-only tail   the folder layout and the credits/licenses, which have to
                     travel with the weights (NOTICE.md is a licence condition)

Every number on the card therefore comes from one place. When README.md
changes, run this, commit both, and publish the card.
"""
import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
CARD = REPO_ROOT / "models" / "lilly" / "README.md"
GITHUB = "https://github.com/ssaaffaakk/Lilly/blob/main/"
RAW = "https://raw.githubusercontent.com/ssaaffaakk/Lilly/main/"
# Sections of the existing card that are not in the README and must stay.
TAIL_START = "# Credits and licenses"
LAYOUT_START = "## Layout"


def yaml_header(card_text: str) -> str:
    m = re.match(r"^---\n.*?\n---\n", card_text, re.S)
    if not m:
        raise SystemExit("the existing card has no YAML header to keep")
    return m.group(0)


def section(card_text: str, start: str, end: str | None) -> str:
    """From `start` up to `end`, or -- when end is None -- to the end of the
    card. For the layout section `end` is the next H1 heading, whatever it is,
    so the tool can re-read the card it wrote (where Credits follows Layout)."""
    a = card_text.find(start)
    if a < 0:
        raise SystemExit(f"the existing card has no {start!r} section")
    if end is None:
        b = len(card_text)
    else:
        b = card_text.find(end, a)
        if b < 0:
            raise SystemExit(f"the existing card has no {end!r} after {start!r}")
    return card_text[a:b].rstrip("\n") + "\n"


def rewrite_links(body: str) -> str:
    # images first: markdown ![alt](docs/images/x) and <img src="docs/images/x">
    body = re.sub(r"\]\((docs/images/[^)\s]+)\)", lambda m: f"]({RAW}{m.group(1)})", body)
    body = re.sub(r'src="(docs/images/[^"]+)"', lambda m: f'src="{RAW}{m.group(1)}"', body)
    # then every other relative link into the repository
    body = re.sub(r"\]\(((?:docs|training|models|bench|scripts|app|data|space|tests)/[^)\s]*)\)",
                  lambda m: f"]({GITHUB}{m.group(1)})", body)
    return body


def render() -> str:
    card_text = CARD.read_text(encoding="utf-8")
    body = README.read_text(encoding="utf-8")
    header = yaml_header(card_text)
    layout = section(card_text, LAYOUT_START, "\n# ")
    tail = section(card_text, TAIL_START, None)
    note = ("\n> This card is the project's README, mirrored here so the numbers live in one "
            "place (`scripts/sync_model_card.py`). The folder layout and the licenses that "
            "travel with these weights are at the end.\n\n")
    return (header + note + rewrite_links(body).rstrip("\n") + "\n\n---\n\n"
            + layout + "\n" + tail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare, do not write")
    args = ap.parse_args()
    text = render()
    if args.check:
        same = CARD.read_text(encoding="utf-8") == text
        print("model card matches README.md" if same else "model card is NOT what README.md would give; run scripts/sync_model_card.py")
        return 0 if same else 1
    CARD.write_text(text, encoding="utf-8")
    print(f"wrote {CARD} ({len(text):,} chars) from README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
