#!/usr/bin/env python3
"""Ekavian -> ijekavian rewriter for Serbian sentence data (Lane C pilot).

Whole-word-form lookup against a curated lexicon. A token is rewritten only if
its lowercase whole form is in the lexicon; case is preserved (uppercase,
title-case, lowercase). No substring matching, no suffix morphing, no
morphology tools.

Lexicon sources (merged in this order):
  1. base  : bench/terms.tsv rows where category == "yat" and alt_variety
             == "sr". Only rows whose bosnian_forms/alt_forms pipe-lists have
             equal length contribute (aligned positionally, ekavian -> ijekavian).
  2. supple: a small curated supplement of frequent unambiguous families
             (reka, dete, mleko, hleb, gde/ovde, resiti) not covered by bench.
Then EXCLUSIONS are removed. Every excluded form collides with a homograph
carrying a non-yat meaning:
  - svet family: "world" vs "holy"/"saint"/"council".
  - zahteva: noun "zahtev" gen.pl (-> zahtjeva) vs verb "zahtevati" 3sg
    (-> zahtijeva); the bare ekavian form is ambiguous, so it is left alone.
  - rekom: instrumental of "reka" is dominated in practice by the acronym
    REKOM (Regional Commission on establishing the facts), which must not be
    rewritten; excluding the form kills both the acronym bug and the rare
    legitimate instrumental.

The exclusion set, supplement, and measured precision are documented in
training/RESULTS-ekavica-precision.md.
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TERMS = REPO_ROOT / "bench" / "terms.tsv"

_WORD = re.compile(r"(\w+)")

EXCLUSIONS = frozenset(
    {
        "svet", "sveta", "svetu", "sveti", "svete", "svetom",
        "svetima", "svetova", "svetovima",
        "zahteva", "rekom",
    }
)

SUPPLEMENT = {
    "reka": "rijeka",
    "reku": "rijeku",
    "rekom": "rijekom",
    "rekama": "rijekama",
    "dete": "dijete",
    "deteta": "djeteta",
    "detetu": "djetetu",
    "mleko": "mlijeko",
    "mleka": "mlijeka",
    "mleku": "mlijeku",
    "hleb": "hljeb",
    "hleba": "hljeba",
    "hlebu": "hljebu",
    "hlebom": "hljebom",
    "gde": "gdje",
    "ovde": "ovdje",
    "resiti": "riješiti",
    "resio": "riješio",
    "resila": "riješila",
    "resenje": "rješenje",
}


def _load_base(terms_path):
    lexicon = {}
    with open(terms_path, encoding="utf-8") as fh:
        for line in fh:
            row = line.rstrip("\n").split("\t")
            if len(row) < 5:
                continue
            term_id, category, bosnian, alt, variety = row[0], row[1], row[2], row[3], row[4]
            if category != "yat" or variety != "sr":
                continue
            bosnian_forms = bosnian.split("|")
            alt_forms = alt.split("|")
            if len(bosnian_forms) != len(alt_forms):
                continue
            for ek, ij in zip(alt_forms, bosnian_forms):
                ek = ek.strip().lower()
                ij = ij.strip().lower()
                if ek and ij:
                    lexicon[ek] = ij
    return lexicon


def build_lexicon(terms_path=DEFAULT_TERMS):
    lexicon = _load_base(terms_path)
    lexicon.update({k: v for k, v in SUPPLEMENT.items() if k not in EXCLUSIONS})
    for ex in EXCLUSIONS:
        lexicon.pop(ex, None)
    return lexicon


def _rewrite_token(token, lexicon):
    key = token.lower()
    if key not in lexicon:
        return token
    value = lexicon[key]
    if token.isupper():
        return value.upper()
    if token[:1].isupper():
        return value.capitalize()
    return value


class EkavicaToIjekavica:
    def __init__(self, terms_path=DEFAULT_TERMS):
        self.lexicon = build_lexicon(terms_path)

    def convert_text(self, text):
        return _WORD.sub(lambda m: _rewrite_token(m.group(1), self.lexicon), text)

    def audit_text(self, text):
        hits = []
        for m in _WORD.finditer(text):
            token = m.group(1)
            if token.lower() in self.lexicon:
                hits.append((token, self.lexicon[token.lower()]))
        return hits


def _main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="free-form text to rewrite")
    parser.add_argument("--file", help="read newline-separated lines from file")
    parser.add_argument("--audit", action="store_true", help="also print per-token rewrites")
    parser.add_argument("--terms", default=str(DEFAULT_TERMS), help="path to bench/terms.tsv")
    args = parser.parse_args(argv)

    converter = EkavicaToIjekavica(args.terms)
    total_rewritten = 0

    def process_line(line):
        nonlocal total_rewritten
        if args.audit:
            hits = converter.audit_text(line)
            total_rewritten += len(hits)
            return converter.convert_text(line), hits
        out = converter.convert_text(line)
        total_rewritten += len(converter.audit_text(line))
        return out, None

    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            for line in fh:
                out, hits = process_line(line.rstrip("\n"))
                if args.audit:
                    tokens = " ".join(f"{t}>{r}" for t, r in hits)
                    print(f"{out}\t{tokens}" if tokens else out)
                else:
                    print(out)
    elif args.text is not None:
        out, hits = process_line(args.text)
        print(out)
        if args.audit:
            for t, r in hits:
                print(f"{t}\t{r}")
    else:
        for line in sys.stdin:
            out, hits = process_line(line.rstrip("\n"))
            if args.audit:
                tokens = " ".join(f"{t}>{r}" for t, r in hits)
                print(f"{out}\t{tokens}" if tokens else out)
            else:
                print(out)

    print(f"lexicon_forms={len(converter.lexicon)} rewritten_tokens={total_rewritten}", file=sys.stderr)


if __name__ == "__main__":
    _main(sys.argv[1:])