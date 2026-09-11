"""Which standard is the Bosnian side of the training data actually written in?

Bosnian, Croatian and Serbian are mutually intelligible and share most of their
vocabulary, so no sentence can be labelled from grammar. What separates them is
a small set of words each standard uses to the exclusion of the others. This
counts those.

A sentence votes only if it carries a marker. Most will not, and that is the
expected result rather than a failure -- most sentences in any of the three
standards are the same sentence. The number that matters is the split *among
the sentences that do carry one*.

Read the Croatian column as "written to the Croatian standard, which is not the
Bosnian standard", not as "this text is foreign". Bosnian accepts a wider range
of forms than either neighbour, so the Bosnian-only list below is the shortest
of the three. That is the language, not an oversight.

Markers are whole words, not stems. A first version of this file used stems and
was wrong: `beo` (Serbian for white, Bosnian *bijel*) matched **Beograd**,
`vremen` matched **vremena** which is ijekavian too, and `vlak` (Croatian for
train) matched **vlakna**, fibres, which all three standards share. Every entry
here is a form that does not appear in standard Bosnian.

    python3 training/audit_bosnian.py [tsv ...]
"""
import re
import sys
from collections import Counter
from pathlib import Path

# Serbian: ekavian reflexes of yat where Bosnian is ijekavian, plus lexis.
# Listed as full forms -- the ekavian/ijekavian split does not survive
# stemming, since ijekavian *vrijeme* inflects to *vremena* like the ekavian.
SERBIAN_WORDS = """
vreme mleko mleka mleku dete deteta detetu deca decu deci decom
lep lepa lepo lepe lepi lepu lepog lepim lepše lepši
mesto mesta mestu mestom mestima
čovek čoveka čoveku čovekom
deo posle ovde gde sneg snega beo bela belo bele beli
telo tela telu nedelja nedelje nedelju
razumeo verovao verovati želeo hteo
predsednik predsednika predsedniku predsednika predsednički
"""
SERBIAN_STEMS = """
istorij opšt uopšte hleb hleba obezbedi razumev predsedni
"""

# Croatian forms that are not the Bosnian standard. The month names are the
# strongest family: Bosnian and Serbian both use januar/februar/mart.
CROATIAN_WORDS = """
vlak vlaka vlaku vlakom vlakovi vlakova
otok otoka otoku otoci otoke
tocka točka točke točku točaka
juha juhe glede unatoč
"""
CROATIAN_STEMS = """
tisuć povijes tjedan tjedno zrakoplov tipkovnic
siječnj siječanj veljač ožujk ožujak travnj travanj svibnj svibanj
lipnj lipanj srpnj srpanj kolovoz rujan rujna listopad prosinc prosinac
europsk kemij glazb tvrtk tijekom odvjetni knjižnic sveučilišt žlic
tiskovn nazoč priopć
"""

# Bosnian only: neither Croatian nor Serbian writes these.
#   historija -- Croatian povijest, Serbian istorija
#   sedmica   -- Croatian tjedan, Serbian nedelja
#   kahva     -- Croatian kava, Serbian kafa
#   lahko / mehko / sahat -- the retained /h/, the language's signature
# Deliberately excluded: hiljada (shared with Serbian), Evropa (shared with
# Serbian), and every word about Bosnia -- a sentence's topic is not its
# standard.
BOSNIAN_STEMS = """
historij sedmic kahv lahk mehk sahat mahram hudovic
"""


def build(words="", stems=""):
    alts = []
    for w in words.split():
        alts.append(w)
    for s in stems.split():
        alts.append(s + r"\w*")
    alts.sort(key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(alts) + r")\b", re.IGNORECASE)


PATTERNS = {
    "bosnian": build(stems=BOSNIAN_STEMS),
    "croatian": build(CROATIAN_WORDS, CROATIAN_STEMS),
    "serbian": build(SERBIAN_WORDS, SERBIAN_STEMS),
}
ORDER = ("bosnian", "croatian", "serbian", "mixed")


def classify(text):
    hits = {k: p.findall(text) for k, p in PATTERNS.items()}
    score = {k: len(v) for k, v in hits.items()}
    if not any(score.values()):
        return None, hits
    top = max(score, key=score.get)
    if sum(1 for v in score.values() if v == score[top]) > 1:
        return "mixed", hits
    return top, hits


def audit(path):
    per_corpus, words, total = {}, Counter(), 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0].strip():
                continue
            # 3 columns: corpus, Bosnian, English. 2: Bosnian, English.
            # 1: plain text, one sentence per line (FLORES, FLEURS).
            corpus, bs = (parts[0], parts[1]) if len(parts) > 2 else ("-", parts[0])
            total += 1
            verdict, hits = classify(bs)
            per_corpus.setdefault(corpus, Counter())[verdict or "no marker"] += 1
            if verdict in ("croatian", "serbian"):
                for w in hits[verdict]:
                    words[(verdict, w.lower())] += 1
    return total, per_corpus, words


def report(path):
    total, per_corpus, words = audit(path)
    print(f"\n=== {path}  ({total:,} pairs) ===\n")
    header = f"{'corpus':<14}{'pairs':>9}{'marked':>8}" + \
             "".join(f"{k:>13}" for k in ORDER)
    print(header)
    print("-" * len(header))

    def row(name, c):
        n = sum(c.values())
        marked = n - c["no marker"]
        cells = "".join(f"{c[k]:>8,}{100 * c[k] / marked if marked else 0:>5.0f}%"
                        for k in ORDER)
        print(f"{name[:13]:<14}{n:>9,}{marked:>8,}" + cells)
        return marked, n

    grand = Counter()
    for corpus, c in sorted(per_corpus.items(), key=lambda kv: -sum(kv[1].values())):
        grand.update(c)
        row(corpus, c)
    print("-" * len(header))
    marked, n = row("ALL", grand)
    print(f"\n{marked:,} of {n:,} sentences carry a marker ({100 * marked / n:.1f}%); "
          f"the percentages are shares of those {marked:,}, not of the corpus.")
    print("\ncommonest non-Bosnian forms found:")
    for (lang, w), k in words.most_common(20):
        print(f"  {lang:<9} {w:<18} {k:>7,}")


if __name__ == "__main__":
    for p in sys.argv[1:] or ["data/clean/train-mix.tsv"]:
        report(p) if Path(p).is_file() else print(f"missing: {p}", file=sys.stderr)
