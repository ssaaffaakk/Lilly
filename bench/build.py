#!/usr/bin/env python3
"""Derive bench/terms.tsv and bench/cases.tsv by counting, not by intuition.

The metric has to answer one question — is Lilly reading *Bosnian*, or generic
Serbo-Croatian — and it may not answer it with sentences or word lists written
here. Writing the exam and sitting it are the same act; the result would be a
tautology. So every pair and every sentence below comes out of translations
professionals already made:

    data/flores/{dev,devtest}.bs      FLORES-200, professional Bosnian
    data/extra/extra-train.tsv        the NTREX-128 rows, professional
    data/clean/{train,test}.tsv       the SETIMES rows, Bosnian news, human

and one corpus that is counted but is not evidence:

    data/raw/OpenSubtitles/*.bs       OpenSubtitles v2024, 18.5M subtitle lines

News and Wikipedia contain no Turkisms at all, so rejecting kahva and čaršija
on a count of zero was a fact about the register, not about Bosnian.
OpenSubtitles is the register those words are spoken in, and it is now counted
so the rejection can be read off real numbers. It gets no vote: measured, it is
about a third ekavica and half Croatian by its own markers, and adding it to
the evidence would eject svijet and fudbal rather than admit anything. The long
note above OPENSUBS has the arithmetic and the conclusion, which is that the
0.98 gate cannot admit a Turkism from any corpus.

A pair is admitted only if the counts say professionals chose the Bosnian form
and did not choose the alternative. The alternative must also turn up somewhere
in the repo's mixed-BCS text (WikiMatrix + Wikipedia), or a count of zero in the
Bosnian sources proves nothing — it could be a word nobody anywhere writes.

Which of the sources a case may be drawn from is a contamination question, not
a style one. FLORES is the project's held-out eval set and has never been
trained on. NTREX sits in data/extra/extra-train.tsv, which
training/train_translation.py does not read — but training/PREREGISTRATION.md
schedules those rows into the next corpus, and on that day NTREX cases stop
being held out and this file has to be rebuilt without them. SETIMES cases come
from the test split only; the train split is in the fine-tune.

    python3 bench/build.py

Three or four minutes, most of it counting SETIMES; rewrites both TSVs and
bench/terms-rejected.tsv from the corpora each time.
"""
import collections
import hashlib
import itertools
import json
import unicodedata
import csv
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "bench"

# One definition of "letter", so the streamed OpenSubtitles pass draws its word
# boundaries exactly where the tokeniser below draws them.
LETTERS = "a-zA-ZčćžšđČĆŽŠĐ"
WORD = re.compile(f"[{LETTERS}]+", re.UNICODE)
ENWORD = re.compile(r"[A-Za-z]+")

# Evidence corpora: professional Bosnian. Counts that decide admission come only
# from these three.
BOS = ("flores", "ntrex", "setimes")
# Case corpora: the subset of the evidence the current fine-tune has not been
# trained on, so a case cannot be scored by recall of a memorised row.
CASE_SOURCES = ("flores", "ntrex", "setimes_test")

MIN_BOS = 20        # below this the share is a coin flip, not a measurement
MIN_SHARE = 0.98    # professionals must pick the Bosnian form ~always
MIN_ALT_MIXED = 1   # the rejected form has to be a real word someone writes

# English words too common to carry evidence that a Bosnian term was understood.
STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been
being it its this that these those he she they we you i his her their our your not no than then
there here which who whom whose what when where why how all any both each few more most other
some such only own same so too very can will just should now also has have had do does did up
down out over under again further once about into after before between during above below
against because while until s t d ll m re ve y""".split())


def read_tsv(path, want):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 3 and p[0] == want and p[1].strip() and p[2].strip():
                rows.append((p[1].strip(), p[2].strip()))
    return rows


def load_corpora():
    c = {}
    flores = []
    for split in ("dev", "devtest"):
        bs = (REPO / "data/flores" / f"{split}.bs").read_text(encoding="utf-8").splitlines()
        en = (REPO / "data/flores" / f"{split}.en").read_text(encoding="utf-8").splitlines()
        if len(bs) != len(en):
            raise SystemExit(f"flores {split}: {len(bs)} bs against {len(en)} en")
        flores += [(a.strip(), b.strip()) for a, b in zip(bs, en) if a.strip() and b.strip()]
    c["flores"] = flores
    c["ntrex"] = read_tsv(REPO / "data/extra/extra-train.tsv", "ntrex")
    c["setimes"] = read_tsv(REPO / "data/clean/train.tsv", "SETIMES")
    c["setimes_test"] = read_tsv(REPO / "data/clean/test.tsv", "SETIMES")
    # Mixed BCS: web-mined and Wikipedia, not professional and not reliably
    # Bosnian. Used for one thing only — to show a rejected form is a real word.
    c["mixed"] = (read_tsv(REPO / "data/clean/train.tsv", "WikiMatrix")
                  + read_tsv(REPO / "data/extra/extra-train.tsv", "wikimedia"))
    return c


CORP = load_corpora()
FREQ, INDEX, EN_DF = {}, {}, {}
for name, rows in CORP.items():
    f, ix, df = collections.Counter(), collections.defaultdict(list), collections.Counter()
    for i, (bs, en) in enumerate(rows):
        toks = [w.lower() for w in WORD.findall(bs)]
        f.update(toks)
        for w in set(toks):
            ix[w].append(i)
        df.update(set(w.lower() for w in ENWORD.findall(en)))
    FREQ[name], INDEX[name], EN_DF[name] = f, ix, df


def n(form, corpus):
    return FREQ[corpus][form.lower()]


def n_bos(form):
    return sum(FREQ[c][form.lower()] for c in BOS)


def occurrences(form, corpora):
    for c in corpora:
        for i in INDEX[c].get(form.lower(), ()):
            yield (c, i) + CORP[c][i]


def gloss(form, corpora=BOS, top=4, min_p=0.25, min_lift=20, keep_ratio=0.7):
    """What professionals wrote in English when this Bosnian form appeared.

    Scored by P(english word | the term is in the sentence), not by a Dice
    coefficient. Dice rewards rarity, and rarity is what a collocation has: it
    ranked "eurovision" above "song" for pjesmu and "getty" alongside "left"
    for lijevo, because those ride along in a handful of captions. The
    conditional probability puts the translation first every time. `lift` then
    throws out words that are merely common, and keep_ratio keeps a runner-up
    only when it is nearly as reliable as the best — which holds "train"
    beside "trains" and drops "getty" beside "left".

    Nobody types the accepted answers. Typing them would make the exam and the
    answer key the same document.
    """
    rows = list(occurrences(form, corpora))
    if len(rows) < 3:
        return []
    co = collections.Counter()
    for _, _, _, en in rows:
        co.update(set(w.lower() for w in ENWORD.findall(en)))
    df, total = collections.Counter(), 0
    for c in corpora:
        df.update(EN_DF[c])
        total += len(CORP[c])
    scored = []
    for w, k in co.items():
        if w in STOP or len(w) <= 2 or w.isdigit() or not df[w]:
            continue
        p = k / len(rows)
        if p >= min_p and p / (df[w] / total) >= min_lift:
            scored.append((p, k, w))
    scored.sort(reverse=True)
    scored = scored[:top]
    cut = scored[0][0] * keep_ratio if scored else 0
    return [(w, k, round(p, 3)) for p, k, w in scored if p >= cut]


CONS = "bcčćdđfghjklmnpqrsštvwxzž"

# "prijem" (reception) keeps its ije in Serbian too — the substitution produces
# "prem", which the counts cannot reject because "prem" is not a word anywhere.
# It is the one candidate left standing that the rules below do not catch.
NOT_YAT = {"prijem"}


def capitalised(form, threshold=0.5):
    """Proper names carry an ije that is not a yat — "Alijem" is the name Ali,
    not a word with a Serbian counterpart. Measured, not guessed at.

    Only capitals inside a sentence count. Counting sentence-initial ones as
    well threw out "tijelo" and "svjetski", which are ordinary words that happen
    to open headlines.
    """
    rows = [bs for _, _, bs, _ in occurrences(form, BOS)]
    if not rows:
        return False
    upper = form.capitalize()
    caps = 0
    for bs in rows:
        for m in re.finditer(rf"\b{re.escape(upper)}\b", bs):
            before = bs[:m.start()].rstrip(" \u201e\u201c\"'(")
            if before and before[-1] not in ".!?:":
                caps += 1
                break
    return caps / len(rows) >= threshold

# The hypothesis space, and only that. Every entry below is a distinction the
# BCS contrastive literature and docs/BOSNIAN_METRIC.md name; which of them
# survives is decided underneath by n_bos/share, never here. Entries that fail
# are printed as rejected and kept out of terms.tsv.
LEXICAL = [
    # id, category, bosnian forms, alternative forms, which variety the alternative is
    ("sedmica", "lex", ["sedmica", "sedmice", "sedmici", "sedmicu", "sedmicama"],
     ["tjedan", "tjedna", "tjednu", "tjedni", "tjedana", "nedelja", "nedelje", "nedelju"], "hr/sr"),
    ("historija", "lex", ["historija", "historije", "historiju", "historiji", "historijski",
                          "historijska", "historijske", "historijskog", "historijskom", "historijskih"],
     ["istorija", "istorije", "istoriju", "istoriji", "istorijski", "istorijska", "istorijske",
      "istorijskog", "povijest", "povijesti", "povijesni", "povijesna"], "sr/hr"),
    ("hiljada", "lex", ["hiljada", "hiljadu", "hiljade", "hiljadama"],
     ["tisuća", "tisuću", "tisuće", "tisućama"], "hr"),
    ("općina", "lex", ["općina", "općine", "općini", "općinu", "općinama", "općinski",
                       "općinskog", "općinskih", "općinske"],
     ["opština", "opštine", "opštini", "opštinu", "opštinama", "opštinski", "opštinskog",
      "opštinskih", "opštinske"], "sr"),
    ("opći", "lex", ["opći", "opća", "opće", "općeg", "općem", "općih", "općenito"],
     ["opšti", "opšta", "opšte", "opšteg", "opštem", "opštih", "opšte"], "sr"),
    ("utjecaj", "lex", ["utjecaj", "utjecaja", "utjecaju", "utjecali", "utjecalo", "utjecati"],
     ["uticaj", "uticaja", "uticaju", "uticali", "uticalo", "uticati"], "sr"),
    ("univerzitet", "lex", ["univerzitet", "univerziteta", "univerzitetu", "univerziteti",
                            "univerzitetski", "univerzitetskog"],
     ["sveučilište", "sveučilišta", "sveučilištu", "sveučilišni"], "hr"),
    ("fudbal", "lex", ["fudbal", "fudbala", "fudbalu", "fudbalski", "fudbalskog", "fudbaler",
                       "fudbalera", "fudbaleri"],
     ["nogomet", "nogometa", "nogometu", "nogometni", "nogometaš", "nogometaša"], "hr"),
    ("avion", "lex", ["avion", "aviona", "avionu", "avioni", "avionima", "avionska", "avionski"],
     ["zrakoplov", "zrakoplova", "zrakoplovu", "zrakoplovi"], "hr"),
    ("porodica", "lex", ["porodica", "porodice", "porodici", "porodicu", "porodicama",
                         "porodični", "porodičnog", "porodične"],
     ["obitelj", "obitelji", "obiteljima", "obiteljski", "obiteljskog"], "hr"),
    ("tačka", "lex", ["tačka", "tačke", "tački", "tačku", "tačno", "tačan", "tačna"],
     ["točka", "točke", "točki", "točku", "točno", "točan", "točna"], "hr"),
    ("voz", "lex", ["voz", "voza", "vozu", "vozovi", "vozova"],
     ["vlak", "vlaka", "vlaku", "vlakovi", "vlakova"], "hr"),
    ("štampa", "lex", ["štampa", "štampe", "štampi", "štampu", "štampanje", "štampani"],
     ["tisak", "tiska", "tisku"], "hr"),
    ("ambasada", "lex", ["ambasada", "ambasade", "ambasadi", "ambasadu", "ambasador",
                         "ambasadora", "ambasadoru"],
     ["veleposlanstvo", "veleposlanstva", "veleposlanstvu", "veleposlanik", "veleposlanika"], "hr"),
    ("Evropa", "lex", ["evropa", "evrope", "evropi", "evropu", "evropski", "evropska", "evropske",
                       "evropskog", "evropskom", "evropskih", "evropskoj", "evropskim"],
     ["europa", "europe", "europi", "europu", "europski", "europska", "europske", "europskog",
      "europskom", "europskih", "europskoj"], "hr"),
    ("oktobar", "lex", ["januar", "januara", "februar", "februara", "april", "aprila", "juni",
                        "juna", "juli", "jula", "august", "augusta", "septembar", "septembra",
                        "oktobar", "oktobra", "novembar", "novembra", "decembar", "decembra"],
     ["siječanj", "siječnja", "veljača", "veljače", "travanj", "travnja", "lipanj", "lipnja",
      "srpanj", "srpnja", "kolovoz", "kolovoza", "rujan", "rujna", "listopad", "listopada",
      "studeni", "studenog", "prosinac", "prosinca"], "hr"),
    ("saobraćaj", "lex", ["saobraćaj", "saobraćaja", "saobraćaju", "saobraćajni", "saobraćajne",
                          "saobraćajnih"],
     ["prometni", "prometne", "prometnih", "prometa"], "hr"),
    ("učestvovati", "lex", ["učestvovati", "učestvovao", "učestvuju", "učestvovali", "učestvuje",
                            "učestvovala", "učešće", "učešća"],
     ["sudjelovati", "sudjelovao", "sudjeluju", "sudjelovali", "sudjeluje", "sudjelovanje"], "hr"),
    ("hemija", "lex", ["hemija", "hemije", "hemijski", "hemijska", "hemijskog", "hemijske",
                       "hemijskih", "hemijsko"],
     ["kemija", "kemije", "kemijski", "kemijska", "kemijskog", "kemijske"], "hr"),
    # h kept where Serbian drops it
    ("duhan", "h", ["duhan", "duhana", "duhanu", "duhanski", "duhanske", "duhanskih"],
     ["duvan", "duvana", "duvanu", "duvanski", "duvanske"], "sr"),
    ("kuhati", "h", ["kuhati", "kuha", "kuhanje", "kuhano", "kuhinja", "kuhinje", "kuhinji"],
     ["kuvati", "kuva", "kuvanje", "kuvano", "kujna", "kujne"], "sr"),
    ("kahva", "h", ["kahva", "kahve", "kahvu", "kahvom", "kahvana", "kahvane"],
     ["kafa", "kafe", "kafu", "kafom", "kava", "kave", "kavu", "kafana", "kafane"], "sr/hr"),
    ("lahko", "h", ["lahko", "lahak", "lahka", "lahke"], ["lako", "lako", "lagan", "lagano"], "sr/hr"),
    ("mehko", "h", ["mehko", "mehak", "mehka"], ["meko", "mekan", "mekano", "mekana"], "sr/hr"),
    ("sahat", "h", ["sahat", "sahata", "sahati"], ["sat", "sata", "sati", "satu"], "sr/hr"),
    ("suh", "h", ["suh", "suha", "suho", "suhi", "suhe"], ["suv", "suva", "suvo", "suvi", "suve"], "sr"),
    ("uho", "h", ["uho", "uha", "uhu"], ["uvo", "uva", "uvu"], "sr"),
    ("hudovica", "h", ["hudovica", "hudovice"], ["udovica", "udovice", "udovicu"], "sr/hr"),
    # Turkish/Ottoman loans
    ("čaršija", "turkism", ["čaršija", "čaršije", "čaršiji", "čaršiju", "čaršijom"],
     ["trg", "trga", "pijaca", "pijace", "tržnica", "tržnice"], "sr/hr"),
    ("avlija", "turkism", ["avlija", "avlije", "avliji", "avliju"],
     ["dvorište", "dvorišta", "dvorištu"], "sr/hr"),
    ("komšija", "turkism", ["komšija", "komšije", "komšiju", "komšijama", "komšiluk", "komšiluku"],
     ["susjed", "susjeda", "susjedi", "susjedima", "susjeda"], "sr/hr"),
    ("dućan", "turkism", ["dućan", "dućana", "dućanu", "dućani"],
     ["prodavnica", "prodavnice", "prodavnici", "radnja", "radnje"], "sr/hr"),
    ("sokak", "turkism", ["sokak", "sokaka", "sokaku", "sokake"],
     ["ulica", "ulice", "ulici", "ulicu"], "sr/hr"),
    ("mahala", "turkism", ["mahala", "mahale", "mahali", "mahalu"],
     ["kvart", "kvarta", "naselje", "naselja"], "sr/hr"),
    ("ćuprija", "turkism", ["ćuprija", "ćuprije", "ćupriji"], ["most", "mosta", "mostu"], "sr/hr"),
    ("insan", "turkism", ["insan", "insana", "insani"], ["čovjek", "čovjeka", "čovjeku"], "sr/hr"),
    ("čorba", "turkism", ["čorba", "čorbe", "čorbu"], ["supa", "supe", "juha", "juhe"], "sr/hr"),
    ("bašča", "turkism", ["bašča", "bašče", "bašču"], ["vrt", "vrta", "vrtu", "bašta", "bašte"], "sr/hr"),
    # Household and feeling words: the register news never reaches. All four
    # read zero in the professional corpora, and the subtitle columns say
    # whether that is Bosnian not using them or this project not hearing them —
    # jorgan 73 and merak 23 across 18.5M subtitle lines, pendžer and sevdah 0.
    ("pendžer", "turkism", ["pendžer", "pendžera", "pendžeru", "pendžeri"],
     ["prozor", "prozora", "prozoru", "prozori"], "sr/hr"),
    ("jorgan", "turkism", ["jorgan", "jorgana", "jorganu", "jorgani"],
     ["deka", "deke", "deku", "pokrivač", "pokrivača"], "sr/hr"),
    ("merak", "turkism", ["merak", "meraka", "meraku", "meraklija"],
     ["užitak", "užitka", "uživanje", "uživanja"], "sr/hr"),
    ("sevdah", "turkism", ["sevdah", "sevdaha", "sevdahu", "sevdalinka", "sevdalinke"],
     ["čežnja", "čežnje", "ljubav", "ljubavi"], "sr/hr"),
]

# A form written twice in one of the lists above gets counted twice, and every
# consumer below sums over the list: admit(), the per-corpus columns, and the
# OpenSubtitles counts. Three lists had a repeat — "opšte", "lako", "susjeda" —
# so opći, lahko and komšija were each measured against an alternative inflated
# by one whole form, which pushes the Bosnian share down. It changed no verdict
# (opći passes either way, the other two fail either way), but it made komšija
# read 0.55 in the subtitle column where it is 0.62. Deduplicate once, here,
# instead of asking every future reader of the lists to spot a repeat.
LEXICAL = [(tid, cat, list(dict.fromkeys(bos)), list(dict.fromkeys(alt)), variety)
           for tid, cat, bos, alt, variety in LEXICAL]


# ---------------------------------------------------------------------------
# OpenSubtitles: the register the Turkisms live in, and why it does not vote
# ---------------------------------------------------------------------------
# Every Turkism above was rejected on a count of zero: kahva, čaršija, avlija,
# sokak, mahala do not occur once in news or Wikipedia, so the rejection said
# nothing about Bosnian and everything about the register. Subtitles are where
# those words are spoken, so OpenSubtitles v2024 bs-en — 18,477,297 pairs — is
# counted here and its counts are written into both TSVs.
#
# It is counted, and it does not vote. Three measurements, reprinted by every
# build so they are not a claim anyone has to take on trust:
#
#   variety     67% of its 471,655 yat markers are ijekavian, so about a third
#               of the corpus is Serbian; 31% of its hr/bs lexical markers are
#               the Croatian one (tko 48,110 against ko 127,692; povijest 1,551
#               against historija 250). OPUS labels it bs. It is BCS.
#   consequence adding it to BOS would eject the benchmark's own terms, not
#               admit new ones. Every one of the 17 admitted lex and h terms
#               falls under the 0.98 gate on the combined counts, and so does
#               the yat family: it writes svijet 14,828 times and svet 6,170,
#               a 0.71 share. A corpus that rejects the terms it is supposed to
#               corroborate is not evidence about them.
#   effect      it does not rescue a single Turkism anyway. Cut to the 2.6% of
#               it that is ijekavian *and* not Croatian by its own markers —
#               476,500 lines — kahva reaches a 0.035 share against kafa/kava,
#               dućan 0.034, mahala 0.286, komšija 0.779. Nothing reaches 0.98.
#
# The block test behind `effect` is coarse and is meant to be: a block that
# straddles two subtitle files reads as mixed and is thrown out, so the test can
# only overstate how Croatian or Serbian the corpus is — it errs against the
# conclusion it is used to support.
#
# `effect` is the finding, and it is about this file, not about the corpus.
# The 0.98 gate works on yat pairs because they are in complementary
# distribution — a Bosnian text writes svijet and never svet. Turkisms are not:
# kafa, ulica, most, prozor, susjed and vrt are all ordinary Bosnian, so the
# Turkism and its "alternative" sit in the same Bosnian sentence and the share
# cannot approach 1 in any corpus, however pure. Loosening the gate until they
# pass would not be finding evidence, it would be lowering the bar until the
# absence of evidence stopped showing, so the gate is left where it is and the
# counts are published instead.
#
# Licence: OPUS redistributes text it does not own. Use requires citing Lison &
# Tiedemann (2016), OpenSubtitles2016 (LREC), and linking opensubtitles.org.
OPENSUBS = REPO / "data/raw/OpenSubtitles/OpenSubtitles.bs"
OPENSUBS_CACHE = REPO / "data/raw/OpenSubtitles/.form-counts.json"

# Yat and hr/bs markers, counted to establish what variety the corpus is. Paired
# so each row is one word two ways; "što" is deliberately absent, because
# Bosnian uses it too and it would not be a marker.
PROBE_YAT = [("mlijeko", "mleko"), ("dijete", "dete"), ("vrijeme", "vreme"),
             ("cvijet", "cvet"), ("svijet", "svet"), ("rijeka", "reka"),
             ("lijep", "lep"), ("bijeli", "beli"), ("vjera", "vera"),
             ("mjesto", "mesto"), ("djeca", "deca"), ("čovjek", "čovek"),
             ("gdje", "gde"), ("nedjelja", "nedelja"), ("sjever", "sever")]
PROBE_HR = [("tko", "ko"), ("tisuću", "hiljadu"), ("nogomet", "fudbal"),
            ("kruh", "hljeb"), ("otok", "ostrvo"), ("glazba", "muzika"),
            ("povijest", "historija"), ("tjedan", "sedmica"), ("tvrtka", "firma"),
            ("uvjet", "uslov"), ("zrak", "vazduh"), ("vlak", "voz")]


# A subtitle file's lines are contiguous in the moses release and there are no
# document ids in it, so a fixed run of lines stands in for a document. 500 is
# under a typical film's line count, so a block usually sits inside one file;
# the blocks that straddle two are mixed, and mixed blocks fail the test below,
# which is the safe direction to be wrong in.
BLOCK = 500
BLOCK_MIN_MARKERS = 10   # under this a ratio is a coin flip, as with MIN_BOS
BLOCK_IJE, BLOCK_HR = 0.9, 0.1


def opensubtitles_scan():
    """Count the LEXICAL forms in OpenSubtitles, streaming, in one pass.

    Counted twice: over the whole corpus, and over the part of it that is
    Bosnian by its own orthography — ijekavian and not Croatian, block by
    block. The second count is what answers "is the Turkism rejection the
    corpus's fault or the rule's", so it is computed here rather than asserted
    in a comment. The two marker families that select the blocks are yat and
    hr/bs lexical doublets; the Turkisms then counted inside them are neither,
    so the selection is not the measurement in disguise.

    The yat family is left out of the term scan on purpose. Those 200-odd pairs
    are settled by the professional corpora, adding them would treble the size
    of the alternation, and a count from a corpus that is not admitted as
    evidence would only be decoration next to counts that are.

    About an hour over 570 MB the first time, then cached against the file's
    size and mtime. Returns None when the corpus is not on disk, and build.py
    runs without it — the columns come out empty and nothing else changes,
    because these counts decide nothing.
    """
    forms = sorted({f for _, _, bos, alt, _ in LEXICAL for f in bos + alt}
                   | {w for pair in PROBE_YAT + PROBE_HR for w in pair})
    if not OPENSUBS.exists():
        return None
    st = OPENSUBS.stat()
    # Fingerprint the content, not the mtime: re-running fetch_opensubtitles.py
    # unpacks the same bytes with a new mtime, and an hour of counting should
    # not be thrown away because a file was rewritten identically. Size plus the
    # ends of the file is enough to notice a different corpus.
    # `shape` invalidates the cache when what the pass computes changes rather
    # than what it reads — a cache written by an older version of this function
    # is missing fields the report needs.
    fp = hashlib.sha1()
    with open(OPENSUBS, "rb") as f:
        fp.update(f.read(1 << 20))
        if st.st_size > (2 << 20):
            f.seek(-(1 << 20), 2)
            fp.update(f.read())
    key = {"shape": 3, "size": st.st_size, "fingerprint": fp.hexdigest(),
           "forms": hashlib.sha1("\n".join(forms).encode()).hexdigest(),
           "blocks": [BLOCK, BLOCK_MIN_MARKERS, BLOCK_IJE, BLOCK_HR]}
    if OPENSUBS_CACHE.exists():
        cached = json.loads(OPENSUBS_CACHE.read_text(encoding="utf-8"))
        if cached.get("key") == key:
            return cached
    print(f"  counting {OPENSUBS.name} ({st.st_size / 1e6:.0f} MB) — "
          f"once, then cached in {OPENSUBS_CACHE.name}")

    rx = re.compile(f"(?<![{LETTERS}])("
                    + "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
                    + f")(?![{LETTERS}])")
    # The leak filter's own set, reused: OpenSubtitles is not in the training
    # corpus by provenance, but subtitle lines are short and stock phrases
    # collide, so the overlap is measured rather than argued from provenance.
    trained = trained_on()
    ije_words = {a for a, _ in PROBE_YAT}
    eka_words = {b for _, b in PROBE_YAT}
    hr_words = {a for a, _ in PROBE_HR}
    bs_words = {b for _, b in PROBE_HR}
    counts, bosnian_counts = collections.Counter(), collections.Counter()
    lines = overlap = overlap_in_window = in_window = 0
    blocks = bosnian_blocks = bosnian_lines = 0
    with open(OPENSUBS, encoding="utf-8", errors="replace") as f:
        while batch := list(itertools.islice(f, BLOCK)):
            lines += len(batch)
            blocks += 1
            hits = collections.Counter(
                m.group(1) for m in rx.finditer("".join(batch).lower()))
            counts.update(hits)
            ije = sum(v for w, v in hits.items() if w in ije_words)
            eka = sum(v for w, v in hits.items() if w in eka_words)
            hr = sum(v for w, v in hits.items() if w in hr_words)
            bs = sum(v for w, v in hits.items() if w in bs_words)
            if (ije + eka >= BLOCK_MIN_MARKERS and hr + bs >= BLOCK_MIN_MARKERS
                    and ije / (ije + eka) >= BLOCK_IJE
                    and hr / (hr + bs) <= BLOCK_HR):
                bosnian_blocks += 1
                bosnian_lines += len(batch)
                bosnian_counts.update(hits)
            for line in batch:
                s = unicodedata.normalize("NFC", line.strip())
                if not s:
                    continue
                window = MIN_WORDS <= len(WORD.findall(s)) <= MAX_WORDS
                in_window += window
                if s in trained:
                    overlap += 1
                    overlap_in_window += window
    out = {"key": key, "lines": lines, "counts": dict(counts), "overlap": overlap,
           "in_window": in_window, "overlap_in_window": overlap_in_window,
           "blocks": blocks, "bosnian_blocks": bosnian_blocks,
           "bosnian_lines": bosnian_lines, "bosnian_counts": dict(bosnian_counts)}
    OPENSUBS_CACHE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


_SCAN = None


def spoken(forms):
    """Occurrences of `forms` in OpenSubtitles, or None if it is not on disk."""
    if _SCAN is None:
        return None
    return sum(_SCAN["counts"].get(f.lower(), 0) for f in forms)


def yat_candidates():
    """Every ijekavica/ekavica pair the corpora themselves offer.

    Not a list anyone typed: take each frequent Bosnian word, apply the two yat
    rules (long ije -> e, short je -> e) and keep the pairs whose ekavica side
    is attested in the mixed-BCS text. lj and nj are digraphs, so a 'je' after
    l or n is not a short yat; a word-final -ije whose -ija nominative exists is
    an ija-stem genitive, not a yat either. Both filters are spelling facts, not
    judgements about which word is Bosnian.
    """
    pool = collections.Counter()
    for c in BOS:
        pool.update(FREQ[c])
    out = []
    for w, k in pool.items():
        if k < 30 or len(w) < 5 or w in NOT_YAT or capitalised(w):
            continue
        alts = set()
        for m in re.finditer("ije", w):
            if m.end() == len(w) and (w[:m.start()] + "ija") in pool:
                continue
            # -iji is the comparative and the ciji/koji pronoun ending, and its
            # -ije form is the same in Serbian: "kasnije" stays "kasnije", it
            # does not become "kasne". Attested stem+iji is the tell.
            if (w[:m.start()] + "iji") in pool:
                continue
            alts.add(w[:m.start()] + "e" + w[m.end():])
        for m in re.finditer("je", w):
            if m.start() == 0 or w[m.start() - 1] in "ln" or w[m.start() - 1] not in CONS:
                continue
            alts.add(w[:m.start()] + "e" + w[m.end():])
        for a in alts:
            if a != w and n(a, "mixed") >= MIN_ALT_MIXED:
                out.append((w, a))
    return sorted(out, key=lambda p: -pool[p[0]])


def admit(bos_forms, alt_forms):
    nb = sum(n_bos(w) for w in bos_forms)
    na = sum(n_bos(w) for w in alt_forms)
    nm = sum(n(w, "mixed") for w in alt_forms)
    share = nb / (nb + na) if nb + na else 0.0
    ok = nb >= MIN_BOS and share >= MIN_SHARE and nm >= MIN_ALT_MIXED
    return ok, nb, na, nm, share


def build_terms():
    terms, rejected = [], []
    for w, a in yat_candidates():
        ok, nb, na, nm, share = admit([w], [a])
        row = dict(term_id=w, category="yat", bosnian_forms=w, alt_forms=a, alt_variety="sr",
                   n_flores=n(w, "flores"), n_ntrex=n(w, "ntrex"), n_setimes=n(w, "setimes"),
                   n_bosnian_sources=nb, n_alt_in_bosnian_sources=na,
                   n_alt_in_mixed_bcs=nm, bosnian_share=round(share, 4),
                   n_opensubtitles="", n_alt_opensubtitles="",
                   opensubtitles_share="",
                   accept_en="|".join(g[0] for g in gloss(w)))
        (terms if ok else rejected).append(row)

    for tid, cat, bos_forms, alt_forms, variety in LEXICAL:
        ok, nb, na, nm, share = admit(bos_forms, alt_forms)
        acc = collections.Counter()
        for w in bos_forms:
            for en, k, d in gloss(w):
                acc[en] += k
        # Reported, never added to nb/na: see the note above OPENSUBS for the
        # arithmetic that says why a mixed-BCS corpus cannot cast a vote here.
        sb, sa = spoken(bos_forms), spoken(alt_forms)
        row = dict(term_id=tid, category=cat, bosnian_forms="|".join(bos_forms),
                   alt_forms="|".join(alt_forms), alt_variety=variety,
                   n_flores=sum(n(w, "flores") for w in bos_forms),
                   n_ntrex=sum(n(w, "ntrex") for w in bos_forms),
                   n_setimes=sum(n(w, "setimes") for w in bos_forms),
                   n_bosnian_sources=nb, n_alt_in_bosnian_sources=na,
                   n_alt_in_mixed_bcs=nm, bosnian_share=round(share, 4),
                   n_opensubtitles="" if sb is None else sb,
                   n_alt_opensubtitles="" if sa is None else sa,
                   opensubtitles_share="" if sb is None or not (sb + sa)
                   else round(sb / (sb + sa), 4),
                   accept_en="|".join(w for w, _ in acc.most_common(8)))
        (terms if ok else rejected).append(row)
    return terms, rejected


COLUMNS = ["term_id", "category", "bosnian_forms", "alt_forms", "alt_variety",
           "n_flores", "n_ntrex", "n_setimes", "n_bosnian_sources",
           "n_alt_in_bosnian_sources", "n_alt_in_mixed_bcs", "bosnian_share",
           "n_opensubtitles", "n_alt_opensubtitles", "opensubtitles_share",
           "accept_en"]


def write_terms(terms, rejected):
    with open(BENCH / "terms.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for row in sorted(terms, key=lambda r: (r["category"], -r["n_bosnian_sources"])):
            w.writerow(row)
    with open(BENCH / "terms-rejected.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for row in sorted(rejected, key=lambda r: (r["category"], -r["n_bosnian_sources"])):
            w.writerow(row)


# A variant sentence is the professional sentence with one word swapped for its
# Serbian/Croatian counterpart — nobody writes a new sentence, and the English
# reference stays the professional's. Only swaps that leave gender and
# declension alone are listed, because "Univerzitet je" -> "Sveučilište je"
# changes agreement elsewhere in the sentence and would measure that instead.
VARIANT_MAP = {
    "tačka": {"tačka": "točka", "tačke": "točke", "tački": "točki", "tačku": "točku",
              "tačno": "točno", "tačan": "točan", "tačna": "točna"},
    "Evropa": {"evropa": "europa", "evrope": "europe", "evropi": "europi", "evropu": "europu",
               "evropski": "europski", "evropska": "europska", "evropske": "europske",
               "evropskog": "europskog", "evropskom": "europskom", "evropskih": "europskih",
               "evropskoj": "europskoj", "evropskim": "europskim"},
    "oktobar": {"januar": "siječanj", "januara": "siječnja", "februar": "veljača",
                "februara": "veljače", "april": "travanj", "aprila": "travnja",
                "juni": "lipanj", "juna": "lipnja", "juli": "srpanj", "jula": "srpnja",
                "august": "kolovoz", "augusta": "kolovoza", "septembar": "rujan",
                "septembra": "rujna", "oktobar": "listopad", "oktobra": "listopada",
                "novembar": "studeni", "novembra": "studenog", "decembar": "prosinac",
                "decembra": "prosinca"},
    "općina": {"općina": "opština", "općine": "opštine", "općini": "opštini",
               "općinu": "opštinu", "općinama": "opštinama", "općinski": "opštinski",
               "općinskog": "opštinskog", "općinskih": "opštinskih", "općinske": "opštinske"},
    "opći": {"opći": "opšti", "opća": "opšta", "opće": "opšte", "općeg": "opšteg",
             "općem": "opštem", "općih": "opštih", "općenito": "uopšteno"},
    "utjecaj": {"utjecaj": "uticaj", "utjecaja": "uticaja", "utjecaju": "uticaju",
                "utjecali": "uticali", "utjecalo": "uticalo", "utjecati": "uticati"},
    "duhan": {"duhan": "duvan", "duhana": "duvana", "duhanu": "duvanu",
              "duhanski": "duvanski", "duhanske": "duvanske", "duhanskih": "duvanskih"},
    "historija": {"historija": "istorija", "historije": "istorije", "historiju": "istoriju",
                  "historiji": "istoriji", "historijski": "istorijski",
                  "historijska": "istorijska", "historijske": "istorijske",
                  "historijskog": "istorijskog", "historijskom": "istorijskom",
                  "historijskih": "istorijskih"},
    "hiljada": {"hiljada": "tisuća", "hiljadu": "tisuću", "hiljade": "tisuće",
                "hiljadama": "tisućama"},
    "porodica": {"porodica": "obitelj", "porodice": "obitelji", "porodici": "obitelji",
                 "porodicu": "obitelj", "porodicama": "obiteljima"},
    "avion": {"avion": "zrakoplov", "aviona": "zrakoplova", "avionu": "zrakoplovu",
              "avioni": "zrakoplovi", "avionima": "zrakoplovima"},
    "voz": {"voz": "vlak", "voza": "vlaka", "vozu": "vlaku", "vozovi": "vlakovi",
            "vozova": "vlakova"},
    "fudbal": {"fudbal": "nogomet", "fudbala": "nogometa", "fudbalu": "nogometu",
               "fudbalski": "nogometni", "fudbaler": "nogometaš", "fudbalera": "nogometaša"},
    "hemija": {"hemija": "kemija", "hemije": "kemije", "hemijski": "kemijski",
               "hemijska": "kemijska", "hemijskog": "kemijskog", "hemijske": "kemijske",
               "hemijskih": "kemijskih", "hemijsko": "kemijsko"},
    "saobraćaj": {"saobraćaj": "promet", "saobraćaja": "prometa", "saobraćaju": "prometu",
                  "saobraćajni": "prometni", "saobraćajne": "prometne",
                  "saobraćajnih": "prometnih"},
    "učestvovati": {"učestvovati": "sudjelovati", "učestvovao": "sudjelovao",
                    "učestvuju": "sudjeluju", "učestvovali": "sudjelovali",
                    "učestvuje": "sudjeluje", "učestvovala": "sudjelovala"},
}

# One budget per term_id. The yat family has 233 of them and the lexical family
# 15, so an equal cap would bury the lexical evidence under inflected forms of
# "mjesto"; the caps are set to bring the two families to a comparable size.
# How many cases one term may contribute. Sources are then filled greedily in a
# fixed order — flores, ntrex, setimes_test — so dropping cases from an earlier
# source frees quota that a later one takes up: when the leak filter removed 58
# ntrex cases, setimes_test grew from 34 to 40. That is arithmetic, not a
# choice. Nothing in this file reads a model's score, so the set cannot be
# tuned, knowingly or otherwise, to the numbers it is about to produce.
MAX_PER_TERM = {"yat": 2, "lex": 14, "h": 14, "turkism": 14}
MIN_WORDS, MAX_WORDS = 6, 40


def match_case(replacement, original):
    return replacement.capitalize() if original[:1].isupper() else replacement


def trained_on() -> set:
    """Every Bosnian sentence the shipped model was trained on.

    A benchmark case that is in the training data measures memory, not
    understanding. This is not hypothetical here: NTREX was added to the
    training corpus and is also one of this benchmark's sources, and 61 of the
    first 398 cases — 15.3% — turned out to be sentences the model had trained
    on. Holding 462 NTREX rows out was not enough, because the benchmark drew
    from all of NTREX rather than from the part that was held back.

    So the check lives here, where cases are made, rather than in a sentence on
    the model card claiming there is no overlap. A leaking case cannot enter the
    set.
    """
    seen = set()
    for name in ("train-mix.tsv", "train.tsv"):
        path = REPO / "data" / "clean" / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1].strip():
                seen.add(unicodedata.normalize("NFC", parts[1].strip()))
    return seen


def build_cases(terms):
    by_form = {}
    for row in terms:
        for form in row["bosnian_forms"].split("|"):
            variant = (VARIANT_MAP.get(row["term_id"], {}).get(form)
                       if row["category"] != "yat" else row["alt_forms"])
            by_form.setdefault(form, (row, variant))

    used, cases = collections.Counter(), []
    trained = trained_on()
    leaked = collections.Counter()
    # flores first: cleanest text and provably outside the training data. ntrex
    # is a benchmark source AND part of the training corpus, so every case is
    # checked against what was trained on rather than trusted by provenance.
    for corpus in CASE_SOURCES:
        for i, (bs, en) in enumerate(CORP[corpus]):
            if unicodedata.normalize("NFC", bs.strip()) in trained:
                leaked[corpus] += 1
                continue
            toks = WORD.findall(bs)
            if not (MIN_WORDS <= len(toks) <= MAX_WORDS):
                continue
            en_words = set(w.lower() for w in ENWORD.findall(en))
            targets = []
            for tok in toks:
                low = tok.lower()
                if low not in by_form:
                    continue
                row, variant = by_form[low]
                if (used[row["term_id"]] >= MAX_PER_TERM[row["category"]]
                        or any(t[0] == low for t in targets)):
                    continue
                # Accepted English = what professionals wrote for this term across
                # the corpora, narrowed to what this sentence's own translator
                # chose. An empty intersection means the term is not recoverable
                # from this reference, so the sentence cannot test it.
                accept = [g for g, _, _ in gloss(low) if g in en_words]
                if not accept:
                    continue
                targets.append((low, tok, row, variant, accept))
            if not targets:
                continue
            variant_bs, swapped = bs, []
            for low, surface, row, variant, accept in targets:
                if not variant:
                    continue
                new = re.sub(rf"\b{re.escape(surface)}\b", match_case(variant, surface),
                             variant_bs)
                if new != variant_bs:
                    variant_bs, _ = new, swapped.append(f"{surface}>{match_case(variant, surface)}")
            for low, surface, row, variant, accept in targets:
                used[row["term_id"]] += 1
            cases.append(dict(
                case_id=f"{corpus}-{i}",
                source=corpus,
                bs=bs,
                en=en,
                terms="|".join(t[0] for t in targets),
                categories="|".join(t[2]["category"] for t in targets),
                accept_en="|".join(",".join(t[4]) for t in targets),
                variant_bs=variant_bs if swapped else "",
                variant_swaps="|".join(swapped),
            ))
    if leaked:
        print(f"  dropped as already trained on: {dict(leaked)}")
    return cases


CASE_COLUMNS = ["case_id", "source", "bs", "en", "terms", "categories", "accept_en",
                "variant_bs", "variant_swaps"]


def report_opensubtitles(terms, rejected):
    """What the subtitle corpus is, and what it did to the Turkisms.

    Printed every build rather than written down once, because all three of
    these are claims about a file on disk and they should be re-derived from it.
    """
    if _SCAN is None:
        print("\nopensubtitles: not on disk — python3 bench/fetch_opensubtitles.py")
        print("  the h and turkism families are then counted from news and wiki only,")
        print("  where they read zero. Nothing else in this build depends on it.")
        return
    c = _SCAN["counts"]
    ije = sum(c.get(a, 0) for a, _ in PROBE_YAT)
    eka = sum(c.get(b, 0) for _, b in PROBE_YAT)
    hr = sum(c.get(a, 0) for a, _ in PROBE_HR)
    bs = sum(c.get(b, 0) for _, b in PROBE_HR)
    print(f"\nopensubtitles v2024, bs side: {_SCAN['lines']:,} lines")
    print(f"  variety: {ije / (ije + eka):.0%} of {ije + eka:,} yat markers are ijekavian, "
          f"{hr / (hr + bs):.0%} of {hr + bs:,} hr/bs markers are the Croatian one")
    print("  → mixed BCS, not a Bosnian source. Counted below; admits nothing.")
    # The reason it gets no vote, computed rather than asserted: add these
    # counts to the professional ones, re-run the admission rule, and see what
    # survives. Nothing does.
    counted = [r for r in terms if r["n_opensubtitles"] != ""]
    ejected = [r["term_id"] for r in counted
               if (r["n_bosnian_sources"] + r["n_opensubtitles"]
                   + r["n_alt_in_bosnian_sources"] + r["n_alt_opensubtitles"])
               and (r["n_bosnian_sources"] + r["n_opensubtitles"])
               / (r["n_bosnian_sources"] + r["n_opensubtitles"]
                  + r["n_alt_in_bosnian_sources"] + r["n_alt_opensubtitles"]) < MIN_SHARE]
    sv, se = c.get("svijet", 0), c.get("svet", 0)
    print(f"  if it did vote: {len(ejected)} of the {len(counted)} admitted lex and h "
          f"terms would fall under the {MIN_SHARE:.0%} gate and leave the benchmark")
    if sv + se:
        print(f"    the yat family goes the same way — svijet {sv:,} against svet "
              f"{se:,} is a {sv / (sv + se):.2f} share in this corpus")
    print("    a corpus that rejects svijet cannot corroborate kahva, so it is")
    print("    counted, reported, and never allowed to decide anything")
    print(f"  already in the training corpus: {_SCAN['overlap']:,} lines "
          f"({_SCAN['overlap'] / _SCAN['lines']:.2%}), of which "
          f"{_SCAN['overlap_in_window']:,} are in the {MIN_WORDS}..{MAX_WORDS}-word "
          f"case window ({_SCAN['overlap_in_window'] / max(_SCAN['in_window'], 1):.4%} "
          f"of {_SCAN['in_window']:,})")

    fams = ("turkism", "h")
    rows = [r for r in terms + rejected if r["category"] in fams
            and r["n_opensubtitles"] != ""]
    gate20 = [r for r in rows if r["n_opensubtitles"] >= MIN_BOS]
    gate98 = [r for r in gate20 if r["opensubtitles_share"] != ""
              and r["opensubtitles_share"] >= MIN_SHARE]
    print(f"\nthe admission rule, applied to opensubtitles counts "
          f"({len(rows)} turkism and h terms):")
    print(f"  clear >= {MIN_BOS} occurrences: {len(gate20)}")
    print(f"  clear >= {MIN_SHARE:.0%} share against the alternative: {len(gate98)}")
    for r in sorted(rows, key=lambda r: -(r["opensubtitles_share"] or 0)):
        share = r["opensubtitles_share"]
        print(f"  {r['term_id']:10} {r['category']:8} "
              f"n={r['n_opensubtitles']:6d} alt={r['n_alt_opensubtitles']:6d} "
              f"share={share if share != '' else 0:.4f}"
              f"{'' if r['n_opensubtitles'] >= MIN_BOS else '   (too rare to measure)'}")
    # The corpus is mixed, so the obvious objection to the table above is that
    # the alternatives are winning because half the text is Croatian. Answer it
    # by counting again inside the blocks that are Bosnian by their own
    # orthography, which is a different feature from the one being counted.
    bc = _SCAN["bosnian_counts"]
    print(f"\n  counted again in the {_SCAN['bosnian_blocks']:,} of "
          f"{_SCAN['blocks']:,} blocks that are ijekavian and not Croatian "
          f"({_SCAN['bosnian_lines']:,} lines, "
          f"{_SCAN['bosnian_lines'] / _SCAN['lines']:.1%} of the corpus):")
    # kuhati and duhan are already admitted on the professional counts, so their
    # subtitle share is not evidence about anything still open. "Best" below
    # means best among the terms this rule currently rejects.
    open_terms = {r["term_id"] for r in rejected}
    best = None
    for r in rows:
        nb = sum(bc.get(f, 0) for f in r["bosnian_forms"].split("|"))
        na = sum(bc.get(f, 0) for f in r["alt_forms"].split("|"))
        if not nb:
            continue
        share = nb / (nb + na)
        # Only terms that clear the occurrence gate can be "the best share":
        # one lucky hit out of one is a share of 1.0 and means nothing.
        if nb >= MIN_BOS and r["term_id"] in open_terms:
            best = max(best or (0, ""), (share, r["term_id"]))
        print(f"    {r['term_id']:10} n={nb:5d} alt={na:5d} share={share:.4f}"
              f"{'   ADMITTED' if nb >= MIN_BOS and share >= MIN_SHARE else ''}"
              f"{'' if r['term_id'] in open_terms else '   (already admitted)'}")

    if not gate98:
        present = [r for r in rows if r["n_opensubtitles"] > 0]
        loudest = max(rows, key=lambda r: r["n_opensubtitles"])
        print(f"\n  no term admitted. {len(present)} of the {len(rows)} do occur here — "
              f"{loudest['term_id']} {loudest['n_opensubtitles']:,} times — but so do")
        print("  their alternatives, which are ordinary Bosnian too. A share gate built")
        print("  for yat pairs, which are in complementary distribution, cannot admit a")
        print("  word whose alternative is also correct. That is a fact about the rule,")
        if best:
            print(f"  not about the data: of the terms still rejected, the best share in "
                  f"Bosnian-only text is {best[0]:.2f} ({best[1]}).")


def main():
    global _SCAN
    _SCAN = opensubtitles_scan()
    terms, rejected = build_terms()
    write_terms(terms, rejected)
    cases = build_cases(terms)
    with open(BENCH / "cases.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, CASE_COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for c in cases:
            w.writerow(c)

    cats = collections.Counter(r["category"] for r in terms)
    print(f"terms admitted {len(terms)}  rejected {len(rejected)}   {dict(cats)}")
    print(f"cases {len(cases)}  with a variant {sum(1 for c in cases if c['variant_bs'])}")
    print("  by source:", dict(collections.Counter(c["source"] for c in cases)))
    print("\nrejected, with the counts that rejected them:")
    for r in sorted(rejected, key=lambda r: -r["n_bosnian_sources"])[:40]:
        print(f"  {r['term_id']:14} {r['category']:8} bs={r['n_bosnian_sources']:6d} "
              f"alt_in_bs={r['n_alt_in_bosnian_sources']:6d} share={r['bosnian_share']:.2f}")
    report_opensubtitles(terms, rejected)


if __name__ == "__main__":
    main()
