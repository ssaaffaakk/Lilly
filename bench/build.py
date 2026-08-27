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
import csv
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "bench"

WORD = re.compile(r"[a-zA-ZčćžšđČĆŽŠĐ]+", re.UNICODE)
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
]


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
                   accept_en="|".join(g[0] for g in gloss(w)))
        (terms if ok else rejected).append(row)

    for tid, cat, bos_forms, alt_forms, variety in LEXICAL:
        ok, nb, na, nm, share = admit(bos_forms, alt_forms)
        acc = collections.Counter()
        for w in bos_forms:
            for en, k, d in gloss(w):
                acc[en] += k
        row = dict(term_id=tid, category=cat, bosnian_forms="|".join(bos_forms),
                   alt_forms="|".join(alt_forms), alt_variety=variety,
                   n_flores=sum(n(w, "flores") for w in bos_forms),
                   n_ntrex=sum(n(w, "ntrex") for w in bos_forms),
                   n_setimes=sum(n(w, "setimes") for w in bos_forms),
                   n_bosnian_sources=nb, n_alt_in_bosnian_sources=na,
                   n_alt_in_mixed_bcs=nm, bosnian_share=round(share, 4),
                   accept_en="|".join(w for w, _ in acc.most_common(8)))
        (terms if ok else rejected).append(row)
    return terms, rejected


COLUMNS = ["term_id", "category", "bosnian_forms", "alt_forms", "alt_variety",
           "n_flores", "n_ntrex", "n_setimes", "n_bosnian_sources",
           "n_alt_in_bosnian_sources", "n_alt_in_mixed_bcs", "bosnian_share", "accept_en"]


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
MAX_PER_TERM = {"yat": 2, "lex": 14, "h": 14, "turkism": 14}
MIN_WORDS, MAX_WORDS = 6, 40


def match_case(replacement, original):
    return replacement.capitalize() if original[:1].isupper() else replacement


def build_cases(terms):
    by_form = {}
    for row in terms:
        for form in row["bosnian_forms"].split("|"):
            variant = (VARIANT_MAP.get(row["term_id"], {}).get(form)
                       if row["category"] != "yat" else row["alt_forms"])
            by_form.setdefault(form, (row, variant))

    used, cases = collections.Counter(), []
    # flores and ntrex first: cleanest text, and neither is in the current
    # fine-tune's training data.
    for corpus in CASE_SOURCES:
        for i, (bs, en) in enumerate(CORP[corpus]):
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
    return cases


CASE_COLUMNS = ["case_id", "source", "bs", "en", "terms", "categories", "accept_en",
                "variant_bs", "variant_swaps"]


def main():
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


if __name__ == "__main__":
    main()
