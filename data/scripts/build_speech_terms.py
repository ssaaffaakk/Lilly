#!/usr/bin/env python3
"""Mine the word pairs that separate Bosnian from Croatian and from Serbian.

training/speech_bench.py scores whether the listener writes Bosnian or writes a
neighbour's variety. It can only see a drift it has a word pair for, and the
pairs it had came from bench/terms.tsv, which the translation bench built for a
different purpose. The result was measured and it is the reason this file
exists: **73 of its 85 targets were yat pairs whose alternative form is
Serbian** — mjesto/mesto, vrijeme/vreme — and the training run those 85 targets
were asked to judge is the run that added three and a half thousand clips of
*Croatian*. The instrument passed the model. It had almost nothing for that
particular failure to land on.

Croatian is ijekavian like Bosnian. Every yat pair in the world is blind to it.
What separates Bosnian from Croatian is somewhere else: Bosnian keeps the
internationalism where Croatian coins a native replacement (hiljada/tisuca,
avion/zrakoplov, univerzitet/sveuciliste), and Croatian has its own month names outright. Those
pairs are what this file goes looking for.

## What it will not do

It will not take a word list somebody typed. The pairs are mined from
professional translation into all three varieties, and a pair is admitted only
on evidence measured here:

  1. RATE, not count. Bosnian form appears at least MIN_RATE per million tokens
     in professional Bosnian text, and the counterpart at least that in
     professional Croatian (or Serbian) text. Rates rather than raw counts
     because the three corpora are not the same size, and a raw-count gate
     silently admits whatever the biggest corpus happens to contain.

  2. EXCLUSIVITY, both ways. Of the two rates, the Bosnian form's share must be
     at least MIN_SHARE in Bosnian text, and the counterpart's share at least
     MIN_SHARE in the neighbour's text. A pair where both varieties use both
     words is not a contrast, it is a synonym, and scoring it would count a
     coin-flip as drift.

  3. ALIGNMENT. The two words must actually be each other's counterpart, and
     that is read off sentence-aligned text: when the Bosnian rendering of a
     sentence uses the Bosnian form and the Croatian rendering of the *same*
     sentence does not, the Croatian form has to be the word that turns up in
     its place, in at least MIN_ASSOC of those sentences.

Gate 3 is the one that cannot be replaced by a dictionary, and it is why the
aligned corpora are worth the download. It is also the gate that fails loudly:
bs and hr renderings of a FLORES sentence are independent translations and
often share barely a content word, so a wrong pairing does not reach 0.5.

## The corpora, and why alignment survives paraphrase

  FLORES-200 dev + devtest   997 + 1,012 sentences, bos/hrv/srp, professional
  NTREX-128                  1,997 sentences, the same three
  SETIMES v2                 news, aligned to English on each side; the three
                             varieties are joined here on the *English* string,
                             which is the same sentence by construction

Serbian arrives in Cyrillic in two of the three. It is transliterated, because
the drift that matters is a Latin-script listener writing *mesto*: a Cyrillic
corpus cannot be compared with a Latin one token by token, and refusing to
transliterate would silently drop Serbian from the evidence entirely.

## Held out from what it grades

Every sentence that is also a transcript in data/speech/test.tsv is removed
from the evidence — from all three sides of the triple, so the alignment stays
honest. FLEURS transcripts are FLORES sentences, so without this a pair could be
admitted on the strength of the very sentences it is about to be scored on.
Reported, not assumed: the run prints how many triples that removed.

    python3 data/scripts/build_speech_terms.py
    python3 data/scripts/build_speech_terms.py --min-share 0.95   # looser, for a look
    python3 data/scripts/build_speech_terms.py --fetch-only

The text is fetched on first run into data/speech-extra/text/, which is ignored
by git — about 150 MB, too big to track, and the fetch is here rather than in a
shell history so the term file can be rebuilt from nothing. bench/speech/
contrast-terms.tsv, the thing this produces, IS tracked: an instrument whose
inputs are not in the repository is one disk failure from being unauditable,
which is a mistake this project has already made once with truth.json.

Writes bench/speech/contrast-terms.tsv. Loads no model and needs no GPU.
"""
import argparse
import collections
import csv
import re
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TEXT = REPO / "data" / "speech-extra" / "text"
OUT = REPO / "bench" / "speech" / "contrast-terms.tsv"
TEST_TSV = REPO / "data" / "speech" / "test.tsv"

WORD = re.compile(r"[a-zA-ZčćžšđČĆŽŠĐ]+", re.UNICODE)

# Admission thresholds. Deliberately in one place and deliberately not tuned
# after looking at how the listener scores: a threshold moved to keep a term
# that flatters a model is the exam being written to the answers.
MIN_RATE = 8.0      # per million tokens, in the variety that owns the form
MIN_SHARE = 0.97    # of the two rates, the owning variety's share
MIN_ASSOC = 0.40    # of aligned sentences, how often the counterpart turns up
MIN_DICE = 0.30     # and how specific that is, counted from both sides
MIN_ALIGNED = 6     # aligned sentences the association is computed over

# Gaj's Latin, the standard one-to-one mapping. Serbian is written in both
# scripts and the two are officially equivalent; this is a change of alphabet,
# not a translation.
CYR = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "đ", "е": "e",
    "ж": "ž", "з": "z", "и": "i", "ј": "j", "к": "k", "л": "l", "љ": "lj",
    "м": "m", "н": "n", "њ": "nj", "о": "o", "п": "p", "р": "r", "с": "s",
    "т": "t", "ћ": "ć", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "č",
    "џ": "dž", "ш": "š",
}


def to_latin(text: str) -> str:
    out = []
    for ch in text:
        low = ch.lower()
        if low in CYR:
            mapped = CYR[low]
            out.append(mapped.upper() if ch.isupper() else mapped)
        else:
            out.append(ch)
    return "".join(out)


def is_cyrillic(text: str) -> bool:
    letters = [c for c in text[:200_000] if c.isalpha()]
    if not letters:
        return False
    cyr = sum(1 for c in letters if "CYRILLIC" in unicodedata.name(c, ""))
    return cyr > len(letters) / 2


def tokens(sentence: str) -> list:
    return [w.lower() for w in WORD.findall(sentence)]


# ---------------------------------------------------------------- fetching

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
NTREX_URL = ("https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/"
             "NTREX-128/{name}")
# OPUS names a pair alphabetically, which is why two of these read backwards.
SETIMES_URL = "https://object.pouta.csc.fi/OPUS-SETIMES/v2/moses/{pair}.txt.zip"
SETIMES_PAIRS = {"bs": "bs-en", "hr": "en-hr", "sr": "en-sr"}
# A second, unrelated Bosnian corpus. Its whole job is to veto — see
# independent_bosnian() below for the defect that put it here.
WIKIMEDIA_URL = ("https://object.pouta.csc.fi/OPUS-wikimedia/v20260327/moses/"
                 "bs-en.txt.zip")


def get(url: str) -> bytes:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "lilly-speech-terms"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def fetch() -> None:
    """Put the three aligned corpora on disk, skipping whatever is already there."""
    import io
    import tarfile
    import zipfile

    TEXT.mkdir(parents=True, exist_ok=True)

    want = {f"flores.{split}.{lang}": f"{split}/{code}.{split}"
            for split in ("dev", "devtest")
            for lang, code in (("bs", "bos_Latn"), ("hr", "hrv_Latn"),
                               ("sr-cyrl", "srp_Cyrl"))}
    if any(not (TEXT / name).exists() for name in want):
        print(f"fetching {FLORES_URL}", flush=True)
        blob = get(FLORES_URL)
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            members = {Path(m.name).parent.name + "/" + Path(m.name).name: m
                       for m in tar.getmembers() if m.isfile()}
            for name, inner in want.items():
                member = members.get(inner)
                if member is None:
                    raise SystemExit(f"{inner} not in the FLORES archive")
                (TEXT / name).write_bytes(tar.extractfile(member).read())
        print(f"  wrote {len(want)} FLORES files")

    for lang, code in (("bs", "bos"), ("hr", "hrv"), ("sr", "srp-Latn")):
        out = TEXT / f"ntrex.{lang}"
        if not out.exists():
            name = f"newstest2019-ref.{code}.txt"
            print(f"fetching NTREX {name}", flush=True)
            out.write_bytes(get(NTREX_URL.format(name=name)))

    wiki = TEXT / "wikimedia.bs"
    if not wiki.exists():
        print("fetching OPUS wikimedia bs", flush=True)
        with zipfile.ZipFile(io.BytesIO(get(WIKIMEDIA_URL))) as z:
            name = next(n for n in z.namelist() if n.endswith(".bs"))
            wiki.write_bytes(z.read(name))

    for lang, pair in SETIMES_PAIRS.items():
        out, out_en = TEXT / f"setimes.{lang}", TEXT / f"setimes.{lang}.en"
        if out.exists() and out_en.exists():
            continue
        print(f"fetching SETIMES {pair}", flush=True)
        with zipfile.ZipFile(io.BytesIO(get(SETIMES_URL.format(pair=pair)))) as z:
            names = [n for n in z.namelist() if n.endswith((f".{lang}", ".en"))]
            for name in names:
                target = out_en if name.endswith(".en") else out
                target.write_bytes(z.read(name))
        if not (out.exists() and out_en.exists()):
            raise SystemExit(f"SETIMES {pair} did not contain both sides")


# ---------------------------------------------------------------- corpora


def lines(path: Path) -> list:
    text = path.read_text(encoding="utf-8", errors="replace")
    if is_cyrillic(text):
        text = to_latin(text)
    return [l.strip() for l in text.splitlines()]


def aligned_triples(exclude: set) -> tuple:
    """(bs, hr, sr) sentences that are renderings of the same sentence.

    Returns the triples and a per-source tally, because a source that silently
    contributes nothing is the kind of thing that stays unnoticed for a week.
    """
    triples, tally, dropped = [], {}, 0

    def add(name, bs_list, hr_list, sr_list):
        nonlocal dropped
        if not (len(bs_list) == len(hr_list) == len(sr_list)):
            raise SystemExit(f"{name}: sides disagree — "
                             f"{len(bs_list)}/{len(hr_list)}/{len(sr_list)}")
        kept = 0
        for b, h, s in zip(bs_list, hr_list, sr_list):
            if not (b and h and s):
                continue
            if b in exclude:
                dropped += 1
                continue
            triples.append((b, h, s))
            kept += 1
        tally[name] = kept

    for split in ("dev", "devtest"):
        bs, hr = TEXT / f"flores.{split}.bs", TEXT / f"flores.{split}.hr"
        sr = TEXT / f"flores.{split}.sr-cyrl"
        if bs.exists() and hr.exists() and sr.exists():
            add(f"flores-{split}", lines(bs), lines(hr), lines(sr))

    n_bs, n_hr = TEXT / "ntrex.bs", TEXT / "ntrex.hr"
    n_sr = TEXT / "ntrex.sr" if (TEXT / "ntrex.sr").exists() else TEXT / "ntrex.sr-cyrl"
    if n_bs.exists() and n_hr.exists() and n_sr.exists():
        add("ntrex", lines(n_bs), lines(n_hr), lines(n_sr))

    # SETIMES: three two-column corpora joined on the English side. Only English
    # sentences that are unique within each file are used — a repeated English
    # line has no single Bosnian counterpart, and guessing one would fabricate
    # an alignment rather than find it.
    have = all((TEXT / f"setimes.{lang}{suffix}").exists()
               for lang in ("bs", "hr", "sr") for suffix in ("", ".en"))
    if have:
        side = {}
        for lang in ("bs", "hr", "sr"):
            text_lines = lines(TEXT / f"setimes.{lang}")
            en_lines = lines(TEXT / f"setimes.{lang}.en")
            if len(text_lines) != len(en_lines):
                raise SystemExit(f"setimes.{lang}: {len(text_lines)} lines against "
                                 f"{len(en_lines)} English")
            counts = collections.Counter(en_lines)
            side[lang] = {en: t for en, t in zip(en_lines, text_lines)
                          if counts[en] == 1 and en and t}
        shared = set(side["bs"]) & set(side["hr"]) & set(side["sr"])
        bs_l = [side["bs"][en] for en in sorted(shared)]
        hr_l = [side["hr"][en] for en in sorted(shared)]
        sr_l = [side["sr"][en] for en in sorted(shared)]
        add("setimes", bs_l, hr_l, sr_l)

    if not triples:
        raise SystemExit(f"no aligned text under {TEXT} — see this file's docstring")
    return triples, tally, dropped


# ---------------------------------------------------------------- mining


def independent_bosnian() -> list:
    """Bosnian text from somewhere other than SETIMES, used only to refuse pairs.

    Found by one of the bench's own controls rather than reasoned about in
    advance. The Croatian and Serbian marker words were counted in the graded
    Bosnian transcripts, where they should be near zero, and among the handful
    that turned up were *kancelarija* and *kancelarije* — filed here as the
    Serbian counterparts of *ured*. Both are ordinary Bosnian. What had happened
    is that SETIMES is one publisher's house style: its Bosnian desk writes
    *ured*, so *kancelarija* looked exclusively Serbian to a gate that had only
    SETIMES to look at, and 97% of the aligned Bosnian text is SETIMES.

    Where the Bosnian standard genuinely allows both forms the corpus picks one,
    and the miner then calls the other a foreign word. So every rate on the
    Bosnian side is taken as the LARGER of what the aligned corpus says and what
    this one says. On the Bosnian form that is a loosening and on the
    alternative it is a tightening, and both are the same rule read in the two
    directions: a word is Bosnian if any professional Bosnian corpus uses it,
    and belongs to the neighbour only if none does.

    Wikipedia content translations, a different register and a different set of
    editors from a news wire. Returns an empty list if it is not on disk, and
    the run says so rather than quietly dropping the gate.
    """
    path = TEXT / "wikimedia.bs"
    if not path.exists():
        return []
    return [l.strip() for l in
            path.read_text(encoding="utf-8", errors="replace").splitlines()
            if l.strip()]


def document_frequency(sentences: list) -> tuple:
    """Sentences containing each token, and the total token count."""
    df, total = collections.Counter(), 0
    for s in sentences:
        toks = tokens(s)
        total += len(toks)
        for t in set(toks):
            df[t] += 1
    return df, total


def rates(df: collections.Counter, total: int) -> dict:
    scale = 1_000_000 / max(total, 1)
    return {w: n * scale for w, n in df.items()}


def counterparts(triples: list, side_a: int, side_b: int) -> dict:
    """For each word of side A absent from side B, what turns up instead.

    Counted only over sentences where A has the word and B does not, which is
    exactly the situation the substitution metric is about.
    """
    seen = collections.Counter()
    pairs = collections.defaultdict(collections.Counter)
    for triple in triples:
        a_toks, b_toks = set(tokens(triple[side_a])), set(tokens(triple[side_b]))
        only_a, only_b = a_toks - b_toks, b_toks - a_toks
        if not only_b:
            continue
        for w in only_a:
            seen[w] += 1
            for x in only_b:
                pairs[w][x] += 1
    return seen, pairs


def char_edits(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def best_counterparts(triples: list, from_index: int, to_index: int,
                      rate_from: dict, rate_to: dict, args,
                      seen_back: dict = None, pre=None) -> dict:
    """For each word of the `from` variety, the `to` variety's word for it.

    Both words must clear the rate and exclusivity gates in their own variety,
    the counterpart must turn up in at least `min_assoc` of the sentences where
    the first word appears and it does not, AND that co-occurrence must be
    specific in both directions.

    That last clause is Dice, and it is not decoration. Without it the miner
    filed Serbian *vestima* against Bosnian *takoder*: SETIMES runs a recurring
    news-digest formula, so of the 650 sentences whose Serbian side says
    *vestima*, 615 have *takoder* somewhere in the Bosnian side — 94.6%, which
    sails past any one-directional threshold. It is not a counterpart, it is a
    common word inside a repeated sentence shape. *takoder* appears in about
    4,180 sentences without a Serbian counterpart, so Dice reads
    2x615/(650+4180) = 0.25 and the pair is refused, while *novosti*, the word
    that really answers *vestima*, reads 0.71 and is kept.
    """
    seen, pairs = pre if pre is not None else counterparts(triples, from_index,
                                                          to_index)
    seen_back = seen_back or {}
    admitted = {}
    for word, n_seen in seen.items():
        if n_seen < args.min_aligned:
            continue
        r_from, r_to = rate_from.get(word, 0.0), rate_to.get(word, 0.0)
        if r_from < args.min_rate:
            continue
        if r_from / (r_from + r_to) < args.min_share:        # its own variety
            continue
        best = None
        for cand, n in pairs[word].most_common(40):
            assoc = n / n_seen
            if assoc < args.min_assoc:
                break                                        # most_common is sorted
            c_from, c_to = rate_from.get(cand, 0.0), rate_to.get(cand, 0.0)
            if c_to < args.min_rate:
                continue
            if c_to / (c_from + c_to) < args.min_share:      # the other variety
                continue
            dice = 2 * n / (n_seen + seen_back.get(cand, n))
            if dice < args.min_dice:
                continue
            score = (dice, assoc, -char_edits(word, cand))
            if best is None or score > best[0]:
                best = (score, cand, n, assoc, dice)
        if best is not None:
            _, cand, n, assoc, dice = best
            admitted[word] = {"alt": cand, "n_aligned": n_seen, "n_pair": n,
                              "assoc": assoc, "dice": dice}
    return admitted


def same_lemma(a: str, b: str) -> bool:
    """Two forms of one word, near enough for the mutual check below.

    Deliberately mean about short words. A four-character prefix rule would call
    *mjesto* and *mjesec* the same lemma, and the mutual check exists to catch
    exactly that kind of near-miss, so the second clause asks for five shared
    characters. The third clause is for the words too short to have five —
    jul/juli, not jul/junak — where one form must be a prefix of the other and
    at most one character shorter.
    """
    if a == b:
        return True
    if len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5] and abs(len(a) - len(b)) <= 3:
        return True
    short, long = sorted((a, b), key=len)
    return len(short) >= 3 and long.startswith(short) and len(long) - len(short) <= 1


def mine(triples: list, rate_bs: dict, rate_other: dict, other_index: int,
         args) -> tuple:
    """Bosnian form -> the neighbour's form, mined in both directions.

    The forward pass alone produces a specific and repeatable mistake. Bosnian
    writes *generalnog sekretara* where Croatian writes *glavnog tajnika*: two
    words change together, so *generalnog* co-occurs with *tajnika* just as
    often as *sekretara* does, and the forward pass is free to pair the wrong
    halves. The pair still detects drift — a listener writing *tajnika* has
    drifted whichever Bosnian word it is filed under — but it misattributes
    which word drifted, and a term list nobody can read is a term list nobody
    can check.

    So the neighbour's words are mined back to Bosnian under the same gates, and
    a forward pair is thrown out when the two directions name different words.

    Only when they *disagree*. A silent reverse direction is not evidence
    against the pair, and treating it as such cost real terms the first time
    this was written: Croatian *tvrtka* answers Bosnian firma, kompanija and
    preduzece all three, so no single Bosnian word holds enough of it to be its
    best counterpart, and firma/tvrtka — as clean a contrast as this list has —
    was refused. Many-to-one is a property of vocabulary, not a fault in a pair.

    The reverse direction's own pairs are then added, which is what fixes the
    inflection gap: the forward pass matched Bosnian *duzini* to Croatian
    *duljine* because that is the case the aligned sentences happened to use,
    and the reverse pass supplies *duljina* and *duljini* beside it.

    Rejections are returned rather than dropped, because "what did the gate
    throw out" is the question that says whether the gate is sane.
    """
    fwd_pre = counterparts(triples, 0, other_index)
    rev_pre = counterparts(triples, other_index, 0)
    forward = best_counterparts(triples, 0, other_index, rate_bs, rate_other, args,
                                seen_back=rev_pre[0], pre=fwd_pre)
    reverse = best_counterparts(triples, other_index, 0, rate_other, rate_bs, args,
                                seen_back=fwd_pre[0], pre=rev_pre)
    admitted, rejected = {}, {}
    for word, info in forward.items():
        back = reverse.get(info["alt"])
        if back is not None and not same_lemma(back["alt"], word):
            rejected[word] = (info["alt"], f"reverse says {back['alt']}")
        else:
            admitted[word] = {**info, "direction": "bs->alt"}
    for alt, info in reverse.items():
        word = info["alt"]
        if word in admitted or word in rejected:
            continue
        admitted[word] = {"alt": alt, "n_aligned": info["n_aligned"],
                          "n_pair": info["n_pair"], "assoc": info["assoc"],
                          "dice": info["dice"], "direction": "alt->bs"}
    return admitted, rejected


# ---------------------------------------------------------------- families


def families(words: list, hr: dict, sr: dict) -> dict:
    """Group inflections of one word so a sentence yields one decision, not three.

    The reason to group at all is statistical. Two inflections of one lemma in
    one sentence are not two independent chances to drift, and counting them as
    two would narrow every interval the bench prints. The alternative forms are
    pooled over the family too, so a listener that drifts to *duljini* when the
    mined pair recorded *duljine* is still scored as drift.

    That pooling is why the grouping cannot be a prefix rule on the Bosnian form
    alone, which is what this was first. Five shared characters put *prijedlog*
    and *prijevoznik* in one family, and *svijet* with *svijest* — so the family
    for *prijedlog* carried *avioprevoznik* among its Serbian forms, and a
    listener writing that word anywhere in the sentence would have been recorded
    as drift on a word it never touched.

    So two forms join only when their counterparts agree as well: a shared
    five-character prefix on the Bosnian side, and a shared four-character
    prefix on the alternative side in every variety where both forms have one.
    Where they have no variety in common there is no evidence they belong
    together, and they stay apart. That still merges *svjetlo* with *svjetski*,
    whose Serbian forms both begin *svet* — harmless, because every alternative
    in that cluster is an ekavian spelling and writing any of them is the drift
    the target is there to catch.

    Crude on purpose: the alternative is a lemmatiser this project does not
    have, and every grouping it makes is written into the output file where it
    can be read and disagreed with.
    """
    def agrees(a: str, b: str) -> bool:
        shared = False
        for table in (hr, sr):
            x, y = table.get(a), table.get(b)
            if not (x and y):
                continue
            shared = True
            if x["alt"][:4] != y["alt"][:4]:
                return False
        return shared

    out, roots = {}, []
    for w in sorted(words, key=lambda w: (len(w), w)):
        placed = False
        for root in roots:
            if (len(root) >= 5 and w.startswith(root[:5])
                    and len(w) - len(root) <= 3 and agrees(w, root)):
                out[w] = root
                placed = True
                break
        if not placed:
            roots.append(w)
            out[w] = w
    return out


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-rate", type=float, default=MIN_RATE)
    ap.add_argument("--min-share", type=float, default=MIN_SHARE)
    ap.add_argument("--min-assoc", type=float, default=MIN_ASSOC)
    ap.add_argument("--min-dice", type=float, default=MIN_DICE)
    ap.add_argument("--min-aligned", type=int, default=MIN_ALIGNED)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--fetch-only", action="store_true",
                    help="download the corpora and stop")
    args = ap.parse_args()

    fetch()
    if args.fetch_only:
        return 0

    from training.train_speech import read_tsv
    graded = {t for _, t in read_tsv(TEST_TSV)}
    print(f"graded transcripts held out of the evidence: {len(graded)} distinct")

    triples, tally, dropped = aligned_triples(graded)
    print(f"aligned bs/hr/sr triples: {len(triples)}")
    for name, n in tally.items():
        print(f"  {name:<16}{n:>8}")
    print(f"  {'held out':<16}{dropped:>8}  (also a graded transcript)")

    df_bs, tot_bs = document_frequency([t[0] for t in triples])
    df_hr, tot_hr = document_frequency([t[1] for t in triples])
    df_sr, tot_sr = document_frequency([t[2] for t in triples])
    rate_bs, rate_hr, rate_sr = (rates(df_bs, tot_bs), rates(df_hr, tot_hr),
                                 rates(df_sr, tot_sr))
    print(f"tokens: bs {tot_bs:,}  hr {tot_hr:,}  sr {tot_sr:,}")

    wiki = [s for s in independent_bosnian() if s not in graded]
    if wiki:
        df_w, tot_w = document_frequency(wiki)
        rate_w = rates(df_w, tot_w)
        raised = sum(1 for w, r in rate_w.items() if r > rate_bs.get(w, 0.0))
        for word, r in rate_w.items():
            if r > rate_bs.get(word, 0.0):
                rate_bs[word] = r
        print(f"independent Bosnian: {len(wiki):,} wikimedia sentences, "
              f"{tot_w:,} tokens; raised the Bosnian rate of {raised:,} forms")
    else:
        print("independent Bosnian: MISSING — the veto gate is not running")

    hr, hr_out = mine(triples, rate_bs, rate_hr, 1, args)
    sr, sr_out = mine(triples, rate_bs, rate_sr, 2, args)
    print(f"\nmined pairs: {len(hr)} bosnian-vs-croatian, {len(sr)} bosnian-vs-serbian")
    print(f"refused by the mutual check: {len(hr_out)} hr, {len(sr_out)} sr")
    for word, (alt, why) in sorted(hr_out.items())[:12]:
        print(f"  hr  {word:<18} -> {alt:<18} {why}")

    words = sorted(set(hr) | set(sr))
    fam = families(words, hr, sr)
    print(f"{len(words)} Bosnian forms in {len(set(fam.values()))} families")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["family", "bosnian_form", "hr_form", "sr_form",
                    "rate_bs", "rate_hr", "rate_sr",
                    "share_vs_hr", "share_vs_sr", "assoc_hr", "assoc_sr",
                    "n_aligned_hr", "n_aligned_sr", "mined_hr", "mined_sr"])
        for word in words:
            h, s = hr.get(word), sr.get(word)
            r_bs = rate_bs.get(word, 0.0)
            r_hr, r_sr = rate_hr.get(word, 0.0), rate_sr.get(word, 0.0)
            w.writerow([
                fam[word], word,
                h["alt"] if h else "", s["alt"] if s else "",
                f"{r_bs:.2f}", f"{r_hr:.2f}", f"{r_sr:.2f}",
                f"{r_bs / (r_bs + r_hr):.4f}" if r_bs + r_hr else "",
                f"{r_bs / (r_bs + r_sr):.4f}" if r_bs + r_sr else "",
                f"{h['assoc']:.3f}" if h else "", f"{s['assoc']:.3f}" if s else "",
                h["n_aligned"] if h else 0, s["n_aligned"] if s else 0,
                h["direction"] if h else "", s["direction"] if s else "",
            ])
    print(f"wrote {args.out}")

    both = sorted(set(hr) & set(sr))
    print(f"\nforms with BOTH a Croatian and a Serbian counterpart: {len(both)}")
    for word in both[:15]:
        print(f"  {word:<20} hr {hr[word]['alt']:<20} sr {sr[word]['alt']}")
    print("\nCroatian-only contrasts, the ones this bench was missing:")
    for word in sorted(set(hr) - set(sr), key=lambda w: -hr[w]["n_aligned"])[:25]:
        h = hr[word]
        print(f"  {word:<20} -> {h['alt']:<20} assoc {h['assoc']:.2f} "
              f"on {h['n_aligned']} sentences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
