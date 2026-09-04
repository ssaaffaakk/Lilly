#!/usr/bin/env python3
"""Photo scan: image of Bosnian text -> text.

Bosnian ("bs") is supported directly and phone photos are handled well
(neural text detector). Weights are read from models/lilly/read/.

Usage:
    python3 app/ocr.py photo.jpg
"""
import os
import sys
import threading
from pathlib import Path

from app.lilly import READ_DIR, BadInput

# A photo's file size says nothing about what it costs to read: PNG compresses
# flat colour to almost nothing, so a 172 KB file can decode to a gigabyte. The
# limit that matters is pixels, and it has to be checked before decoding.
MAX_DECLARED_PIXELS = 50_000_000   # beyond this, refuse to decode at all
MAX_WORKING_PIXELS = 2_000_000     # above this, shrink first
# Measured on a 12 MP photo of a sign: reading it at 8 MP takes 22 s, at 4 MP
# 18 s, at 2 MP 9 s — and all three misread the same diacritic, so the detail
# was not buying accuracy. Half the wait is worth more than pixels nobody reads.


def _easyocr_gpu() -> bool:
    """CUDA only. The Mac has none, so the app stays on CPU. Kaggle T4 does.

    Pass-7b harvest auto-crop ran EasyOCR on CPU for ~3.4 hours on a T4 box
    because Reader was hardcoded gpu=False. EasyOCR's maintainers say GPU
    mode is gpu=True when CUDA is present (JaidedAI/EasyOCR#426, docs).
    """
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False

# easyocr's Bosnian character list is missing six of the language's own letters:
# c, c and d with their diacritics, plus the capitals, are absent from lang_char
# while s and z are present. That is not a cosmetic gap. easyocr turns the
# language list into an ignore set and subtracts it from the logits before
# decoding, so those six letters cannot come out at all -- not off a cleaner
# photo, not off a better-trained reader. Measured on a plain white image
# reading "ccdsz CCDSZ": the six came back as different letters every time.
# It means "Carsija" could never have read as "Čaršija" however good the photo,
# which is exactly the failure we were about to blame on the reader's training.
# The allowlist puts them back while keeping the rest of the language
# restriction, which is what stops Cyrillic and Vietnamese lookalikes.
BOSNIAN_LETTERS = "čćđšžČĆĐŠŽ"

_reader = None
_allowlist = None
_cyrillic_reader = None
_cyrillic_allowlist = None
_paddle_reader = None
_reader_lock = threading.Lock()
_read_lock = threading.Lock()

# A second engine, for measurement. LILLY_READER=paddle routes every read
# through PaddleOCR instead of EasyOCR -- same door (read_regions), same
# paragraph grouping, same triples out -- so training/evaluate_ocr.py scores it
# on the same 40 photographs by the same code, which is the only comparison
# worth anything (docs/OCR-ROADMAP.md, step 1; the bar is pre-registered in
# training/PREREGISTRATION.md). Untrained: the point is whether a newer
# detector and recogniser, out of the box, already read these photographs
# better than the reader six fine-tuning passes could not improve.
#
# The model pair is written down rather than left to the library's default, so
# the cache stamp and the results file name exactly what read the photographs.
# PP-OCRv6 medium is what paddleocr 3.7 picks for lang="bs"; PP-OCRv5 is the
# Latin model whose dictionary was checked to hold all of čćđšžČĆĐŠŽ. Document
# preprocessing (orientation, unwarping, textline rotation) is off: these are
# photographs of signs, not scans, and the app's EasyOCR path has none of it.
PADDLE_MODELS = {
    "PP-OCRv6": ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),
    "PP-OCRv5": ("PP-OCRv5_server_det", "latin_PP-OCRv5_mobile_rec"),
}


def _paddle_enabled() -> bool:
    return os.environ.get("LILLY_READER", "").lower() == "paddle"


def paddle_rec_floor():
    """The recogniser confidence below which a Paddle region is dropped, or None.

    LILLY_PADDLE_REC_THRESH, unset by default: the library returns every
    recognition whatever its confidence, and that is what the bake-off measured.
    training/PREREGISTRATION.md, "PP-OCRv6 confidence floor", is the one run
    that sets it; the value is part of reader_identity() so a reading cache
    written at one floor is never scored under another.
    """
    raw = os.environ.get("LILLY_PADDLE_REC_THRESH", "").strip()
    if not raw:
        return None
    floor = float(raw)
    if not 0.0 <= floor <= 1.0:
        raise RuntimeError(f"LILLY_PADDLE_REC_THRESH={raw!r}; want a confidence in [0, 1]")
    return floor


def paddle_models() -> tuple:
    """(detector, recogniser) names the paddle path loads, from the environment."""
    version = os.environ.get("LILLY_PADDLE_VERSION", "PP-OCRv6")
    if version not in PADDLE_MODELS:
        raise RuntimeError(f"LILLY_PADDLE_VERSION={version!r}; known: {sorted(PADDLE_MODELS)}")
    det, rec = PADDLE_MODELS[version]
    return (os.environ.get("LILLY_PADDLE_DET", det),
            os.environ.get("LILLY_PADDLE_REC", rec))


def reader_identity() -> str:
    """Which reader a read goes through, as a string a cache can be stamped with.

    evaluate_ocr.py stamps its reading cache with the weight files on disk.
    That stamp does not change when LILLY_READER switches engines or falls back
    to the stock weights, so without this the first stock or paddle score after
    a trained run would read the trained reader's answers back out of the cache
    and print them under the other name -- the exact failure the stamp exists
    to prevent.
    """
    if _paddle_enabled():
        det, rec = paddle_models()
        try:
            import paddleocr
            version = getattr(paddleocr, "__version__", "?")
        except Exception:
            version = "?"
        floor = paddle_rec_floor()
        return f"paddle:{det}+{rec}:{version}" + (f":rec>={floor:g}" if floor is not None else "")
    if _cyrillic_enabled():
        return "easyocr:lilly+cyrillic"
    trained = READ_DIR / "lilly.pth"
    network = READ_DIR / "user_network"
    stock = os.environ.get("LILLY_READER", "").lower() == "stock"
    if not stock and trained.exists() and (network / "lilly.yaml").exists():
        return "easyocr:lilly"
    return "easyocr:stock"


def get_paddle_reader():
    global _paddle_reader
    if _paddle_reader is None:
        with _reader_lock:
            if _paddle_reader is None:
                from paddleocr import PaddleOCR
                det, rec = paddle_models()
                # oneDNN off. PaddleX turns it on by default on x86 CPUs, and
                # with paddlepaddle 3.3.1 the detector's first run then dies in
                # the executor ("ConvertPirAttribute2RuntimeAttribute not
                # support pir::ArrayAttribute<pir::DoubleAttribute>",
                # onednn_instruction.cc) -- both PP-OCRv5 and PP-OCRv6, measured
                # 3 Sep 2026. The plain CPU kernels read the same weights and
                # are slower, so the timing row of the bake-off says which
                # kernels it timed. The Mac has no oneDNN and is unaffected.
                floor = paddle_rec_floor()
                extra = {"text_rec_score_thresh": floor} if floor is not None else {}
                _paddle_reader = PaddleOCR(
                    text_detection_model_name=det,
                    text_recognition_model_name=rec,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_mkldnn=False,
                    **extra)
    return _paddle_reader


def _field(result, key):
    """A key off a PaddleX result, which is a dict that may lack the key."""
    try:
        return result[key]
    except (KeyError, TypeError, IndexError):
        return None


def _paddle_results_to_regions(results) -> list:
    """PaddleOCR's per-image results as EasyOCR's (box, text, confidence) triples.

    A box is four [x, y] corners, the way readtext returns them, so
    easyocr.utils.get_paragraph and measure_detection.py take them unchanged.
    Paddle's polygons can carry more than four points; the axis-aligned bounds
    are what grouping and overlays need, so that is what is kept.
    """
    import numpy as np

    regions = []
    for result in results:
        texts = _field(result, "rec_texts") or []
        scores = _field(result, "rec_scores")
        scores = list(scores) if scores is not None else [1.0] * len(texts)
        polys = _field(result, "rec_polys")
        if polys is None or len(polys) == 0:
            polys = _field(result, "dt_polys") or []
        for poly, text, score in zip(polys, texts, scores):
            text = str(text)
            if not text.strip():
                continue
            pts = np.asarray(poly, dtype=float).reshape(-1, 2)
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            box = [[int(round(x0)), int(round(y0))], [int(round(x1)), int(round(y0))],
                   [int(round(x1)), int(round(y1))], [int(round(x0)), int(round(y1))]]
            regions.append([box, text, float(score)])
    return regions


def _read_regions_paddle(image, detail: int, paragraph: bool) -> list:
    import numpy as np

    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    # The app hands over RGB (PIL); Paddle, like OpenCV, expects BGR.
    bgr = np.ascontiguousarray(array[:, :, :3][:, :, ::-1])
    reader = get_paddle_reader()
    with _read_lock:
        regions = _paddle_results_to_regions(list(reader.predict(bgr)))
    if paragraph:
        from easyocr.utils import get_paragraph
        regions = get_paragraph(regions, x_ths=1.0, y_ths=0.5, mode="ltr")
    if detail == 0:
        return [r[1] for r in regions]
    return regions

# Bosnian is written in both alphabets and the recogniser only knows one.
# latin_g2 has 351 output classes and not one is Cyrillic, so a Cyrillic sign
# could not be read however good the photograph or the training -- measured on
# the scored set, 7.0% of the answer key's words are Cyrillic across 7 of the 40
# photographs, and 16.2% of the hand-transcribed crops. That is a ceiling, and
# no amount of Latin fine-tuning moves it.
#
# So there is a second recogniser. The detector is shared: CRAFT finds where
# text is without caring what alphabet it is in, and it is the expensive half.
# Only recognition runs twice, once per script, over the same boxes, and each
# region keeps whichever reading came back more confident.
#
# CYRILLIC_MARGIN is how much more confident the Cyrillic reading has to be
# before it displaces the Latin one. It is not a taste setting: 0.0 is the
# obvious choice and it is wrong, because the Cyrillic model will happily read
# Latin text as Cyrillic lookalikes -- CTOP for STOP -- with real confidence.
# The value here was measured on the 40 scored photographs, see
# training/RESULTS-ocr-cyrillic.md.
CYRILLIC_MARGIN = float(os.environ.get("LILLY_CYRILLIC_MARGIN", "0.10"))


class ImageTooLarge(BadInput):
    """Raised when an image is too big to be worth decoding."""


class UnreadableImage(BadInput):
    """Raised when the upload is not an image we can open at all."""


def language_characters(languages) -> set:
    """The letters easyocr says these languages are written with."""
    from easyocr.config import BASE_PATH

    found = set()
    for language in languages:
        path = Path(BASE_PATH) / "character" / f"{language}_char.txt"
        if path.exists():
            found |= set(path.read_text(encoding="utf-8-sig").split())
    return found


def get_reader():
    global _reader, _allowlist
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                import easyocr

                # Our fine-tuned reader ships as a network of its own rather
                # than as a replacement for latin_g2.pth. easyocr checks that
                # file's MD5 against the weights it published, so writing our
                # own there makes it refuse to load and report a corrupt
                # download — which is what happened the first time the trained
                # reader was installed: the photo endpoint returned 500 for
                # every image, and the training that had just improved word
                # accuracy from 67.1% to 86.8% had made the feature unusable.
                # LILLY_READER=stock loads easyocr's own Latin weights
                # instead, which is the reader this one was trained from. The
                # project keeps a "before" build of the translator and of the
                # listener and compares against them; the reader had no such
                # thing, so there was no way to ask whether its training helped
                # on anything but the synthetic set it was trained on. This is
                # that build. It is a measurement path, and it is here rather
                # than in the evaluation script because a comparison is only
                # worth anything if both sides go through the same code.
                trained = READ_DIR / "lilly.pth"
                network = READ_DIR / "user_network"
                stock = os.environ.get("LILLY_READER", "").lower() == "stock"
                extra = {} if stock else (
                    {"recog_network": "lilly",
                     "user_network_directory": str(network)}
                    if trained.exists() and (network / "lilly.yaml").exists() else {})
                reader = easyocr.Reader(["bs", "en"], gpu=_easyocr_gpu(),
                                        model_storage_directory=str(READ_DIR),
                                        download_enabled=False, **extra)
                # Stop loudly rather than read Bosnian without its own letters.
                # If the weights are ever swapped for a set that cannot produce
                # these, silence would look like a reader that reads badly.
                cannot = [c for c in BOSNIAN_LETTERS if c not in reader.character]
                if cannot:
                    raise RuntimeError(
                        f"the recogniser cannot produce {''.join(cannot)} — these "
                        f"weights are not the Latin set Lilly reads Bosnian with")
                # Built from the language files rather than from
                # reader.lang_char, because the two are not the same thing once
                # a custom network is registered: easyocr then sets lang_char to
                # the network's whole character set, and the restriction that
                # keeps Cyrillic and Vietnamese lookalikes out of the guesses
                # quietly disappears. Reading the files keeps the restriction
                # identical whichever reader is loaded.
                # Letters are restricted to the two languages; everything that
                # is not a letter is not. The language files hold only letters —
                # easyocr keeps digits, punctuation and space in a separate
                # symbol set — so a list built from them alone silently forbids
                # every number on every sign. Measured: "Radovi 24. jula, 1995.
                # godine!" came back as "RadoviZAjulaIIIDgodinel", with both
                # dates destroyed.
                symbols = {c for c in reader.character if not c.isalpha()}
                _allowlist = "".join(sorted(language_characters(["bs", "en"])
                                            | set(BOSNIAN_LETTERS) | symbols))
                _reader = reader
    return _reader


def load_within_limits(image_path: str):
    """Open a photo at a size worth working on, or refuse it."""
    import numpy as np
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = MAX_DECLARED_PIXELS
    try:
        img = Image.open(image_path)         # reads the header, not the pixels
    except Exception as exc:
        raise UnreadableImage("that file is not an image we can read") from exc
    with img:
        width, height = img.size
        if width * height > MAX_DECLARED_PIXELS:
            raise ImageTooLarge(f"that image is {width}x{height}; too large to read")
        img = img.convert("RGB")
        if width * height > MAX_WORKING_PIXELS:
            scale = (MAX_WORKING_PIXELS / (width * height)) ** 0.5
            img = img.resize((max(int(width * scale), 1), max(int(height * scale), 1)))
        return np.asarray(img)


def get_cyrillic_reader():
    """The second recogniser, for the half of Bosnian signage written in Cyrillic.

    easyocr's rs_cyrillic is Serbian Cyrillic, which is the same alphabet
    Bosnian uses -- all twelve of Ђ Ј Љ Њ Ћ Џ and their lower case are in its
    character file, checked rather than assumed. The weights are cyrillic_g2,
    fetched from easyocr's own distribution like latin_g2 and CRAFT before it.

    No custom network here: nothing has been fine-tuned on Cyrillic. This reads
    it as well as the published model does, which is the whole of the gain --
    going from "cannot represent the alphabet at all" to "reads it ordinarily".
    """
    global _cyrillic_reader, _cyrillic_allowlist
    if _cyrillic_reader is None:
        with _reader_lock:
            if _cyrillic_reader is None:
                import easyocr
                reader = easyocr.Reader(["rs_cyrillic", "en"], gpu=_easyocr_gpu(),
                                        model_storage_directory=str(READ_DIR),
                                        download_enabled=False)
                symbols = {c for c in reader.character if not c.isalpha()}
                _cyrillic_allowlist = "".join(sorted(
                    language_characters(["rs_cyrillic"]) | symbols))
                _cyrillic_reader = reader
    return _cyrillic_reader


def _pick(latin, cyrillic):
    """One region, two readings, one answer.

    Confidence with a margin, not confidence alone. The Cyrillic model reads
    Latin words as Cyrillic lookalikes — С for S, Р for P, О for O — and does it
    confidently, so a bare comparison hands it text it has merely transliterated
    into the wrong alphabet. The margin makes it earn the region.
    """
    if cyrillic is None:
        return latin
    if latin is None:
        return cyrillic
    return cyrillic if cyrillic[2] > latin[2] + CYRILLIC_MARGIN else latin


def read_regions(image, **kwargs):
    """Every text region the reader finds — the only place recognition happens.

    The allowlist is not optional and not the caller's business. Leaving it to
    each call site is how the label-making path in training/prepare_ocr_data.py
    spent its life producing first-draft labels that could not contain c, c or d
    with their diacritics: it built its reader from this module and then called
    readtext itself, so it inherited the check on the weights and none of the
    fix. One door, and forgetting stops being possible.

    Detection runs once and recognition runs twice, once per alphabet, over the
    same boxes. Grouping into paragraphs happens after the two are merged rather
    than inside either, or each reader would group its own guesses and the
    merge would be comparing paragraphs that do not correspond.
    """
    import numpy as np

    detail = kwargs.pop("detail", 1)
    paragraph = kwargs.pop("paragraph", False)
    if _paddle_enabled():
        return _read_regions_paddle(image, detail, paragraph)
    get_reader()

    with _read_lock:
        if not _cyrillic_enabled():
            out = _reader.readtext(image, allowlist=_allowlist,
                                   detail=detail, paragraph=paragraph, **kwargs)
            return out

        img, img_grey = _reader_module_reformat(image)
        horizontal, free = _reader.detect(img, **kwargs)
        horizontal, free = horizontal[0], free[0]

        latin = _reader.recognize(img_grey, horizontal, free,
                                  allowlist=_allowlist, detail=1,
                                  paragraph=False, reformat=False)
        cyrillic = get_cyrillic_reader().recognize(
            img_grey, horizontal, free, allowlist=_cyrillic_allowlist,
            detail=1, paragraph=False, reformat=False)

        if len(latin) != len(cyrillic):
            # Should not happen: same boxes, same order. If it ever does, the
            # merge would silently pair the wrong regions, so take the Latin
            # side whole rather than produce a scrambled reading.
            merged = latin
        else:
            merged = [_pick(a, b) for a, b in zip(latin, cyrillic)]

    if paragraph:
        from easyocr.utils import get_paragraph
        merged = get_paragraph(merged, x_ths=1.0, y_ths=0.5, mode="ltr")
    if detail == 0:
        return [r[1] for r in merged]
    return merged


def _cyrillic_enabled() -> bool:
    """Opt-in, with LILLY_READER=cyrillic. Default off, because it measured worse.

    The second recogniser reads Cyrillic properly -- Сарајево where the Latin
    model produced "Capajebo" -- and it still made the product worse on the 40
    scored photographs at every setting tried. Numbers in
    training/RESULTS-ocr-cyrillic.md. The short version: choosing between the
    two by confidence loses more good Latin readings than it rescues Cyrillic
    ones, and emitting both gains 2.3 points of recall while tripling the words
    invented out of nothing. For an app that translates what it reads, an
    invented word becomes an invented sentence, so that trade is the wrong way
    round.

    The plumbing stays because it is correct and because the route that should
    work needs it: fine-tuning cyrillic_g2 on the 276 Cyrillic crops now
    transcribed, the same move that took the Latin reader from 36.0% to 54.7%.
    An untrained second model arbitrated at read time is not that.
    """
    return os.environ.get("LILLY_READER", "").lower() == "cyrillic"


def _reader_module_reformat(image):
    """easyocr's own colour/grey pair for an already-loaded array."""
    from easyocr.utils import reformat_input
    return reformat_input(image)


def scan(image_path: str) -> str:
    image = load_within_limits(image_path)
    results = read_regions(image, detail=0, paragraph=True)
    return "\n".join(results).strip()


def main() -> int:
    if len(sys.argv) < 2 or not Path(sys.argv[1]).exists():
        print("usage: python3 app/ocr.py <image-file>", file=sys.stderr)
        return 1
    print(scan(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
