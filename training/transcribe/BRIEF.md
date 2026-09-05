# Transcription brief — one blind pass over a batch of photographs

This is the brief a transcriber (a person, or a vision agent) is handed together
with one `batch-NN.todo.json` from `make_batches.py`. It is the same for pass a
and pass b of every set; the two passes are made by different transcribers who
never see each other's work. `training/build_truth.py` keeps only the words both
passes wrote, so a pass is only worth something if it is the transcriber's alone.

You are one of two independent transcribers building an answer key for photographs
of signs, plaques and boards taken in Bosnia and Herzegovina (and neighbours). So:

**Blindness — non-negotiable**
- Do NOT run any OCR or text-recognition software or API on any photograph (no easyocr,
  paddleocr, tesseract, no vision API, nothing). You read with your own eyes only.
- Do NOT open any file other than: your batch file, the photographs it lists, and the
  zoom images you make from them. In particular never open any `.json`, `.tsv`, `.md`
  or `.txt` under `data/ocr/` or `training/`. Never search the web for the sign.
- Do not consult anyone else's transcription. You have no partner in this task.

**Tools**
- Open a photograph by the `photo` path in your batch file (the `Read` tool shows it).
  Photographs are about 1280 px wide.
- Small text: enlarge it with the zoom helper (it only resizes pixels — it is not OCR).
  Run it from the repository root with the repository's Python:
  - `python3 training/transcribe/zoom.py PHOTO --tiles 2` → four 2× tiles (`--tiles 3` for nine)
  - `python3 training/transcribe/zoom.py PHOTO --box X0 Y0 X1 Y1` → one region enlarged to
    ~1400 px (pixel coordinates of the photograph; add `--scale 4` for more)
  Then open the paths it prints. Zoom whenever letters are small or diacritics are in
  doubt — a č, ć, đ, š, ž hat or stroke missed at thumbnail size is the commonest error.

**Rules of transcription (the same for both passes)**
- Transcribe what is on the sign, letter for letter, diacritics included (č ć đ š ž, and
  Cyrillic letters when the sign is in Cyrillic). Keep the case as printed.
- One line per line of text on the sign; do not merge different signs into one line.
- Do not correct spelling, expand abbreviations or translate. Write exactly what is printed.
- Transcribe every piece of text in the photograph that you can read: street-name plates,
  shop signs, plaques, memorials, notices, posters, banners, road signs, vehicle lettering,
  house numbers, labels — anything. Include digits and punctuation as printed.
- `legibility`: `"clear"` only when you are confident of every letter in the line.
  `"unclear"` when you can see text but cannot read all of it with confidence — put your
  best partial reading in `text`. Use only these two values.
- A photograph with no readable text gets `"lines": []` — and is still `"opened": true`.
  A photograph you never looked at is a hole, not a blank; never leave one.

**Scratch files: use a directory of your own**
Other transcribers are working in the same pass directory at the same time. If you keep
per-photograph working files or a helper script, put them in a directory named after YOUR
batch (e.g. `entries-batch-07/`), never in a shared name like `entries/` — two transcribers
writing `entries/01.json` overwrite each other's readings. Never read or write any file
belonging to another batch. Send zoom output to your own directory too (`--out`).

**Output — save as you go**
Write a JSON list, one entry per photograph in your batch, in the batch's order, to the
output path you were given (`batch-NN.json` beside the todo file). Rewrite that file after
EVERY photograph you finish (the list so far), so that if you are cut off nothing already
read is lost; `make_batches.py --missing` re-batches the photographs that are not in any
finished file yet. Entry shape:

    {"filename": "Mostar_signs.JPG", "opened": true,
     "scene": "street corner, blue name plate, shop fascia",
     "lines": [{"text": "Ulica Maršala Tita", "legibility": "clear"},
               {"text": "Sara...", "legibility": "unclear"}]}

`scene` is a few words describing what you saw (it proves the photograph was opened;
it is not used for scoring). Every entry must have `"opened": true`, and the filename
must be copied exactly from the batch file. When the file is written, verify it with
`python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d))' OUTPUT_PATH`
and report the count of entries and how many photographs had text.

---

**For the person running the passes** (not part of what the transcriber sees):

    python3 training/transcribe/make_batches.py --set test-v2 --pass a --size 20
    python3 training/transcribe/make_batches.py --set test-v2 --pass b --size 20 --offset 7
    # hand each batch-NN.todo.json to a different transcriber with this brief;
    # pass a and pass b transcribers never overlap
    python3 training/transcribe/make_batches.py --set test-v2 --pass a --missing   # after a cut-off
    python3 training/transcribe/merge.py --set test-v2 --pass a
    python3 training/transcribe/merge.py --set test-v2 --pass b
    python3 training/transcription_pass.py check --set test-v2
    python3 training/transcription_pass.py pair --set test-v2
    python3 training/build_truth.py --result data/ocr/real-photos/test-v2/pair.json \
        --out data/ocr/real-photos/test-v2/truth-v2.json

The test-v2 key (280 photographs, 4–5 September 2026) was built this way with two
sets of vision agents, 14 batches of 20 per pass, pass b offset by 7. Recorded in
`training/RESULTS-ocr-test-v2.md`.
