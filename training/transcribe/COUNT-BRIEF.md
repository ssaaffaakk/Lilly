# Counting brief — does a box touch the word? One blind pass over a batch

You are one of two independent counters measuring a text *detector*, not a text
reader. For each photograph you get the original photograph, an overlay of the same
photograph with every region the detector found drawn as a red rectangle (numbered),
and the answer key's list of words that two transcribers agreed are printed somewhere
in the photograph. Your job, for every word in that list: find the word on the
photograph, then look at the overlay and say whether any red rectangle overlaps the
letters of that word.

**Blindness — non-negotiable**
- Do NOT run any OCR or text-recognition software or API on any image. Eyes only.
- Do NOT open any file other than: your batch file, the photographs and overlays it
  lists, and zoom images you make from them. Never open any `.json`, `.tsv`, `.md` or
  `.txt` under `data/ocr/` or `training/`. You have no partner in this task.
- You are never told what the detector read inside a box, and you must not guess it.
  The number printed beside a box is its index, not a reading.

**Tools**
- Open images by the `photo` and `overlay` paths in your batch file (the `Read` tool).
- Small text: `python3 training/transcribe/zoom.py IMAGE --box X0 Y0 X1 Y1 --out DIR`
  enlarges a region (pixel coordinates of that image; `--tiles 3` for nine tiles). It only
  resizes pixels. Run it from the repository root. Zoom the overlay when a box edge and a
  word are close; that is the judgement that matters. Note the overlay may be a smaller
  rendering than the photograph, so coordinates differ between the two.

**Rules of the count (the same for both counters)**
- The key's words are lower-case and in alphabetical order, not reading order; a word
  listed twice is printed twice on the photograph and needs two judgements.
- `covered`: a red rectangle overlaps the glyphs of that word — partial overlap counts,
  a box clipping one letter counts. A box around a *different* printing of the same
  word does not count for this one.
- `missed`: you found the word on the photograph and no rectangle touches its letters.
- `not_located`: you could not find the word on the photograph at all. Use it rather
  than guessing; it is reported separately.
- Every word of `words` goes into exactly one of the three lists, spelled exactly as in
  `words`. The three lists together must be the same multiset as `words`.
- Mark `"opened": true` on every entry you judged. Never leave an entry unopened.

**Output — save as you go**
Rewrite your output file (`counter-X-NN.json` beside the todo file) after EVERY
photograph you finish: the same entries as the todo file, in the same order, with the
three lists filled and `opened` set. Then verify with
`python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sum(1 for e in d if e["opened"]))' OUTPUT_PATH`
and report how many photographs you judged, how many words you put in each bucket, and
which words you could not locate.
