# README assets

Committed images for the GitHub README (keep each file under ~500 KB when possible).

| File | Use |
| :--- | :--- |
| `lilly-hero.jpg` | README hero / brand banner |
| `lilly-modes.jpg` | README "See it work" — the app's "four ways in" section (type · say · photograph · correct), real screenshot |
| `lilly-architecture.jpg` | Illustrated stack (not embedded in README) |
| `architecture.png` | Exact offline architecture diagram (README; GitHub blocks SVG in many views) |
| `kaggle-flow.png` | Speech half-1/2 + OCR training flow |
| `demo-translate.jpg` | README "See it work" — the running app translating a real sentence |
| `architecture.svg` / `kaggle-flow.svg` | Editable sources for the two `.png` diagrams |

The two diagram PNGs are rendered from their `.svg` sources at 2x device scale
(headless Chromium, `svg` element screenshot). Edit the SVG, re-render the PNG,
and keep both facts current — they carried a stale reader (EasyOCR, since
replaced by PP-OCRv6) and a stale plan link (V3 → V4) until 15 Sep 2026.
`lilly-modes.jpg` and `demo-translate.jpg` are real screenshots of the running
app, recaptured whenever the UI changes.

`demo-translate.jpg` was captured from the running app at
http://localhost:8000 with a headless Chromium at 1100x800, device scale 2, then
resized to 1600px wide and saved as JPEG. Optional later, the same way:

1. `demo-speak.jpg` — mic + English voice
2. `demo-photo.jpg` — sign photo + read + translate

```bash
git add docs/images/
```
