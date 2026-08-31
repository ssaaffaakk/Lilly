# README assets

Committed images for the GitHub README (keep each file under ~500 KB when possible).

| File | Use |
| :--- | :--- |
| `lilly-hero.jpg` | Hero / brand banner |
| `lilly-modes.jpg` | Type · Speak · Snap |
| `lilly-architecture.jpg` | Illustrated stack |
| `architecture.png` | Exact offline architecture diagram (README; GitHub blocks SVG in many views) |
| `kaggle-flow.png` | Speech half-1/2 + OCR training flow |
| `demo-translate.jpg` | README hero — the running app translating a real sentence |
| `architecture.svg` / `kaggle-flow.svg` | Editable sources — not embedded in README |

`demo-translate.jpg` was captured from the running app at
http://localhost:8000 with a headless Chromium at 1100x800, device scale 2, then
resized to 1600px wide and saved as JPEG. Optional later, the same way:

1. `demo-speak.jpg` — mic + English voice
2. `demo-photo.jpg` — sign photo + read + translate

```bash
git add docs/images/
```
