# Blind truecase photograph verdicts — Codex — 16 September 2026

These judgments were locked against `training/RESULTS-truecase-photos.md` at
commit `35859a0`, before the A/B key was opened. The evaluator read the blind
renderings and inspected the eight source photographs, but did not read or
open `training/truecase-photos-key.json` and was not given the mapping.

The serious-regression rubric is the one fixed in the blind form: a corrupted
place/person/brand name, a new repetition or hallucination, or a correct
English line broken. Where both renderings have a serious defect, the side is
recorded explicitly so unblinding can determine whether the candidate
introduced or inherited it.

| Photograph | Blind verdict | Serious-regression note |
|---|---|---|
| `Assasination_Plaque.JPG` | **B better** | **A serious:** invents “rival” for the subject's wife; B leaves the damaged first OCR line largely untranslated while preserving the plaque's correct English text. |
| `Entrance_to_Bosnia_and_Herzegovina_at_Brod.jpg` | **A better** | **Both serious, B worse:** A drops the place name **Brod** from the otherwise correct English line; B turns **Brod** into “ship” twice and repeats the bilingual message. |
| `GiPS_Bus_Lion_s_City_03.jpg` | **same** | None. Both preserve `GIPS` and the registration; changing the casing of the already damaged side-sign OCR does not improve its meaning. |
| `Plaque_at_the_Battle_of_the_Sutjeska_memorial.jpg` | **B better** | None under the locked rubric. B is materially more readable and keeps **Sutjeska**, although “Deadly Fighters” remains a poor rendering of the damaged OCR. |
| `Street_in_Međugorje.jpg` | **A better** | None. `Giuseppe` and the English shop labels survive; `Minimarket` is more readable than all caps without changing its meaning. |
| `Trg-žrtava-ŠB03078.JPG` | **A better** | **Both serious:** A literalises the place name **Široki Brijeg** as “Wide hill”; B corrupts it to “SILVER BRIDGE”. A is less misleading overall but neither preserves the place name. |
| `Trg_Krajine_čajavčev_i_ulaz_u_gospodsku.jpg` | **A better** | **Both serious, B worse:** A corrupts `MALBAŠIĆ CO` to `MALBAŠI CO`; B drops `CO` and corrupts the displayed `ПРОЈЕКТ`/project brand OCR to `Pnpoject`. |
| `Tuzla_-_INA_petrol_station_2019_.jpg` | **B better** | None. B preserves the displayed `CLASS PLUS` product name; A recases it as ordinary prose. The shared `CAFÉ` OCR loss is not an A/B regression. |

Blind totals only — not a candidate tally: **A better 4/8, B better 3/8,
same 1/8**. Serious defects are present on sides of four photographs; the
candidate gate cannot be evaluated until the committed key is verified and
revealed.
