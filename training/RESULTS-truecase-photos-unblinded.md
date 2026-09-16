# RESULTS — truecase photograph A/B, unblinded — 16 September 2026

The blind verdicts in `training/truecase-photos-blind-verdicts-codex.md` were
locked at commit `e87c656`, judged against the blind form as it stood at
`35859a0` (where the key's SHA-256 commitment was sealed), before the A/B key was
opened, and unblinded here against `training/truecase-photos-key.json`.

- Key integrity: SHA-256 `d943f7b9c0fc19fea8b28fb98c285fb80d64da4b96ca579e263cab45f6550e10`,
  verified to match the commitment committed at `35859a0`
  (`training/truecase-photos-key.sha256`) before any verdict existed. The key is
  now revealed for the record.
- Reader: shipped PP-OCRv6 at floor 0.9
  (`paddle:PP-OCRv6_medium_det+PP-OCRv6_medium_rec:3.7.0:rec>=0.9`), cv2 4.10.0,
  `scan()`==`scan_regions()[0]` on all 40 (40/40). 30 photographs had text; 8
  render differently between flag off and on.

## Pre-registered gate (locked before the numbers)

Default-on ships only if the candidate is **better in at least 6 of the 8**
changed photographs **and** has **zero serious regressions** — a place / person /
brand name corrupted, a new repetition or hallucination, or a correct English
line broken.

## Unblinded

| Photograph | blind verdict | candidate slot | candidate vs shipped | candidate-side serious regression |
|---|---|---|---|---|
| `Assasination_Plaque.JPG` | B better | A = ON | worse | **yes** — invents "rival" (hallucination) |
| `Entrance_to_Bosnia_and_Herzegovina_at_Brod.jpg` | A better | B = ON | worse | **yes** — "Brod"→"ship" ×2, repetition |
| `GiPS_Bus_Lion_s_City_03.jpg` | same | B = ON | same | no |
| `Plaque_at_the_Battle_of_the_Sutjeska_memorial.jpg` | B better | B = ON | better | no |
| `Street_in_Međugorje.jpg` | A better | A = ON | better | no |
| `Trg-žrtava-ŠB03078.JPG` | A better | A = ON | better | **yes** — "Široki Brijeg"→"Wide hill" (place name) |
| `Trg_Krajine_čajavčev_i_ulaz_u_gospodsku.jpg` | A better | B = ON | worse | **yes** — drops `CO`, brand corrupted |
| `Tuzla_-_INA_petrol_station_2019_.jpg` | B better | A = ON | worse | no |

**Tally.** Candidate **better 3 / 8**, worse 4 / 8, same 1 / 8. Candidate-side
serious regressions: **4**.

## Verdict

The candidate **fails the gate on both arms**: 3 of 8 better against the ≥ 6/8
bar, and 4 serious regressions against the zero bar. One of the three photographs
where the candidate is the better rendering still carries a serious candidate-side
defect (`Trg-žrtava` literalises the place name to "Wide hill"). Recasing helps
the readability of a plain Bosnian sign, but on the real product input it also
lets the translator hallucinate ("rival") and mangle names and repeat bilingual
lines often enough that it is a net loss under a strict, pre-registered bar.

**Decision: `LILLY_TRUECASE` stays OFF; the photograph path is unchanged.** The
re-scoped restorer remains behind the flag as the harness to re-run this exact
bar for any future candidate (for example one that segments a mixed-language line
before recasing, or protects in-language proper nouns) — that would be a new
candidate under its own pre-registration, not this one.
