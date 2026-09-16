# RESULTS — truecase on real photographs (blind A/B) — 16 September 2026

Does the photograph-path truecaser help on real signs? No English reference exists for these signs, so this is a **blind, randomised A/B** for a pass or the owner to read — not a chrF2. Nothing here trains or changes the served build.

- Reader: **shipped** PP-OCRv6 at floor 0.9 (`paddle:PP-OCRv6_medium_det+PP-OCRv6_medium_rec:3.7.0:rec>=0.9`), cv2 `4.10.0`, OCR read fresh this run `2026-09-16T10:56:27.615000+00:00` under the cv2 gate, product-path parity `40/40 identical` (`scan()` == `scan_regions()[0]` on all 40 — both photo paths measured).
- Engine: `bs-en` `app.translate.Engine`, the camera's default direction.
- Each photo shows the OCR text and two renderings, **A** and **B**, in random balanced order. One is today's output, one is the candidate. The mapping is **not given to the evaluator** — only its SHA-256 is committed (`truecase-photos-key.sha256`); the key (`training/truecase-photos-key.json`) is revealed and checked against that hash after the verdicts are locked.

**Pre-registered gate (locked before these numbers existed).** Default-on ships only if, on the changed photographs, the candidate is **better in at least two thirds** of them AND there is **no serious regression**. Serious regression = a place / person / brand name corrupted, a new repetition or hallucination, or a correct English line broken.

8 of 30 photographs with text render differently; label each and note any serious regression.

---

### Assasination_Plaque.JPG

**Sign (OCR, as read)**

```
RANCA FERDINANDA INJEGOVU SUPRUO SOFIJU
FROM THIS PLACE ON 28 JUNE 1914 GAVRILO PRINCIP ASSASSINATED THE HEIR TO THE AUSTRO-HUNGARIAN THRONE FRANZ FERDINAND AND HIS WIFE SOFLA
11.08.201810:46
```

**Rendering A**

```
Ranca ferdinanda and his rival Sofia FROM THIS PLACE ON 28 JUNE 1914 GAVRILO PRINCIPLE ASSASSINATED THE HEIR TO THE AUSTRO-HUNGARIAN THRONE FRANZ FERDINAND AND HIS WIFE SOFLA 11.08.201810:46
```

**Rendering B**

```
RANCA FERDINANDA INJEGOVA SURUVA SOFIA FROM THIS PLACE ON 28 JUNE 1914 GAVRILO PRINCIPLE ASSASSINATED THE HEIR TO THE AUSTRO-HUNGARIAN THRONE FRANZ FERDINAND AND HIS WIFE SOFLA 11.08.201810:46
```

**Verdict: A better / B better / same — _____**

**Serious regression? (name corrupted / new repetition or hallucination / a correct English line broken) — _____**

---

### Entrance_to_Bosnia_and_Herzegovina_at_Brod.jpg

**Sign (OCR, as read)**

```
DOBRO DOŠLI U BOSNU I HERCEGOVINU BROD WELCOME TO BOSNIA AND HERZEGOVINA BROD
```

**Rendering A**

```
WELCOME TO BOSNIA AND HERZEGOVINA
```

**Rendering B**

```
Welcome to Bosnia and Herzegovina ship welcome to Bosnia and Herzegovina ship
```

**Verdict: A better / B better / same — _____**

**Serious regression? (name corrupted / new repetition or hallucination / a correct English line broken) — _____**

---

### GiPS_Bus_Lion_s_City_03.jpg

**Sign (OCR, as read)**

```
GIPS
ENTAR ODOVA LASER-
GIPS
T80-K-611
```

**Rendering A**

```
GIPS ENTAR ODOVA LASER- GIPS T80-K-611
```

**Rendering B**

```
GIPS Entar odova laser GIPS T80-K-611
```

**Verdict: A better / B better / same — _____**

**Serious regression? (name corrupted / new repetition or hallucination / a correct English line broken) — _____**

---

### Plaque_at_the_Battle_of_the_Sutjeska_memorial.jpg

**Sign (OCR, as read)**

```
OVDJE POCIVA 33O1 BORAC SA
POGINULIM BORCIMA S
SUTJESKE
```

**Rendering A**

```
33O1 BORAC WITH HERE MURDERED FIGHTERS SUTJESKA
```

**Rendering B**

```
Here rests a 33o1 fighter with Deadly Fighters Sutjeska
```

**Verdict: A better / B better / same — _____**

**Serious regression? (name corrupted / new repetition or hallucination / a correct English line broken) — _____**

---

### Street_in_Međugorje.jpg

**Sign (OCR, as read)**

```
SOUVENIRS SHOP Giuseppe
MINIMARKET
SHOP
SHO
```

**Rendering A**

```
SOUVENIRS SHOP Giuseppe Minimarket SHOP SHO
```

**Rendering B**

```
SOUVENIRS SHOP Giuseppe MINIMARKET SHOP SHO
```

**Verdict: A better / B better / same — _____**

**Serious regression? (name corrupted / new repetition or hallucination / a correct English line broken) — _____**

---

### Trg-žrtava-ŠB03078.JPG

**Sign (OCR, as read)**

```
ŠIROKI BRIJEG
UVUEKNAT.MJESTU
BRIJESK ZVON
```

**Rendering A**

```
Wide hill Uvueknat. Brijesek bell
```

**Rendering B**

```
SILVER BRIDGE UVEKNAT.PLACE BRIJESK ZVON
```

**Verdict: A better / B better / same — _____**

**Serious regression? (name corrupted / new repetition or hallucination / a correct English line broken) — _____**

---

### Trg_Krajine_čajavčev_i_ulaz_u_gospodsku.jpg

**Sign (OCR, as read)**

```
MALBAŠIĆ CO
SIEMENS
LANACO
PNPOJEKT
VINKOP
```

**Rendering A**

```
MALBAŠI CO SIEMENS LANACO PNPOJEKT VINKOP
```

**Rendering B**

```
Malbašić SIEMENS LANACO Pnpoject VINKOP
```

**Verdict: A better / B better / same — _____**

**Serious regression? (name corrupted / new repetition or hallucination / a correct English line broken) — _____**

---

### Tuzla_-_INA_petrol_station_2019_.jpg

**Sign (OCR, as read)**

```
NA CLASS 2.26 2.3 2.2
0-24 SHOP CAFÉ WASH
CLASS PLUS GORIVO ZA SVE TEMPERATURE
```

**Rendering A**

```
NA CLASS 2.26 2.3 2.2 0-24 SHOP CAF WASH Class plus fuel for all temperatures
```

**Rendering B**

```
NA CLASS 2.26 2.3 2.2 0-24 SHOP CAF WASH CLASS PLUS FUEL FOR ALL TEMPERATURES
```

**Verdict: A better / B better / same — _____**

**Serious regression? (name corrupted / new repetition or hallucination / a correct English line broken) — _____**

---

## After the verdicts are locked

Fill every A/B and regression line above and commit them. Then reveal `training/truecase-photos-key.json`, check its SHA-256 against `truecase-photos-key.sha256`, unblind, and tally against the gate. This file is the evidence; the default-on decision is the tally.

