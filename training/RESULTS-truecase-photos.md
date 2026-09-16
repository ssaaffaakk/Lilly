# RESULTS — truecase on real photographs — 16 September 2026

The photograph bar `training/RESULTS-truecase.md` left open: does the truecaser help on real signs, not uppercased FLORES? No English reference exists for these signs, so this is an A/B for a blind pass or the owner to read, not a chrF2. Nothing here trains or changes the served build.

Re-measured 16 Sep after the restorer was re-scoped: it now lives on the photograph path (`Lilly.translate_photo`), not inside `translate()`, and runs line by line — a line is recased only when it is shouted **and** `app.detect` reads it as Bosnian, so an English caption or a brand on the same sign is left alone. The earlier version recased the whole blob through `translate()` and touched typed text too.

- Reader: PP-OCRv6 cached output, fingerprint `afa719f3435ca2fe` (`data/ocr/real-photos/reader-output-paddle-v6.json`), the reader `app/ocr.py` serves.
- Engine: `bs-en` `app.translate.Engine`, the camera's default direction (`Lilly.translate_photo`).
- Method: OFF = whole OCR text through `translate()` (shipped today). ON = `restore_photo_text` line by line, then `translate()` — the flag up on the photograph path.

## What the flag did

| | count | of 40 |
|---|---|---|
| photographs with OCR text | 33 | 82% |
| source the flag recased | 9 | 22% |
| had a shouted English line kept back | 7 | 17% |
| **translation the flag changed** | **9** | **22%** |

The recaser fires only on predominantly-uppercase lines (≥8 letters, ≥80% upper) that `app.detect` reads as Bosnian. The 9 below are where a user would see a different answer; label each **better / same / worse** against OFF.

---

### Assasination_Plaque.JPG

**Sign (OCR, as read)**

```
JSTROUGARSINOGTNL RANCA FERDINANDA INJEGOVU SUPRUO SOFIJU
FROM THIS PLACE ON 28 JUNE 1914 GAVRILO PRINCIP ASSASSINATED THE HEIR TO THE AUSTRO-HUNGARIAN THRONE FRANZ FERDINAND AND HIS WIFE SOFLA
11.08.201810:46
```

**What the restorer did, line by line**

- `JSTROUGARSINOGTNL RANCA FERDINANDA INJEGOVU SUPRUO SOFIJU` — recased (shouted, Bosnian)
- `FROM THIS PLACE ON 28 JUNE 1914 GAVRILO PRINCIP ASSASSINATED THE HEIR TO THE AUSTRO-HUNGARIAN THRONE FRANZ FERDINAND AND HIS WIFE SOFLA` — kept — read as English
- `11.08.201810:46` — calm — left as read

**Source after recasing (what ON translates)**

```
Jstrougarsinogtnl ranca ferdinanda injegovu supruo sofiju
FROM THIS PLACE ON 28 JUNE 1914 GAVRILO PRINCIP ASSASSINATED THE HEIR TO THE AUSTRO-HUNGARIAN THRONE FRANZ FERDINAND AND HIS WIFE SOFLA
11.08.201810:46
```

**Translation — flag OFF (shipped today)**

```
JSTROUGARSINOGNL RANCA FERDINAND INJEGOVA SURUVA SOFIA FROM THIS PLACE ON 28 JUNE 1914 GAVRILO PRINCIPLE ASSASSINATED THE HEIR TO THE AUSTRO-HUNGARIAN THRONE FRANZ FERDINAND AND HIS WIFE SOFLA 11.08.201810:46
```

**Translation — flag ON (candidate)**

```
Jstrougarsinogtnl rance ferdinanda and his crush Sofia FROM THIS PLACE ON 28 JUNE 1914 GAVRILO PRINCIPLE ASSASSINATED THE HEIR TO THE AUSTRO-HUNGARIAN THRONE FRANZ FERDINAND AND HIS WIFE SOFLA 11.08.201810:46
```

**Verdict (blind): better / same / worse — _____**

---

### Entrance_to_Bosnia_and_Herzegovina_at_Brod.jpg

**Sign (OCR, as read)**

```
6P0A
DOBRO DOŠLI U BOSNU I HERCEGOVINU BROD WELCOME TO BOSNIA AND HERZEGOVINA BROD
```

**What the restorer did, line by line**

- `6P0A` — calm — left as read
- `DOBRO DOŠLI U BOSNU I HERCEGOVINU BROD WELCOME TO BOSNIA AND HERZEGOVINA BROD` — recased (shouted, Bosnian)

**Source after recasing (what ON translates)**

```
6P0A
Dobro došli u Bosnu i Hercegovinu brod welcome to bosnia and herzegovina brod
```

**Translation — flag OFF (shipped today)**

```
6P0A WELCOME TO BOSNIA AND HERZEGOVINA
```

**Translation — flag ON (candidate)**

```
6P0A Welcome to Bosnia and Herzegovina ship welcome to Bosnia and Herzegovina ship
```

**Verdict (blind): better / same / worse — _____**

---

### GiPS_Bus_Lion_s_City_03.jpg

**Sign (OCR, as read)**

```
CIPS-TNZLA
GIPS
ENTAR ODOVA LASER-
GIPS
T80-K-611
```

**What the restorer did, line by line**

- `CIPS-TNZLA` — recased (shouted, Bosnian)
- `GIPS` — calm — left as read
- `ENTAR ODOVA LASER-` — recased (shouted, Bosnian)
- `GIPS` — calm — left as read
- `T80-K-611` — calm — left as read

**Source after recasing (what ON translates)**

```
Cips-tnzla
GIPS
Entar odova laser-
GIPS
T80-K-611
```

**Translation — flag OFF (shipped today)**

```
CIPS-TNZLA GIPS ENTAR ODOVA LASER- GIPS T80-K-611
```

**Translation — flag ON (candidate)**

```
Cyps-tnzla GIPS Entar odova laser GIPS T80-K-611
```

**Verdict (blind): better / same / worse — _____**

---

### Mis_Irbina_Street_in_Sarajevo_03.jpg

**Sign (OCR, as read)**

```
BEERKA UDARAPONOVO
A38-0-357
an A90-T-292
```

**What the restorer did, line by line**

- `BEERKA UDARAPONOVO` — recased (shouted, Bosnian)
- `A38-0-357` — calm — left as read
- `an A90-T-292` — calm — left as read

**Source after recasing (what ON translates)**

```
Beerka udaraponovo
A38-0-357
an A90-T-292
```

**Translation — flag OFF (shipped today)**

```
BEERKA UDARAPONOVO A38-0357 A90-T-292
```

**Translation — flag ON (candidate)**

```
Beerka strikes again A38-0357 A90-T-292
```

**Verdict (blind): better / same / worse — _____**

---

### Plaque_at_the_Battle_of_the_Sutjeska_memorial.jpg

**Sign (OCR, as read)**

```
OVDJE POCIVA 33O1 BORAC SA
POGINULIM BORCIMA S
SUTJESKE
```

**What the restorer did, line by line**

- `OVDJE POCIVA 33O1 BORAC SA` — recased (shouted, Bosnian)
- `POGINULIM BORCIMA S` — recased (shouted, Bosnian)
- `SUTJESKE` — recased (shouted, Bosnian)

**Source after recasing (what ON translates)**

```
Ovdje pociva 33o1 borac sa
Poginulim borcima s
Sutjeske
```

**Translation — flag OFF (shipped today)**

```
33O1 BORAC WITH HERE MURDERED FIGHTERS SUTJESKA
```

**Translation — flag ON (candidate)**

```
Here rests a 33o1 fighter with Deadly Fighters Sutjeska
```

**Verdict (blind): better / same / worse — _____**

---

### Street_in_Međugorje.jpg

**Sign (OCR, as read)**

```
StAIl
SOUVENIRS SHOP Giuseppe
MINIMARKET
CO&DV CENTAN SHOP
ASRS
GOND ORO RE CIGORETI
YARAC
SHO
```

**What the restorer did, line by line**

- `StAIl` — calm — left as read
- `SOUVENIRS SHOP Giuseppe` — calm — left as read
- `MINIMARKET` — recased (shouted, Bosnian)
- `CO&DV CENTAN SHOP` — kept — read as English
- `ASRS` — calm — left as read
- `GOND ORO RE CIGORETI` — kept — read as English
- `YARAC` — calm — left as read
- `SHO` — calm — left as read

**Source after recasing (what ON translates)**

```
StAIl
SOUVENIRS SHOP Giuseppe
Minimarket
CO&DV CENTAN SHOP
ASRS
GOND ORO RE CIGORETI
YARAC
SHO
```

**Translation — flag OFF (shipped today)**

```
Stall SOUVENIRS SHOP Giuseppe MINIMARKET CO&DV CENTAN SHOP ASRS GOND ORO RE CIGORETI YARAC SHO
```

**Translation — flag ON (candidate)**

```
Stall SOUVENIRS SHOP Giuseppe Minimarket CO&DV CENTAN SHOP ASRS GOND ORO RE CIGORETI YARAC SHO
```

**Verdict (blind): better / same / worse — _____**

---

### Trg-žrtava-ŠB03078.JPG

**Sign (OCR, as read)**

```
ŠIROKI BRIJEG
UVUEKNAT.MJESTU
BRIJESK ZVON
```

**What the restorer did, line by line**

- `ŠIROKI BRIJEG` — recased (shouted, Bosnian)
- `UVUEKNAT.MJESTU` — recased (shouted, Bosnian)
- `BRIJESK ZVON` — recased (shouted, Bosnian)

**Source after recasing (what ON translates)**

```
Široki brijeg
Uvueknat.Mjestu
Brijesk zvon
```

**Translation — flag OFF (shipped today)**

```
SILVER BRIDGE UVEKNAT.PLACE BRIJESK ZVON
```

**Translation — flag ON (candidate)**

```
Wide hill Uvueknat. Brijesek bell
```

**Verdict (blind): better / same / worse — _____**

---

### Trg_Krajine_čajavčev_i_ulaz_u_gospodsku.jpg

**Sign (OCR, as read)**

```
MALBAŠIĆ CO
SIEMENS
LANACO
PNPOJEKT
VINKOP
5
```

**What the restorer did, line by line**

- `MALBAŠIĆ CO` — recased (shouted, Bosnian)
- `SIEMENS` — calm — left as read
- `LANACO` — calm — left as read
- `PNPOJEKT` — recased (shouted, Bosnian)
- `VINKOP` — calm — left as read
- `5` — calm — left as read

**Source after recasing (what ON translates)**

```
Malbašić co
SIEMENS
LANACO
Pnpojekt
VINKOP
5
```

**Translation — flag OFF (shipped today)**

```
MALBAŠI CO SIEMENS LANACO PNPOJEKT VINKOP 5
```

**Translation — flag ON (candidate)**

```
Malbašić SIEMENS LANACO Pnpoject VINKOP 5
```

**Verdict (blind): better / same / worse — _____**

---

### Tuzla_-_INA_petrol_station_2019_.jpg

**Sign (OCR, as read)**

```
NA CLASS 2.26 2.3 2.2 05 1.2
INA EP
0-24 SHOP CAFÉ WASH
-PUN
CLASS PLUS GORIVO ZA SVE TEMPERATURE
```

**What the restorer did, line by line**

- `NA CLASS 2.26 2.3 2.2 05 1.2` — calm — left as read
- `INA EP` — calm — left as read
- `0-24 SHOP CAFÉ WASH` — kept — read as English
- `-PUN` — calm — left as read
- `CLASS PLUS GORIVO ZA SVE TEMPERATURE` — recased (shouted, Bosnian)

**Source after recasing (what ON translates)**

```
NA CLASS 2.26 2.3 2.2 05 1.2
INA EP
0-24 SHOP CAFÉ WASH
-PUN
Class plus gorivo za sve temperature
```

**Translation — flag OFF (shipped today)**

```
NA CLASS 2.26 2.3 05 1.2 INA EP 0-24 SHOP CAF WASH -PUN CLASS PLUS FUEL FOR ALL TEMPERATURES
```

**Translation — flag ON (candidate)**

```
NA CLASS 2.26 2.3 05 1.2 INA EP 0-24 SHOP CAF WASH -PUN Class plus fuel for all temperatures
```

**Verdict (blind): better / same / worse — _____**

---

## The judgement this bar needs

For each pair above, fill the blind verdict: is the flag-ON English a better rendering of the sign than flag-OFF, the same, or worse? Turning the flag on by default is a product change; per the pre-registration it ships only on a clear majority of **better** with no serious regressions. Tally the verdicts here before the call — this file is the evidence, not the call itself.

