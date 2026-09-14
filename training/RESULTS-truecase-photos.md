# RESULTS — truecase on real photographs — 14 September 2026

The photograph bar `training/RESULTS-truecase.md` left open: does `LILLY_TRUECASE` help on real signs, not uppercased FLORES? No English reference exists for these signs, so this is an A/B for the owner to read, not a chrF2. Nothing here trains or changes the served build.

- Reader: PP-OCRv6 cached output, fingerprint `afa719f3435ca2fe` (`data/ocr/real-photos/reader-output-paddle-v6.json`), the reader `app/ocr.py` serves.
- Engine: `bs-en` `app.translate.Engine`, the camera's default direction (`Lilly.translate_photo`).
- Method: each photo's whole OCR text through `translate()` twice, `LILLY_TRUECASE` off then on. This is the product path; the flag lives inside `translate()`.

## What the flag did

| | count | of 40 |
|---|---|---|
| photographs with OCR text | 33 | 82% |
| source the flag recased | 13 | 32% |
| **translation the flag changed** | **13** | **32%** |

The recaser only fires on predominantly-uppercase text (≥8 letters, ≥80% upper), so on the 20 photographs whose OCR was not shouted it did nothing and the two columns are identical. The 13 below are where a user would see a different answer.

---

### Assasination_Plaque.JPG

**Sign (OCR, as read)**

```
JSTROUGARSINOGTNL RANCA FERDINANDA INJEGOVU SUPRUO SOFIJU
FROM THIS PLACE ON 28 JUNE 1914 GAVRILO PRINCIP ASSASSINATED THE HEIR TO THE AUSTRO-HUNGARIAN THRONE FRANZ FERDINAND AND HIS WIFE SOFLA
11.08.201810:46
```

**Source after recasing**

```
Jstrougarsinogtnl ranca ferdinanda injegovu supruo sofiju
From this place on 28 june 1914 gavrilo princip assassinated the heir to the austro-hungarian throne franz ferdinand and his wife sofla
11.08.201810:46
```

**Translation — flag OFF (shipped today)**

```
JSTROUGARSINOGNL RANCA FERDINAND INJEGOVA SURUVA SOFIA FROM THIS PLACE ON 28 JUNE 1914 GAVRILO PRINCIPLE ASSASSINATED THE HEIR TO THE AUSTRO-HUNGARIAN THRONE FRANZ FERDINAND AND HIS WIFE SOFLA 11.08.201810:46
```

**Translation — flag ON (candidate)**

```
Jstrougarsinogtnl rance ferdinanda and his crush Sofia From this place on 28 June 1914 gavrilo principle assisted the heir to the Austro-Hungarian throne Franz Ferdinand and his wife sofla 11.08.201810:46
```

---

### Entrance_to_Bosnia_and_Herzegovina_at_Brod.jpg

**Sign (OCR, as read)**

```
6P0A
DOBRO DOŠLI U BOSNU I HERCEGOVINU BROD WELCOME TO BOSNIA AND HERZEGOVINA BROD
```

**Source after recasing**

```
6p0a
Dobro došli u Bosnu i Hercegovinu brod welcome to bosnia and herzegovina brod
```

**Translation — flag OFF (shipped today)**

```
6P0A WELCOME TO BOSNIA AND HERZEGOVINA
```

**Translation — flag ON (candidate)**

```
6p0a Welcome to Bosnia and Herzegovina ship welcome to Bosnia and Herzegovina ship
```

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

**Source after recasing**

```
Cips-tnzla
Gips
Entar odova laser-
Gips
T80-k-611
```

**Translation — flag OFF (shipped today)**

```
CIPS-TNZLA GIPS ENTAR ODOVA LASER- GIPS T80-K-611
```

**Translation — flag ON (candidate)**

```
Cyps-tnzla Gypsum Entar odova laser Gypsum T80-k-611
```

---

### Jewish_Street_Tuzla_Bosnia.jpg

**Sign (OCR, as read)**

```
D
FR1ZER
DOOLis
STELLA ARTOIS
TUORG
```

**Source after recasing**

```
D
Fr1zer
Doolis
Stella artois
Tuorg
```

**Translation — flag OFF (shipped today)**

```
D FR1ZER DOOLI Stella Artois TUORG
```

**Translation — flag ON (candidate)**

```
D Fr1zer Doolis Stella Artois Tuorg
```

---

### Mis_Irbina_Street_in_Sarajevo_03.jpg

**Sign (OCR, as read)**

```
BEERKA UDARAPONOVO
A38-0-357
an A90-T-292
```

**Source after recasing**

```
Beerka udaraponovo
A38-0-357
An a90-t-292
```

**Translation — flag OFF (shipped today)**

```
BEERKA UDARAPONOVO A38-0357 A90-T-292
```

**Translation — flag ON (candidate)**

```
Beerka strikes again A38-0357 An a90-t-292
```

---

### Plaque_at_the_Battle_of_the_Sutjeska_memorial.jpg

**Sign (OCR, as read)**

```
OVDJE POCIVA 33O1 BORAC SA
POGINULIM BORCIMA S
SUTJESKE
```

**Source after recasing**

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

---

### Sarajevo_Trolleybus-4416_Line-102_2011-11-05.jpg

**Sign (OCR, as read)**

```
EUROHAUS
```

**Source after recasing**

```
Eurohaus
```

**Translation — flag OFF (shipped today)**

```
EUROHAUS
```

**Translation — flag ON (candidate)**

```
Eurohaus
```

---

### Spanish_square_08034.JPG

**Sign (OCR, as read)**

```
TTE
ARTURO MUNOZ CASTELLANOS 13-MAY-93 ANGEL TORNEL YANEZ 02-JUN-93 FCO.JAVIER AGUILAR FERNANDEZ 11 -JUN-95 J.ANTONIO DELGADO FERNANDEZ 19 -JUN -93 SAMUEL AGUILAR JIMENEZ 19 -JUN-93 P AGUSTIN MATE COSTA 19 -JUN-93 .L.P. ISAAC PIÑEIRO VARELA 19 -JUN-93 FRANCISCO JIMENEZ JURADO 02-JUL-93 JOSE GAMEZ CHINEA 16-JUL-93 JOSE LEON GOMEZ 30-JUL-93 FERNANDO ALVAREZ RODRIGUEZ 04-DIC-93 SGTO. FERNANDO CASAS MARTIN 22-MAY-94 INTERPRETE MIRKOMIKULIC 22-MAY-94 CABO. D.ALVARO OJEDA BARRERA 04-NOV-94 SOLD. D RAUL BERRAOUERO FORCADA 04-N0V-94 SGTO. D ENRIQUE VEIGAS FERNANDEZ 22-MAR-96 SOLD. D SERGIO FERNANDEZ SANROMA 06-ABR-96 SGTO.I SANTIÀGO ARRANZ GONZALO 02-ENE-98 D.RAUL CABREJAS GIL 03-JUL-98 SGTO STTE. D.JOAQUIN VADILLO ROMERO 22-ENE-03 D.JOSE ANDRES YGARZA-PALOU 13-FEB-03 BGDA.GC. TTE D SANTLAGO HORMIGO LEDESMA 19-JUN-08 D.JOAQUIN LOPEZ MORENO 19-JUN-08 SGTO.
```

**Source after recasing**

```
Tte
Arturo munoz castellanos 13-may-93 angel tornel yanez 02-jun-93 fco.Javier aguilar fernandez 11 -jun-95 j.Antonio delgado fernandez 19 -jun -93 samuel aguilar jimenez 19 -jun-93 p agustin mate costa 19 -jun-93 .L.P. Isaac piñeiro varela 19 -jun-93 francisco jimenez jurado 02-jul-93 jose gamez chinea 16-jul-93 jose leon gomez 30-jul-93 fernando alvarez rodriguez 04-dic-93 sgto. Fernando casas martin 22-may-94 interprete mirkomikulic 22-may-94 cabo. D.Alvaro ojeda barrera 04-nov-94 sold. D raul berraouero forcada 04-n0v-94 sgto. D enrique veigas fernandez 22-mar-96 sold. D sergio fernandez sanroma 06-abr-96 sgto.I santiàgo arranz gonzalo 02-ene-98 d.Raul cabrejas gil 03-jul-98 sgto stte. D.Joaquin vadillo romero 22-ene-03 d.Jose andres ygarza-palou 13-feb-03 bgda.Gc. Tte d santlago hormigo ledesma 19-jun-08 d.Joaquin lopez moreno 19-jun-08 sgto.
```

**Translation — flag OFF (shipped today)**

```
TTE ARTURO MUNIZ CASTELLANOS 13-MAY-93 ANGEL TORNEL YANEZ 02-JUN-93 FCO.JAVIER AGUILAR FERNANDEZ 11-JUN-95 J.ANTONIO DELGADO FERNANDEZ 19-JUN-93 SAMUEL AGUILAR JIMENEZ 19-JUN-93 P AGUSTIN MINT. ISAAC PIEIRO VARELA 19-JUN-93 FRANCISCO JIMENEZ JURADO 02-JUL-93 JOSE GAMEZ CHINEA 16-JUL-93 JOSE LEON GOMEZ 30-JUL-93 FERNANDO ALVAREZ RODRIGUEZ 04-DIC-93 SGTO. FERNANDO CASAS MARTIN 22-MAY-94 INTERPRETE MIRKOMIKULIC 22-MAY-94 CABO. D.ALVARO OJEDA BARRERA 04-NOV-94 SOLD. D RAUL BERRAUERO FORCADA 04-N0V-94 SGTO. D ENRIQUE VEIGAS FERNANDEZ 22-MAR-96 SOLD. D SERGIO FERNANDEZ SANROMA 06-ABR-96 SGTO.I SANTIGO ARRANZ GONZALO 02-EN-98 D.RAUL CABRejAS GIL 03-JUL-98 SGTO STTE. D.JOAQUIN VADILLO ROMERO 22-ENE-03 D.JOSE ANDRES YGARZA-PALOU 13-FEB-03 BGDA.GC. TTE D SANTLAGO HORMIGO LEDESMA 19-JUN-08 D.JOAQUIN LOPEZ MORENO 19-JUN-08 SGTO.
```

**Translation — flag ON (candidate)**

```
Tte Arturo munoz castellanos 13-may-93 angel tornel yanez 02-jun-93 fco.Javier aguilar fernandez 11-jun-95 j.Antonio delgado fernandez 19-jun-93 samuel aguilar jimenez 19-jun-93 p agustin mate costa 19-jun-93. Isaac pieiro varela 19 Jun 93 francisco jimenez jurado 02 Jul 93 jose gamez chinea 16 Jul 93 jose leon gomez 30 Jul 93 fernando lvarez rodriguez 04 dic 93 sgto. Fernando casas martin 22-may-94 interpreta mirkomiculic 22-may-94 cabo. D.Alvaro gnaws barrera 04-nov-94 sold. D raul berraouero forcada 04-n0v-94 sgto. D enrique veigas fernandez 22-mar-96 sold. D sergio fernandez sanroma 06-abr-96 sgto.I santigo arranz gonzalo 02-ene-98 d.Raul cabrejas gil 03-jul-98 sgto stte. D.Joaquin vadillo Romero 22-ene-03 d.Jose andres ygarza-palou 13-feb-03 bgda.Gc. Tte d sanglago hormigo ledesma 19-jun-08 d.Joaquin lopez moreno 19-jun-08 sgto.
```

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

**Source after recasing**

```
Stail
Souvenirs shop giuseppe
Minimarket
Co&dv centan shop
Asrs
Gond oro re cigoreti
Yarac
Sho
```

**Translation — flag OFF (shipped today)**

```
Stall SOUVENIRS SHOP Giuseppe MINIMARKET CO&DV CENTAN SHOP ASRS GOND ORO RE CIGORETI YARAC SHO
```

**Translation — flag ON (candidate)**

```
Stile Souvenir shop Giuseppe Minimarket Co&dv centan shop Asrs Gond oro re cigoreti Yarac Sho
```

---

### Trg-žrtava-ŠB03078.JPG

**Sign (OCR, as read)**

```
ŠIROKI BRIJEG
UVUEKNAT.MJESTU
BRIJESK ZVON
```

**Source after recasing**

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

**Source after recasing**

```
Malbašić co
Siemens
Lanaco
Pnpojekt
Vinkop
5
```

**Translation — flag OFF (shipped today)**

```
MALBAŠI CO SIEMENS LANACO PNPOJEKT VINKOP 5
```

**Translation — flag ON (candidate)**

```
Malbašić Siemens Lanaco Pnpoject Vinkop 5
```

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

**Source after recasing**

```
Na class 2.26 2.3 2.2 05 1.2
Ina ep
0-24 shop café wash
-Pun
Class plus gorivo za sve temperature
```

**Translation — flag OFF (shipped today)**

```
NA CLASS 2.26 2.3 05 1.2 INA EP 0-24 SHOP CAF WASH -PUN CLASS PLUS FUEL FOR ALL TEMPERATURES
```

**Translation — flag ON (candidate)**

```
The class 2.26 2.3 2.2 05 1.2 Ina epic 0-24 shop cafe wash -Poon Class plus fuel for all temperatures
```

---

### WV_banner_NE_Bosnia_Tuzla_old_town.jpg

**Sign (OCR, as read)**

```
ON2
KAPNA
NEIDZER
PLANET SILVER GROUP
200YORK
```

**Source after recasing**

```
On2
Kapna
Neidzer
Planet silver group
200york
```

**Translation — flag OFF (shipped today)**

```
ON2 CAPS NEIDZER Planet Silver Group. 200YORK
```

**Translation — flag ON (candidate)**

```
He2 Kapna Neidzer Planet silver group 200york
```

---

## The judgement this bar needs

For each pair above: is the flag-ON English a better rendering of the sign than flag-OFF? Turning the flag on by default is a product change; it ships only if ON wins clearly across these, per the pre-registration `training/RESULTS-truecase.md` points to. This file is the evidence for that call, not the call itself.

