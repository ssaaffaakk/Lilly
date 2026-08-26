# BosnianBench — Boşnakça-spesifik çeviri metriği

**Amaç:** Lilly'nin asıl iddiasını ölçmek — *"gerçek Boşnakça, jenerik Sırp-Hırvatça değil."*
BLEU bunu ölçemiyor: 20 kelimelik cümlede tek terim yanlışsa 1-2 puan düşer, sinyal
ortalamada kaybolur.

**Çıktı iki tane:**
1. Projeye: `base %X → Lilly %Y` tablosu
2. Dünyaya: HuggingFace'te açık test seti (bu alanda yok)

---

## H0 — Zemin (eğitim biterken, ~1 saat)

- [ ] **H0.1** Eğitim bitince `RESULTS.md`'deki BLEU tablosunu oku. Fark ne çıktı?
- [ ] **H0.2** Bir "hata avı" yap: test setinden 30 cümleyi base ve Lilly ile çevir,
      yan yana koy. **Amaç sayı değil, GÖZLEM:** modelin Boşnakça terimlerde nerede
      düştüğünü kendi gözünle görmek. Bu, kelime listenin tohumu olacak.
- [ ] **H0.3** Gördüğün her hatayı ham haliyle not al (`notes/hata_avi.md`).
      Kategori uydurma — önce topla, sonra kümele. *(Bu, error analysis'in "open coding"
      aşaması; Bölüm 3'te öğreteceğim yöntemin ta kendisi.)*

**Bitti sayılır:** 30 çeviri karşılaştırıldı, en az 10 somut hata not edildi.

---

## H1 — Ayrıştırıcı sözlük (1-2 gün)

Boşnakçayı Sırpça/Hırvatçadan ayıran terimleri topla. Hedef: **150-250 terim.**

- [ ] **H1.1** Dört kategoriyi doldur:

  | Kategori | Örnek | Not |
  |---|---|---|
  | **İjekavca ↔ ekavca** | vri**je**me/vreme, ml**ije**ko/mleko, d**ije**te/dete, čov**je**k/čovek | **EN SİSTEMATİK.** Kelime dağarcığından daha güvenilir: kural tabanlı, çok daha fazla cümlede geçiyor. Buradan başla |
  | Türkçe kökenli | čaršija, avlija, kapija, sevdah, insan, sahat, merhaba | Güçlü sinyal ama seyrek |
  | Korunan **h** | ka**h**va, la**h**ko, me**h**ko, hudovica | Sırpça: kafa, lako, meko |
  | Kültürel/dini | džamija, iftar, bajram, ezan, šehid | Bağlam gerektirir |

  > **Not (lilly-8a ölçümü, 2026-08-25):** Eğitim verisinin (334.790 satır) yalnızca
  > **%0.1'i** Sırpça/Hırvatça işareti taşıyor — yani veri gerçekten Boşnakça.
  > Aynı işaretleri test seti kurarken filtre olarak kullanabilirsin.

- [ ] **H1.2** Her terim için: doğru İngilizce karşılık(lar)ı yaz. **Birden fazla kabul
      edilebilir karşılık olabilir** (`čaršija` → bazaar / old town / marketplace) —
      hepsini listele, yoksa doğru cevabı yanlış sayarsın.
- [ ] **H1.3** Kaynak göster: her terimin nereden geldiğini yaz (dilbilim kaynağı,
      sözlük, ya da "ana dil bilgisi"). Bu, setin savunulabilirliği için şart.
- [ ] **H1.4** ⚠️ **Doğrulatma:** listeyi Boşnakça ana dili olan en az 1 kişiye
      göster. Kendi kurduğun sınavın doğru olduğunu tek başına iddia edemezsin.

**Bitti sayılır:** `bench/terms.tsv` — 150+ terim, kategori, kabul edilen çeviriler, kaynak.

---

## H2 — Test cümleleri (2-3 gün)

- [ ] **H2.1** Her terim için 1-2 **gerçek cümle** bul. Kaynak öncelik sırası:
      (a) `data/clean/` içindeki mevcut korpus — terimi geçen cümleleri grep'le
      (b) Boşnakça haber siteleri / Wikipedia
      (c) Son çare: kendin yaz — ama sentetik olduğunu işaretle
- [ ] **H2.2** Her cümle için İngilizce referans çeviri yaz.
- [ ] **H2.3** **Hedef terimi işaretle** — cümlenin hangi kelimesi karar veriyor:
      ```json
      {"bs": "Sjeli smo u staru čaršiju i popili kahvu.",
       "en": "We sat down in the old bazaar and had coffee.",
       "hedefler": [{"terim": "čaršija", "kabul": ["bazaar", "old town", "marketplace"]},
                    {"terim": "kahva",   "kabul": ["coffee"]}],
       "kategori": "turkism", "kaynak": "SETIMES"}
      ```
- [ ] **H2.4** **Kontrol grubu ekle:** 50 cümle de Boşnakça-spesifik terim İÇERMESİN.
      Neden: modelin genel olarak mı iyileştiğini yoksa sadece bu terimlerde mi
      iyileştiğini ayırt etmek için. Kontrol grubu olmadan kazancın nereden geldiğini
      bilemezsin.

**Bitti sayılır:** `bench/bosnianbench.jsonl` — 200 hedefli + 50 kontrol cümlesi.

---

## H3 — Skorlama (1 gün)

- [ ] **H3.1** `bench/score.py` yaz: modeli çalıştır, her cümlede hedef terimin
      kabul listesindeki bir karşılığı çıktıda geçiyor mu → doğru/yanlış.
- [ ] **H3.2** Kategori bazında rapor et, tek ortalama verme:
      ```
                    turkism   h-korunma   kültürel   ijekavca   KONTROL
      base            38%        61%        44%        71%       —
      Lilly           71%        79%        58%        74%       —
      ```
- [ ] **H3.3** **Güven aralığı ekle.** 200 cümlede %5'lik fark gürültü olabilir.
      *(Bunu Bölüm 3'te konuşacağız — Evan Miller'ın "Adding Error Bars to Evals"i.)*
- [ ] **H3.4** ⚠️ **Metriğin kendisini doğrula — üç kontrol.** *Ölçü aletini ölçmeden
      ona güvenme.* Tek tek elle puanlamak yetmez; şu üçünü koş:

  - [ ] **(a) Pozitif kontrol:** İnsan referans çevirisini metrikten geçir.
        **~%100 almalı.** Almıyorsa skorlama kodun bozuk: büyük/küçük harf, çekim
        ekleri, eksik eş anlamlı listesi.
  - [ ] **(b) Negatif kontrol:** Sırpça/Hırvatça ağırlıklı olduğunu bildiğin bir
        modeli geçir. **Belirgin DÜŞÜK almalı.** Yüksek alıyorsa metrik ayrıştırmıyor —
        terim listen yeterince ayırt edici değil.
  - [ ] **(c) Eşleştirilmiş çift:** Aynı cümlenin iki varyantını yan yana koy
        (`kahvu`/`kafu`, `vrijeme`/`vreme`). Metrik ikisine **farklı** davranmalı.
        Davranmıyorsa hedef terim işaretlemesi çalışmıyor.

  (a)+(b) tutarsa metrik ayrıştırıyor; (c) mekanizmanın doğru yerde çalıştığını gösterir.

  > **Bu boş bir uyarı değil.** Aynı projede ölçüm kodu bugün iki kez yanlış çıktı:
  > dolgu (padding) hatası 8.7 BLEU kaybettiriyordu ve tablo tertemiz görünüyordu;
  > OCR ölçümü sahte hata sayıyordu. Birincisi (a)'ya, ikincisi (c)'ye takılırdı.

**Bitti sayılır:** kategori tablosu + elle doğrulama uyum oranı.

---

## H4 — Analiz (1 gün) — projenin en değerli kısmı

- [ ] **H4.1** Lilly'nin hâlâ kaçırdığı terimleri incele. Ortak yanları ne?
      (eğitim verisinde az mı geçiyor, çok anlamlı mı, bağlam mı gerekiyor?)
- [ ] **H4.2** **Ters vaka ara:** base'in bildiği ama Lilly'nin bozduğu terim var mı?
      Varsa bu **catastrophic forgetting** — fine-tuning'in bilinen yan etkisi ve
      bulursan raporlamak seni ayrıştırır.
- [ ] **H4.3** Kontrol grubuna bak: genel kalite düştü mü? Düştüyse trade-off'u yaz.
- [ ] **H4.4** Tek sayfa bulgu yazısı: yöntem, sonuç, sınırlar, ne öğrendim.

**Bitti sayılır:** `bench/FINDINGS.md` — dürüst, sınırları yazılmış rapor.

---

## H5 — Yayınla (yarım gün) — "portfolyo" → "katkı"

- [ ] **H5.1** HuggingFace dataset olarak yükle: `bosnianbench`
- [ ] **H5.2** Dataset card yaz: neden var, nasıl kuruldu, nasıl kullanılır,
      **sınırları neler** (sınırları yazmak güvenilirliği artırır, azaltmaz)
- [ ] **H5.3** Lisans seç (CC-BY-4.0 uygun), kaynakları atfet
- [ ] **H5.4** README'ye ve LinkedIn'e koy: *"Boşnakça çeviri kalitesini ölçen açık
      test seti — bu alanda ilk"*

**Bitti sayılır:** Public HF dataset linki.

---

## Zaman ve zorluk

| Hedef | Süre | Zorluk | Neden değerli |
|---|---|---|---|
| H0 | 1 saat | kolay | Gözlem, listenin tohumu |
| H1 | 1-2 gün | orta | Domain bilgisi + doğrulatma |
| H2 | 2-3 gün | orta (yorucu) | Setin gövdesi |
| H3 | 1 gün | **zor** | Ölçü aletini doğrulamak |
| H4 | 1 gün | **zor** | Asıl düşünme işi |
| H5 | yarım gün | kolay | Katkıya dönüşüm |

**Toplam: ~1 hafta yoğun çalışma.**

## Riskler — önceden bil

- **Lilly bu sette de iyileşmezse?** Bu da bir sonuç ve dürüstçe raporlamak değerlidir.
  ~~Muhtemel sebep: eğitim verisi de ağırlıklı Sırp-Hırvatça kaynaklı olabilir.~~
  **ÖLÇÜLDÜ VE ÇÜRÜTÜLDÜ:** lilly-8a'nın ölçümüne göre eğitim verisinin yalnızca
  %0.1'i Sırpça/Hırvatça işareti taşıyor. Veri sorunu değil. Yani iyileşme çıkmazsa
  sebebi başka yerde aranmalı: hiperparametreler, LoRA rank/hedef modüller, ya da
  base modelin bu dil ailesinde zaten tavana yakın olması.
  *(Bu satır, "riskleri varsaymak yerine ölçmenin" örneği: tahmin bir ölçümle düştü.)*
- **Kendi setini kendin kurup kendin geçmek:** en büyük tuzak. Terimleri modelin
  bildiği şeylerden seçersen sınavı kolaylaştırmış olursun. Bu yüzden H1'de terimler
  **dilbilimsel kriterle** seçilecek, modelin performansına bakarak değil.
- **Ana dil doğrulaması olmadan yayınlama.** Yanlış bir test seti yayınlamak,
  hiç yayınlamamaktan kötüdür.
