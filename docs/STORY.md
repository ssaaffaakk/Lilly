# Why I Built This

I applied for an internship. They didn’t take me.

Not because my code was weak. The reason was shorter, colder: I didn’t know
Bosnian.

I had heard some version of that sentence before — during internships, job
applications, and conversations that ended before they really began.

Same sentence. Different room.

At some point, language stopped feeling like a language and started feeling like
a door. A door I was always standing on the wrong side of.

The frustrating part was that Bosnian itself was never the problem. It was
learnable. The problem was the technology around it.

Most tools treated Bosnian as a rounding error: one more language inside a
broader South Slavic category, perhaps a different flag on the interface. The
technology could recognize the region. But recognizing South Slavic is not the
same as understanding Bosnian.

So I started asking myself a few simple questions:

If someone is speaking Bosnian, why can’t I press a microphone and hear it in
English?

If someone writes something on a board, why can’t I take a photo and understand
it?

If I need to respond, why can’t I type what I mean and have it spoken out loud?

These didn’t feel like impossible problems.

They felt like problems nobody had bothered to solve properly.

So I decided to build it myself.

I trained on real Bosnian, deliberately. I measured it. I tested it. And I
published the numbers — including the ones that didn’t flatter me:

- **chrF2 67.47** on 2,009 FLORES pairs
- **34.9% WER** on held-out Bosnian speech, now training whisper-large-v3
- **54.7%** of words read correctly per photograph, up from 36%

Type. Speak. Snap.

That decision became Lilly.

They told me I didn’t know Bosnian.

So I built the thing that understands it.

I build things because when I run into a problem, I’d rather solve it than accept
that it can’t be solved.

— [@ssaaffaakk](https://github.com/ssaaffaakk)

---

## Türkçe

Internship’e başvurdum. Beni kabul etmediler.

Kodum zayıf olduğu için değil. Gerekçe daha kısa, daha soğuktu: Boşnakça
bilmiyordum.

Bu cümlenin bir versiyonunu daha önce de duymuştum — internship’lerde, iş
başvurularında ve gerçekten başlamadan biten konuşmalarda.

Aynı cümle. Farklı oda.

Bir noktada dil, dil gibi değil kapı gibi gelmeye başladı. Her seferinde yanlış
tarafında durduğum bir kapı.

Sinir bozucu tarafı şu: sorun Boşnakçanın kendisi hiç değildi. Öğrenilebilirdi.
Sorun etrafındaki teknolojiydi.

Araçların çoğu Boşnakçayı yuvarlama hatası sayıyordu: daha geniş bir South
Slavic kategorisinin içindeki bir dil daha, belki arayüzde farklı bir bayrak.
Teknoloji bölgeyi tanıyabiliyordu. Ama South Slavic’i tanımak, Boşnakçayı
anlamakla aynı şey değil.

Ben de kendime birkaç basit soru sormaya başladım:

Biri Boşnakça konuşuyorsa, neden mikrofona basıp bunu İngilizce duymayayım?

Biri tahtaya bir şey yazıyorsa, neden fotoğrafını çekip anlamayayım?

Cevap vermem gerekiyorsa, neden ne demek istediğimi yazıp sesli söyletmeyeyim?

Bunlar imkânsız problemler gibi durmuyordu.

Kimsenin doğru şekilde çözmeye zahmet etmediği problemler gibi duruyordu.

Ben de kendim yapmaya karar verdim.

Gerçek Boşnakça üzerine, bilerek train ettim. Ölçtüm. Test ettim. Ve rakamları
yayınladım — beni okşamayanlar dahil:

- 2.009 FLORES çiftinde **chrF2 67.47**
- Ayrı tutulan Boşnakça konuşmada **%34.9 WER**; şimdi whisper-large-v3 train ediliyor
- Fotoğraf başına kelimelerin **%54.7’si** doğru okundu (%36’dan yukarı)

Type. Speak. Snap.

Bu karar Lilly oldu.

Bana Boşnakça bilmediğimi söylediler.

Ben de onu anlayan şeyi yaptım.

İnşa ediyorum, çünkü bir problemle karşılaştığımda onu çözmeyi, çözülemeyeceğini
kabul etmeye tercih ediyorum.
