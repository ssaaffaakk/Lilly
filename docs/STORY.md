# Why I Built This

I applied for an internship. They did not take me.

Not because the code was weak. Not because the interview went badly. The reason
came back shorter and colder than that: **you do not know Bosnian. You will not
understand.**

I had heard that sentence before. When I applied for an internet password. When
I applied for jobs. When they talked about the classes I would be sitting in.
Same sentence, different room. The language stopped being a language and became
a door — and I was always on the wrong side of it.

One rejection you can file away. This one followed me around. It starts as their
opinion and slowly turns into your own voice. You begin to argue for the other
side. You get quieter in rooms. You start making yourself smaller so the gap is
less obvious.

I stopped.

Because the problem was never that Bosnian is unlearnable. The problem was that
every tool I could reach treated Bosnian as a rounding error. The big translators
either skip it or fold it into one South Slavic bucket — one model, a different
flag on the label, and they hope nobody in Sarajevo notices the difference.
**Recognizing South Slavic is solved. Serving Bosnian is not.**

So I asked the only questions that mattered to me as a student sitting in that
room:

If they are speaking around me — why can't I press a microphone and hear it in
English?  
If they write on the board — why can't I photograph it and understand it?  
If I need to answer — why can't I type what I mean and let it be spoken out loud?

Nobody was going to build that for me. So I built it.

Not a wrapper. Not a demo. Not a flag swap on someone else's checkpoint. I
trained the listener on Bosnian speech. I trained the reader on real Bosnian
street text — signs, menus, boards — with the letters everyone else quietly
drops: **č ć đ š ž**. Then I measured it, because a story with no numbers is
just a feeling:

- **chrF2 67.47** on 2,009 FLORES pairs
- **34.9% WER** on held-out Bosnian speech, now training whisper-large-v3
- **54.7%** of words read correctly per photograph, up from 36%

Those numbers are honest. Where the benchmark does not flatter me, I publish that
too. The whole point was to stop being told what I can and cannot understand — so
I am not going to start lying to myself with a nicer metric.

That decision is Lilly.

**Type. Speak. Snap.** Bosnian in, English out, on my own machine, offline, built
by the person who was told he would not understand.

I have done this before. I built ProjectBuddy after two university projects went
wrong for reasons that had nothing to do with my ability — a partner who did not
deliver, then a semester where I could not find a partner at all. Two projects.
Two failures. A broken system for forming teams, and marks I paid for it. So I
built the platform instead of complaining about the group chat.

This is the same reflex, aimed at language instead of teamwork. They told me I
would not understand Bosnian. So I built the thing that understands it.

I build things because I run into problems and refuse to accept that they can't
be solved.

— [@ssaaffaakk](https://github.com/ssaaffaakk)

---

## Türkçe

Internship'e başvurdum. Kabul etmediler.

Kod zayıf olduğu için değil. Mülakat kötü gittiği için değil. Gelen cevap daha
kısa ve daha soğuktu: **Boşnakça bilmiyorsun. Anlamazsın.**

Bu cümleyi ilk kez duymuyordum. İnternet şifresi başvurusunda. İş
başvurularında. Gireceğim derslerden bahsedilirken. Aynı cümle, farklı oda. Dil
bir dil olmaktan çıkıp kapıya dönüştü — ve ben her seferinde yanlış taraftaydım.

Bir reddi bir kenara koyabilirsin. Bu peşimi bırakmadı. Onların fikri olarak
başlıyor, yavaşça senin sesin oluyor. Kendi aleyhine savunma yapmaya
başlıyorsun. Odalarda sesin kısılıyor. Fark az görünsün diye kendini küçültüyorsun.

Durdum.

Çünkü sorun Boşnakçanın öğrenilemez olması değildi. Sorun, elimin uzandığı her
aracın Boşnakçayı yuvarlama hatası saymasıydı. Büyük çeviriciler ya onu hiç
listeye almıyor ya da tek bir South Slavic kovasına atıyor — tek model, etikette
farklı bir bayrak ve Saraybosna'da kimsenin farkı görmeyeceği umudu.
**South Slavic'i tanımak çözüldü. Boşnakçaya hizmet etmek çözülmedi.**

O odada oturan bir öğrenci olarak benim için önemli olan tek soruları sordum:

Etrafımda konuşuyorlarsa — neden mikrofona basıp İngilizce duymayayım?  
Tahtaya yazıyorlarsa — neden fotoğrafını çekip anlamayayım?  
Cevap vermem gerekiyorsa — neden yazıp sesli söyletmeyeyim?

Bunu benim için kimse yapmayacaktı. Ben yaptım.

Sarmalayıcı değil. Demo değil. Başkasının checkpoint'ine bayrak yapıştırmak
değil. Dinleyiciyi Boşnakça konuşma üzerine train ettim. Okuyucuyu gerçek Boşnak
sokak metniyle — tabelalar, menüler, tahtalar — ve herkesin sessizce düşürdüğü
harflerle train ettim: **č ć đ š ž**. Sonra ölçtüm, çünkü rakamsız hikâye sadece
bir histir:

- 2.009 FLORES çiftinde **chrF2 67.47**
- Ayrı tutulan Boşnakça konuşmada **%34.9 WER**; şimdi whisper-large-v3 train ediliyor
- Fotoğraf başına kelimelerin **%54.7'si** doğru okundu (%36'dan yukarı)

Bu rakamlar dürüst. Benchmark beni okşamadığı yerde onu da yayınlıyorum. Bütün
amaç neyi anlayıp anlamayacağımın bana söylenmesini bitirmekti — o yüzden daha
güzel bir metrikle kendime yalan söylemeye başlamayacağım.

Lilly bu karar.

**Type. Speak. Snap.** Boşnakça girer, İngilizce çıkar; kendi makinemde, offline,
anlamayacağı söylenen kişi tarafından yapıldı.

Bunu daha önce de yaptım. ProjectBuddy'yi, üniversitede iki proje benim
yeteneğimle hiç ilgisi olmayan sebeplerle battıktan sonra yaptım: teslim etmeyen
bir partner, sonra hiç partner bulamadığım bir dönem. İki proje. İki başarısızlık.
Takım kurmanın bozuk sistemi ve bedelini ödediğim notlar. Grup sohbetine
söylenmek yerine platformu kurdum.

Bu da aynı refleks — bu kez takım değil, dil. Boşnakça anlamayacağımı söylediler.
Ben de onu anlayan şeyi yaptım.

Sorun çıkınca "çözülemez" demeyi kabul etmiyorum. O yüzden inşa ediyorum.
