#!/bin/zsh
# Gece nöbetçisi — Claude uygulamasından bağımsız çalışır.
#
# Neden var: oturum içi zamanlayıcı oturumla ölür, uygulamanın zamanlanmış
# görevi de uygulama kapalıyken beklemeye geçer. Kullanıcı bilgisayarın başında
# olmayacak ve token limiti dolabilir. Bu döngü headless `claude -p` çağırır:
# limit doluysa çağrı başarısız olur, döngü bekler ve limit sıfırlandığında
# kendiliğinden devam eder. Kimseye sormaz, kaldığı yerden alır.
#
# Durdurmak için:  pkill -f night_watch.sh
set -u
REPO=/Users/safaksurmeli/Desktop/Lilly
CLAUDE=/Users/safaksurmeli/.local/bin/claude
# shellcheck source=logs_paths.sh
source "$REPO/scripts/logs_paths.sh"
LOG="$LOG_TRAINING/night-watch.log"
DEADLINE=$(( $(date +%s) + 13*3600 ))   # sabah rapordan sonra kendini bırakır

cd "$REPO" || exit 1

# Tek nöbetçi kuralı. 27 Ağustos gecesi bu döngü iki kez başladı (01:46:31 ve
# 01:47:07). İkisi de "defter 25 dakikadır sessiz" kontrolünü 36 saniye arayla
# geçti, ikisi de birer headless ajan doğurdu, iki ajan da okuyucu eğitimini
# başlattı — ve sonra her biri diğerinin koşusunu fazlalık sanıp öldürdü.
# Sonuç: iki eğitim başladı, ikisi de öldü, saat 01:55'te GPU boştu ve kimse
# bunu bilmiyordu. Defterin yaşına bakmak tavsiye niteliğindedir ve yarışır;
# bu kilit yarışmaz.
LOCKFILE=$REPO/.night_watch.lock
if [ -f "$LOCKFILE" ]; then
  other=$(head -1 "$LOCKFILE" 2>/dev/null | awk '{print $1}')
  if [ -n "$other" ] && kill -0 "$other" 2>/dev/null; then
    echo "$(date '+%H:%M:%S') nöbetçi $other zaten koşuyor, ikinciyi başlatmıyorum" >> "$LOG"
    exit 0
  fi
  echo "$(date '+%H:%M:%S') $other ölmüş, kilidi devralıyorum" >> "$LOG"
fi
echo "$$ $(date '+%Y-%m-%d %H:%M:%S')" > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT INT TERM
echo "$(date '+%H:%M:%S') nöbetçi başladı, bitiş $(date -r $DEADLINE '+%H:%M')" >> "$LOG"

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  # Araya girme kuralı. Üç nöbetçi var: bu oturumun kendi zamanlayıcısı,
  # uygulamanın saatlik görevi, ve bu döngü. Üçü birden çalışırsa hem jeton
  # israfı olur hem aynı eğitimi iki kez başlatabilirler. Koordinasyon için
  # ortak bir nabız kullanılıyor: gece defterine en son ne zaman yazıldığı.
  # Biri son 25 dakikada yazdıysa nöbet onda demektir, bu döngü karışmaz —
  # ve tam da onlar sustuğunda (jeton limiti dolduğunda) devreye girer.
  # Bir önceki tur hâlâ çalışıyorsa karışma. Döngü ajanı senkron çağırıyor,
  # yani bu normalde olamaz — ama başka bir nöbetçi ya da uygulamanın
  # zamanlanmış görevi doğurmuşsa olur, ve iki ajan aynı işi iki kez yapar.
  if pgrep -f "claude -p" > /dev/null; then
    echo "$(date '+%H:%M:%S') bir ajan hâlâ çalışıyor, tur atlandı" >> "$LOG"
    sleep 300
    continue
  fi
  if pgrep -f "train_ocr.py|train_translation.py" > /dev/null; then
    echo "$(date '+%H:%M:%S') yerel eğitim koşuyor, dokunmadım" >> "$LOG"
    sleep 900
    continue
  fi
  if [ -f NIGHT-LOG.md ]; then
    age=$(( $(date +%s) - $(stat -f %m NIGHT-LOG.md) ))
    if [ "$age" -lt 1500 ]; then
      echo "$(date '+%H:%M:%S') defter $((age/60)) dk önce yazılmış, nöbet başkasında" >> "$LOG"
      sleep 600
      continue
    fi
  fi

  echo "$(date '+%H:%M:%S') --- nöbet turu ---" >> "$LOG"
  "$CLAUDE" -p "$(cat <<'PROMPT'
Lilly projesinde gece nöbetindesin, /Users/safaksurmeli/Desktop/Lilly. Kullanıcı uyuyor, bilgisayarın başında değil. Bu koşunun sohbet geçmişi yok — durumu diskten oku.

Sırayla: `cat RESUME.md`, `tail -40 NIGHT-LOG.md`, `cat training/PREREGISTRATION.md`.

Koşan işleri kontrol et:
  .venv/bin/kaggle kernels status afaksrmeli/lilly-translation
  pgrep -fl "train_ocr|train_translation|evaluate"
  tail -5 "$LOG_TRAINING/wer-trained.log" "$LOG_OCR/ocr-train.log" 2>/dev/null

Karar ver ve uygula:
- Biten iş varsa sonucunu OKU ve doğrula — çıkış kodunu değil, ürettiği sayıyı. Eşikler ön-kayıtta ve yeniden yorumlanamaz.
- Düşen iş varsa log'u indir, sebebi bul, KÖKTEN düzelt, yeniden başlat. Bandaj yasak.
- GPU boşsa RESUME.md'deki sıradaki eğitimi `nohup` ile başlat ki bu oturum bitince ölmesin.
- Yapacak şey yoksa tek satır durum yaz.

Ne yaptıysan NIGHT-LOG.md sonuna tek satır ekle: saat + ne oldu + sayı.

DOKUNMA: data/clean/test.tsv, data/clean/valid.tsv, data/flores/, models/lilly/keep-2026-08-26/, CREDITS.md.
PROMPT
)" --permission-mode bypassPermissions >> "$LOG" 2>&1

  status=$?
  if [ $status -ne 0 ]; then
    # En muhtemel sebep token limiti. Bu bir hata değil, bir bekleme sebebi:
    # limit sıfırlanınca sonraki tur kendiliğinden çalışır.
    echo "$(date '+%H:%M:%S') çağrı başarısız (kod $status) — limit dolmuş olabilir, 30 dk sonra tekrar" >> "$LOG"
    sleep 1800
  else
    sleep 900
  fi
done
echo "$(date '+%H:%M:%S') nöbetçi süresi doldu, çıkıyor" >> "$LOG"
