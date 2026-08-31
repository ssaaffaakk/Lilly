# Lilly v2 — full measurement pass
Started: Sat Aug 29 18:42:29 CEST 2026


## DINLEME — WER (yeni model, 200 clip)
```
objc[33350]: Class AVFFrameReceiver is implemented in both /Users/safaksurmeli/Desktop/Lilly/.venv/lib/python3.12/site-packages/av/.dylibs/libavdevice.62.3.102.dylib (0x1121803a8) and /Users/safaksurmeli/Desktop/Lilly/.venv/lib/python3.12/site-packages/cv2/.dylibs/libavdevice.61.3.100.dylib (0x117e203a8). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
objc[33350]: Class AVFAudioReceiver is implemented in both /Users/safaksurmeli/Desktop/Lilly/.venv/lib/python3.12/site-packages/av/.dylibs/libavdevice.62.3.102.dylib (0x1121803f8) and /Users/safaksurmeli/Desktop/Lilly/.venv/lib/python3.12/site-packages/cv2/.dylibs/libavdevice.61.3.100.dylib (0x117e203f8). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
model: /Users/safaksurmeli/Desktop/Lilly/models/lilly/listen
clips: 200

word error rate: 34.9%  (1360 wrong of 3901 words)

worst 3:
  00072.wav — 80%
    said:  Njegov renome epicentra luksuza započet je oko 400. godine nove ere, a trajao je do oko 1100. godine nove ere.
    heard: Njegu vrenom je epicentralukcu za započit joko 400-te godini novi eri, a trajevo je do oko 1100-te godini novi eri.
  00025.wav — 79%
    said:  Čopori lavova djeluju slično poput čopora vukova ili pasa, životinja koje su ponašanjem iznenađujuće slične lavovima (ali ne i drugim velikim mačkama), a takođe vrlo smrtonosne po svoj plijen.
    heard: Chopore lavova djelu je slična poput čoporavu kova alpasa životinja koji sponašnaju mi se naživajuće slične lavova, ali ne i drugi i veliki imačkama, a također vula s prtonasem po svoj plijeni.
  00116.wav — 78%
    said:  U toku borbe za nezavisnost koju je organizirao pokret Mau, mirni skup u gradu završio se ubistvom vrhovnog poglavara Tupua Tamasesea Lealofija III.
    heard: Tokuporbe za nizavisnost koja je organizirao pokrijed Mao, Milijsku pogradu završio se u bistvom vrhovnog poklavara Tupua, Tamasesa, Lealofi, a Trećjek.
```


## DINLEME — WER (önceki model, 200 clip)
```
objc[33408]: Class AVFFrameReceiver is implemented in both /Users/safaksurmeli/Desktop/Lilly/.venv/lib/python3.12/site-packages/av/.dylibs/libavdevice.62.3.102.dylib (0x11028c3a8) and /Users/safaksurmeli/Desktop/Lilly/.venv/lib/python3.12/site-packages/cv2/.dylibs/libavdevice.61.3.100.dylib (0x1185d43a8). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
objc[33408]: Class AVFAudioReceiver is implemented in both /Users/safaksurmeli/Desktop/Lilly/.venv/lib/python3.12/site-packages/av/.dylibs/libavdevice.62.3.102.dylib (0x11028c3f8) and /Users/safaksurmeli/Desktop/Lilly/.venv/lib/python3.12/site-packages/cv2/.dylibs/libavdevice.61.3.100.dylib (0x1185d43f8). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.
/Users/safaksurmeli/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/multiprocessing/resource_tracker.py:279: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
