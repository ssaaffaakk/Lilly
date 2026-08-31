#!/usr/bin/env python3
"""Fine-tune Lilly's photo reading on Bosnian letters.

Reading a photo is two models: one finds where the text is, one reads it. Only
the reader can be retrained — the finder's training code was never released —
and the reader is where the actual failure lives. Measured on real photos, the
shipped reader turns `Čaršija` into `Caršija` and `Đačka` into `Backa`: the
diacritics that separate Bosnian from plain Latin are exactly what it drops.

The character it needs is already in the model's vocabulary — all ten of
č ć đ š ž Č Ć Đ Š Ž are in its 351-character set. It is not that it cannot write
them, it is that it has barely seen them. So this continues training the shipped
weights on Bosnian words rather than starting over, and the vocabulary is left
untouched so every pretrained weight loads.

    python3 data/scripts/generate_ocr_data.py --count 20000   # make the images
    python3 training/train_ocr.py                             # then this

There is no separate evaluation step, and this docstring named one -- an
evaluate_ocr.py that has never existed. A target of "67.1% words, 69.4%
diacritics" was carried in the night notes for four hours as though that script
had produced it; nothing had. The run scores itself instead: it measures the
published weights on the held-out crops before training and the trained ones
after, in one process, with one scorer, and refuses to install the result
unless the diacritics improved. Those two numbers are the comparison. A number
that did not come out of this run's own before/after pair is not a measurement
of it.

Writes models/lilly/read/latin_g2.pth, keeping the previous one beside it as
latin_g2-previous.pth so there is a way back and something to measure against.
"""
import argparse
import atexit
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Apple's GPU backend has no CTC loss, which is the loss this model trains with.
# Without this the run dies at the first step; with it, that one operation falls
# back to the CPU and everything else stays on the GPU. Must be set before torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

READ_DIR = REPO_ROOT / "models" / "lilly" / "read"
# The weights easyocr publishes, by their own checksum. Training must start from
# these and nothing else: a run that starts from a previous run's output measures
# itself against that output, and its "before" number is meaningless. One
# four-step test run left a toy model in place and the next real run compared
# against it without noticing.
PRISTINE_MD5 = "469869130aad1a34e8f9086f4262bc59"
OCR_DATA = REPO_ROOT / "data" / "ocr"
IMG_HEIGHT = 64          # what the shipped reader was trained at
MAX_WIDTH = 600
DIACRITICS = "čćđšžČĆĐŠŽ"
MIN_STEPS = 200          # below this a run has not trained, it has only run
LOCK = READ_DIR / ".train_ocr.lock"


def pid_alive(pid: int) -> bool:
    """Signal 0 asks the kernel about a pid without sending anything."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True      # not ours to signal, but it exists
    return True


def pid_is_this_trainer(pid: int) -> bool:
    """Alive is not enough — the pid has to still be *this* script.

    The kernel recycles pids, and on a machine that has been churning through
    background work all day it recycles them fast. On 28 August a training run
    died when macOS tore down its Metal compiler service; a few minutes later
    pid 1602 belonged to ManagedPreferencesSubscriber, an OS daemon. The lock
    still named 1602, `pid_alive` cheerfully said yes, and the next run refused
    to start with "pid 1602 is already training the reader" -- guarding weights
    against a preferences daemon.

    So: ask what the pid actually is. Only a python process running train_ocr
    counts. Anything else means the trainer is gone and the lock is stale,
    whoever owns the number now.
    """
    if not pid_alive(pid):
        return False
    try:
        out = subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return True          # cannot tell -- keep the guard rather than race
    return "train_ocr" in out


def hold_lock() -> None:
    """Refuse to start if another run already has the reader.

    Not hypothetical either. Three things watch this repo overnight — this
    session's own timer, the app's scheduled task, and scripts/night_watch.sh —
    and each starts a headless agent that reads the same disk state. On the
    night of 27 August two of them looked within four minutes of each other,
    both saw no training running and a free GPU, and both started this script.
    They loaded the same pristine weights, trained separately, and were both
    heading for the same latin_g2.pth: last writer wins, and the winner's
    "before" number belongs to a model the loser had already replaced. They
    also shared one log, and the second truncated the first's.

    The callers cannot fix this between themselves — a check in one watcher
    cannot see an agent another watcher is about to spawn. So the guard lives
    here, next to the weights it protects, where every path in has to pass it.
    """
    if LOCK.exists():
        try:
            owner = int(LOCK.read_text(encoding="utf-8").split()[0])
        except (ValueError, IndexError, OSError):
            owner = None
        if owner is not None and owner != os.getpid():
            if pid_is_this_trainer(owner):
                raise SystemExit(
                    f"pid {owner} is already training the reader (lock: {LOCK}). "
                    f"Two runs would race to overwrite {READ_DIR/'latin_g2.pth'} "
                    f"and each would report a before/after pair the other "
                    f"invalidated. Wait for it, or kill it first.")
            print(f"stale lock from pid {owner} — not a trainer any more, taking over")
    LOCK.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                    encoding="utf-8")
    atexit.register(release_lock)


def release_lock() -> None:
    """Only ever drop a lock this process owns."""
    try:
        if LOCK.exists() and int(LOCK.read_text(encoding="utf-8").split()[0]) == os.getpid():
            LOCK.unlink()
    except (ValueError, IndexError, OSError):
        pass


def load_charset() -> str:
    import easyocr.config as config
    return config.recognition_models["gen2"]["latin_g2"]["characters"]


def build_model(num_class: int):
    """EasyOCR's own network, so every pretrained weight lands where it belongs."""
    from easyocr.model.vgg_model import Model
    return Model(input_channel=1, output_channel=256, hidden_size=256,
                 num_class=num_class)


def checksum(path: Path) -> str:
    import hashlib
    return hashlib.md5(path.read_bytes()).hexdigest()


def ensure_pristine(weights: Path) -> None:
    """Start from the published weights, restoring them if something replaced them."""
    if checksum(weights) == PRISTINE_MD5:
        return
    backup = weights.with_name(weights.stem + "-previous.pth")
    if backup.exists() and checksum(backup) == PRISTINE_MD5:
        shutil.copy(backup, weights)
        print(f"{weights.name} was not the published model — restored from {backup.name}")
        return
    raise SystemExit(
        f"{weights} is not the published reader and no clean copy is beside it. "
        f"Re-fetch it (python3 scripts/fetch_models.py) before training, or the "
        f"before/after comparison measures nothing.")


def load_weights(model, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    # saved from a DataParallel wrapper, so every key carries a module. prefix
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def save_weights(model, path: Path) -> None:
    """Back in the shape the app loads: keys prefixed, plain state dict."""
    torch.save({f"module.{k}": v for k, v in model.state_dict().items()}, path)


def prepare(image_path: Path) -> torch.Tensor:
    """What the app does to a crop before reading it — grayscale, 64 tall, scaled
    to [-1, 1]. Padding happens per batch, not here."""
    from torchvision import transforms
    img = Image.open(image_path).convert("L")
    ratio = img.width / max(img.height, 1)
    width = min(max(int(IMG_HEIGHT * ratio), 8), MAX_WIDTH)
    img = img.resize((width, IMG_HEIGHT), Image.BICUBIC)
    return transforms.ToTensor()(img).sub_(0.5).div_(0.5)


class Crops(Dataset):
    """A folder of word images plus a gt.txt of "image<TAB>text"."""

    def __init__(self, folder: Path, limit=None):
        self.folder = folder
        self.rows = []
        gt = folder / "gt.txt"
        if gt.exists():
            for line in gt.read_text(encoding="utf-8").splitlines():
                parts = line.split("\t")
                if len(parts) == 2 and parts[1].strip() and (folder / parts[0]).exists():
                    self.rows.append((parts[0], parts[1].strip()))
        if limit and limit < len(self.rows):
            # Take an even stride through the file, not the first N. gt.txt is
            # written in whatever order the generators and any repair produced,
            # and after the split was rebuilt by label text the synthetic rows
            # landed first: the first 500 rows of data/ocr/valid were 500
            # synthetic crops and not one of its 408 photographs. A reader
            # scored on that subset is scored on the easy half of its job, and
            # the number that comes out is not the reader's accuracy. Striding
            # keeps whatever mix the file has, and keeps it the same every run.
            step = len(self.rows) / limit
            self.rows = [self.rows[int(i * step)] for i in range(limit)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        name, text = self.rows[idx]
        return prepare(self.folder / name), text

    def subset(self, real: bool):
        """Real crops are every filename that is not syn*. Pass-5 was refused
        on a pooled valid set that is mostly synthetic — the easy half."""
        other = Crops.__new__(Crops)
        other.folder = self.folder
        if real:
            other.rows = [(n, t) for n, t in self.rows if not n.startswith("syn")]
        else:
            other.rows = [(n, t) for n, t in self.rows if n.startswith("syn")]
        return other


def collate(batch):
    """Pad to the widest image in this batch, not to a fixed width.

    A word image is about 168 pixels wide at this height and the widest is under
    400, so padding everything to the 600-pixel ceiling spends 70% of the
    computation on blank space. Padding repeats the last column, which is what
    the app does, so the model sees the same thing either way.
    """
    widest = max(b[0].size(2) for b in batch)
    images = torch.zeros(len(batch), 1, IMG_HEIGHT, widest)
    for i, (tensor, _) in enumerate(batch):
        width = tensor.size(2)
        images[i, :, :, :width] = tensor
        if width < widest:
            images[i, :, :, width:] = tensor[:, :, width - 1:width].expand(
                1, IMG_HEIGHT, widest - width)
    return images, [b[1] for b in batch]


def decode(logits, converter):
    # the converter indexes with numpy semantics, so it needs numpy — handing it a
    # torch tensor indexes with a tuple instead and returns arrays, not characters
    sizes = [logits.size(1)] * logits.size(0)
    _, indices = logits.max(2)
    return converter.decode_greedy(indices.view(-1).cpu().numpy(), sizes)


def surviving_diacritics(truth: str, pred: str) -> tuple:
    """How many of the Bosnian letters in `truth` are still there in `pred`.

    Counted, not aligned. Walking the two strings in step — `zip(truth, pred)` —
    only holds while they are the same length: one dropped or inserted character
    shifts every position after it, and each shifted position then reads as a
    Bosnian letter that came out wrong. On data/ocr/valid with the shipped
    reader that scoring reported 106 diacritic errors, of which only 5 were a
    diacritic the model had actually lost; the rest were the shift. Comparing
    counts of each letter cannot be moved by an error elsewhere in the word.
    """
    right = total = 0
    for ch in DIACRITICS:
        want = truth.count(ch)
        if want:
            total += want
            right += min(want, pred.count(ch))
    return right, total


def accuracy(model, loader, converter, device):
    """Two numbers: whole words right, and how often a Bosnian letter survives."""
    model.eval()
    exact = total = 0
    dia_right = dia_total = 0
    with torch.no_grad():
        for images, texts in loader:
            preds = decode(model(images.to(device), None), converter)
            for pred, truth in zip(preds, texts):
                total += 1
                exact += pred.strip() == truth.strip()
                right, want = surviving_diacritics(truth, pred)
                dia_right += right
                dia_total += want
    model.train()
    return (100 * exact / max(total, 1),
            100 * dia_right / max(dia_total, 1), dia_total)


def diacritic_cases(model, crops, converter, device):
    """Every held-out real crop that carries a Bosnian letter.

    The crop-gate letter number is 25 occurrences across about 23 images.
    64% -> 60% is exactly 16/25 -> 15/25 — one letter on one crop. Pass-8
    and pass-9 both printed only the aggregate, so the next mix had to guess
    which word moved. This list is that word.
    """
    if not crops:
        return []
    model.eval()
    rows = []
    loader = DataLoader(crops, batch_size=16, collate_fn=collate, num_workers=0)
    with torch.no_grad():
        for images, texts in loader:
            preds = decode(model(images.to(device), None), converter)
            for pred, truth in zip(preds, texts):
                right, want = surviving_diacritics(truth, pred)
                if want == 0:
                    continue
                rows.append((truth, pred.strip(), right, want))
    model.train()
    return rows


def print_diacritic_cases(label, rows):
    letters = sum(want for _, _, _, want in rows)
    print(f"{label} diacritic crops ({letters} letters / {len(rows)} crops):")
    for truth, pred, right, want in rows:
        flag = "ok" if right == want else f"{right}/{want}"
        print(f"  [{flag}] {truth} -> {pred}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weights", type=Path, default=None,
                    help="starting weights (default: latin_g2.pth). Pass lilly.pth "
                         "to continue a previous fine-tune.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--quick-test", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=5.0,
                    help="max gradient norm (default 5; pass-7 uses 1)")
    ap.add_argument("--warmup-frac", type=float, default=0.0,
                    help="fraction of steps for linear LR warmup (default 0; "
                         "pass-7b uses 0.05)")
    ap.add_argument("--keep-trained", type=Path, metavar="PATH",
                    help="always write the trained weights here, whatever the "
                         "install gate decides — so a refused run can still be "
                         "scored on the real photographs")
    args = ap.parse_args()

    train_dir, valid_dir = OCR_DATA / "train", OCR_DATA / "valid"
    if not (train_dir / "gt.txt").exists():
        print(f"no training images at {train_dir} — run "
              f"data/scripts/generate_ocr_data.py first", file=sys.stderr)
        return 1

    if args.quick_test:
        args.epochs, args.batch_size, args.limit = 1.0, 2, 8

    hold_lock()

    charset = load_charset()
    from easyocr.utils import CTCLabelConverter
    converter = CTCLabelConverter(charset, separator_list={}, dict_pathlist={})

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(converter.character))
    weights = args.weights or (READ_DIR / "latin_g2.pth")
    if weights.resolve() == (READ_DIR / "latin_g2.pth").resolve():
        ensure_pristine(weights)
    elif not weights.is_file():
        raise SystemExit(f"weights not found: {weights}")
    else:
        print(f"continuing from {weights} (not the published latin_g2 baseline)")
    load_weights(model, weights)
    model = model.to(device).train()
    print(f"device: {device}  |  vocabulary: {len(converter.character)} classes")

    train = Crops(train_dir, args.limit)
    # Full valid, not a 500-row cap: that cap, plus synthetic rows landing
    # first, is how a photograph-aimed run was judged on the easy half.
    valid = Crops(valid_dir, args.limit)
    valid_real = valid.subset(real=True)
    valid_syn = valid.subset(real=False)
    print(f"training crops: {len(train):,}  |  held out: {len(valid):,} "
          f"({len(valid_real)} real, {len(valid_syn)} synthetic)")
    if not train:
        print("nothing to train on", file=sys.stderr)
        return 1

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)

    def score(crops):
        if not crops:
            return None
        loader = DataLoader(crops, batch_size=args.batch_size, collate_fn=collate,
                            num_workers=0)
        return accuracy(model, loader, converter, device)

    before = score(valid)
    before_real = score(valid_real)
    before_syn = score(valid_syn)
    print(f"before pooled: {before[0]:.1f}% words, "
          f"{before[1]:.1f}% of {before[2]} Bosnian letters")
    if before_real:
        print(f"before real:   {before_real[0]:.1f}% words, "
              f"{before_real[1]:.1f}% of {before_real[2]} Bosnian letters "
              f"({len(valid_real)} crops)")
        print_diacritic_cases("before", diacritic_cases(
            model, valid_real, converter, device))
    if before_syn:
        print(f"before syn:    {before_syn[0]:.1f}% words, "
              f"{before_syn[1]:.1f}% of {before_syn[2]} Bosnian letters "
              f"({len(valid_syn)} crops)")

    criterion = nn.CTCLoss(zero_infinity=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps = int(len(train_loader) * args.epochs)
    warmup_steps = max(1, int(steps * args.warmup_frac))
    print(f"{steps:,} steps  (warmup {warmup_steps}, grad-clip {args.grad_clip})")

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        progress = (current_step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    step = 0
    t0 = time.time()
    stop = False
    while step < steps and not stop:
        for images, texts in train_loader:
            if step >= steps:
                break
            targets, lengths = converter.encode(texts)
            logits = model(images.to(device), None).log_softmax(2).permute(1, 0, 2)
            input_len = logits.size(0)
            input_lengths = torch.full(
                (logits.size(1),), input_len, dtype=torch.int32, device=device)
            lengths = lengths.to(device=device, dtype=torch.int32)
            loss = criterion(logits, targets.to(device), input_lengths, lengths)
            finite = math.isfinite(loss.item())
            if not finite:
                print(f"\nloss became non-finite at step {step} — stopping early")
                stop = True
                break
            optimiser.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                                  args.grad_clip)
            optimiser.step()
            scheduler.step()
            step += 1
            if step % 50 == 0 or step == steps:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  {step}/{steps}  loss {loss.item():.4f}  "
                      f"lr {cur_lr:.2e}  "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)

    after = score(valid)
    after_real = score(valid_real)
    after_syn = score(valid_syn)
    print(f"after pooled: {after[0]:.1f}% words, "
          f"{after[1]:.1f}% of {after[2]} Bosnian letters")
    if after_real:
        print(f"after real:   {after_real[0]:.1f}% words, "
              f"{after_real[1]:.1f}% of {after_real[2]} Bosnian letters")
        print_diacritic_cases("after", diacritic_cases(
            model, valid_real, converter, device))
    if after_syn:
        print(f"after syn:    {after_syn[0]:.1f}% words, "
              f"{after_syn[1]:.1f}% of {after_syn[2]} Bosnian letters")

    # The gate below decides whether to install. --keep-trained is written only
    # when the gate passes so a refused run cannot be mistaken for shippable
    # weights (see the install block below where it is written).

    if args.quick_test:
        print("\nquick test — the shipped reader is left alone")
        return 0

    # A run that stopped on a non-finite loss, or that came out unable to read at
    # all, has nothing to offer the gate below: its "after" number is a broken
    # model's, and comparing it is only a slower way of refusing it. The weights
    # were written above so the failure can be looked at, not installed.
    if stop or (before[0] > 50 and after[0] < 10):
        print("\ntraining collapsed — the weights above are for inspection only")
        return 1

    # A run this short cannot have learned anything, whatever the numbers say —
    # on a handful of steps the two measurements are noise against each other and
    # the comparison below will happily wave through a model trained for seconds.
    if steps < MIN_STEPS:
        print(f"\nonly {steps} steps — too short to mean anything")
        return 1

    # Pass-5 (Kaggle v10) trained to the end and was refused: pooled words
    # 88.8% -> 88.0% on a valid set that is mostly synthetic, which is already
    # at ceiling. The run was aimed at photographs. Judging it on the generator
    # is the same mistake as scoring 500 synthetic rows and calling that the
    # reader. Written before pass-6 produces a number:
    #
    #   real held-out crops: words strictly up; Bosnian letters not down
    #   synthetic held-out:  neither number down (a floor, not a target)
    #
    # Both halves, not either. A real-crop gain paid for with synthetic
    # collapse does not install. A synthetic tick with real crops unchanged
    # does not install either — that is pass-4's job, already done.
    if before_real is None or after_real is None or len(valid_real) < 40:
        print("\ntoo few real valid crops to decide a photograph-aimed run")
        return 1
    if before_syn is None or after_syn is None:
        print("\nno synthetic valid crops — the mix is missing, not replacing")
        return 1
    real_words_up = after_real[0] > before_real[0]
    real_letters_ok = after_real[1] >= before_real[1]
    syn_ok = after_syn[0] >= before_syn[0] and after_syn[1] >= before_syn[1]
    print(f"\nreal  words {before_real[0]:.1f}% -> {after_real[0]:.1f}%, "
          f"letters {before_real[1]:.1f}% -> {after_real[1]:.1f}%")
    print(f"syn   words {before_syn[0]:.1f}% -> {after_syn[0]:.1f}%, "
          f"letters {before_syn[1]:.1f}% -> {after_syn[1]:.1f}%")
    if not (real_words_up and real_letters_ok and syn_ok):
        if not real_words_up:
            print("real-crop words did not rise — not replacing the shipped reader")
        elif not real_letters_ok:
            print("real-crop words rose but Bosnian letters fell — a trade. "
                  "Not replacing it.")
        else:
            print("real crops improved but synthetic valid fell — not replacing")
        return 1

    keep = READ_DIR / "latin_g2-previous.pth"
    if not keep.exists():
        shutil.copy(weights, keep)
        print(f"kept the shipped reader at {keep}")
    # latin_g2.pth is deliberately NOT written. It is the file easyocr
    # published and checks the MD5 of, it is what LILLY_READER=stock loads for
    # the before-and-after comparison, and it is what ensure_pristine() restores
    # from at the start of every run. Overwriting it with a trained reader
    # silently turns the "before" build into another copy of the "after" one,
    # and a comparison against yourself always shows no change.
    #
    # The trained weights go where the app reads them, and nowhere else.
    #
    # app/ocr.py builds its reader with recog_network="lilly", so easyocr loads
    # models/lilly/read/lilly.pth through user_network/lilly.yaml. latin_g2.pth
    # is only the starting point: easyocr verifies its MD5 against the weights
    # it published and refuses to load a file that does not match, which is why
    # the trained reader ships as a network of its own in the first place.
    #
    # Nothing wrote lilly.pth. Training rewrote latin_g2.pth, said "the app
    # reads with this now", and the app went on reading the previous lilly.pth
    # untouched. A run that improved held-out crops from 62.2% to 85.4% changed
    # nothing a user would see, and scoring it on the photographs afterwards
    # would have measured the old reader and printed the result under the new
    # one's name. That is the same shape as every measurement mistake in this
    # project's history: the number was real, it was just of something else.
    #
    # The two files are the same architecture -- 44 tensors, matching shapes,
    # checked before this was written -- so installing is a copy.
    app_weights = READ_DIR / "lilly.pth"
    if app_weights.exists():
        previous = READ_DIR / "lilly-previous.pth"
        if not previous.exists():
            shutil.copy(app_weights, previous)
            print(f"kept the reader the app was using at {previous.name}")
    # Write --keep-trained only here, after the gate passed. Writing it earlier
    # (before the gate) left a refused run's weights sitting on disk looking
    # like shippable output.
    if args.keep_trained:
        args.keep_trained.parent.mkdir(parents=True, exist_ok=True)
        save_weights(model.to("cpu"), args.keep_trained)
        model = model.to(device)
        print(f"trained weights written to {args.keep_trained}")

    # Install the trained network, not the file we started from.
    # Copying the init checkpoint into lilly.pth shipped latin_g2 as the app
    # reader (md5 46986913) after Kaggle pass-1 had actually trained.
    if args.keep_trained and args.keep_trained.is_file():
        shutil.copy(args.keep_trained, app_weights)
    else:
        save_weights(model.cpu(), app_weights)
        model = model.to(device)
    print(f"wrote {app_weights} — the app reads with this now")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
