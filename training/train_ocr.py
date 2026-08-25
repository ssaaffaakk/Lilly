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
    python3 training/evaluate_ocr.py                          # did it help

Writes models/lilly/read/latin_g2.pth, keeping the previous one beside it as
latin_g2-previous.pth so there is a way back and something to measure against.
"""
import argparse
import os
import shutil
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
OCR_DATA = REPO_ROOT / "data" / "ocr"
IMG_HEIGHT = 64          # what the shipped reader was trained at
MAX_WIDTH = 600
DIACRITICS = "čćđšžČĆĐŠŽ"


def load_charset() -> str:
    import easyocr.config as config
    return config.recognition_models["gen2"]["latin_g2"]["characters"]


def build_model(num_class: int):
    """EasyOCR's own network, so every pretrained weight lands where it belongs."""
    from easyocr.model.vgg_model import Model
    return Model(input_channel=1, output_channel=256, hidden_size=256,
                 num_class=num_class)


def load_weights(model, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    # saved from a DataParallel wrapper, so every key carries a module. prefix
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def save_weights(model, path: Path) -> None:
    """Back in the shape the app loads: keys prefixed, plain state dict."""
    torch.save({f"module.{k}": v for k, v in model.state_dict().items()}, path)


def prepare(image_path: Path) -> torch.Tensor:
    """Exactly what the app does to a crop before reading it — grayscale, 64 tall,
    scaled to [-1, 1], right-padded by repeating the last column."""
    from torchvision import transforms
    img = Image.open(image_path).convert("L")
    ratio = img.width / max(img.height, 1)
    width = min(max(int(IMG_HEIGHT * ratio), 8), MAX_WIDTH)
    img = img.resize((width, IMG_HEIGHT), Image.BICUBIC)
    tensor = transforms.ToTensor()(img).sub_(0.5).div_(0.5)
    padded = torch.zeros(1, IMG_HEIGHT, MAX_WIDTH)
    padded[:, :, :width] = tensor
    if width < MAX_WIDTH:
        padded[:, :, width:] = tensor[:, :, width - 1:width].expand(
            1, IMG_HEIGHT, MAX_WIDTH - width)
    return padded


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
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        name, text = self.rows[idx]
        return prepare(self.folder / name), text


def collate(batch):
    images = torch.stack([b[0] for b in batch])
    return images, [b[1] for b in batch]


def decode(logits, converter):
    # the converter indexes with numpy semantics, so it needs numpy — handing it a
    # torch tensor indexes with a tuple instead and returns arrays, not characters
    sizes = [logits.size(1)] * logits.size(0)
    _, indices = logits.max(2)
    return converter.decode_greedy(indices.view(-1).cpu().numpy(), sizes)


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
                for a, b in zip(truth, pred.ljust(len(truth))):
                    if a in DIACRITICS:
                        dia_total += 1
                        dia_right += a == b
    model.train()
    return (100 * exact / max(total, 1),
            100 * dia_right / max(dia_total, 1), dia_total)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--quick-test", action="store_true")
    args = ap.parse_args()

    train_dir, valid_dir = OCR_DATA / "train", OCR_DATA / "valid"
    if not (train_dir / "gt.txt").exists():
        print(f"no training images at {train_dir} — run "
              f"data/scripts/generate_ocr_data.py first", file=sys.stderr)
        return 1

    if args.quick_test:
        args.epochs, args.batch_size, args.limit = 1.0, 2, 8

    charset = load_charset()
    from easyocr.utils import CTCLabelConverter
    converter = CTCLabelConverter(charset, separator_list={}, dict_pathlist={})

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(converter.character))
    weights = READ_DIR / "latin_g2.pth"
    load_weights(model, weights)
    model = model.to(device).train()
    print(f"device: {device}  |  vocabulary: {len(converter.character)} classes")

    train = Crops(train_dir, args.limit)
    valid = Crops(valid_dir, min(args.limit or 500, 500))
    print(f"training crops: {len(train):,}  |  held out: {len(valid):,}")
    if not train:
        print("nothing to train on", file=sys.stderr)
        return 1

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    valid_loader = DataLoader(valid, batch_size=args.batch_size, collate_fn=collate,
                              num_workers=0)

    before = accuracy(model, valid_loader, converter, device)
    print(f"before: {before[0]:.1f}% words exact, "
          f"{before[1]:.1f}% of {before[2]} Bosnian letters right")

    criterion = nn.CTCLoss(zero_infinity=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps = int(len(train_loader) * args.epochs)
    print(f"{steps:,} steps")

    step = 0
    t0 = time.time()
    while step < steps:
        for images, texts in train_loader:
            if step >= steps:
                break
            targets, lengths = converter.encode(texts)
            logits = model(images.to(device), None).log_softmax(2).permute(1, 0, 2)
            input_lengths = torch.IntTensor([logits.size(0)] * logits.size(1))
            loss = criterion(logits, targets.to(device), input_lengths, lengths)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimiser.step()
            step += 1
            if step % 50 == 0 or step == steps:
                print(f"  {step}/{steps}  loss {loss.item():.4f}  "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)

    after = accuracy(model, valid_loader, converter, device)
    print(f"after:  {after[0]:.1f}% words exact, "
          f"{after[1]:.1f}% of {after[2]} Bosnian letters right")

    if args.quick_test:
        print("\nquick test — the shipped reader is left alone")
        return 0

    if after[1] < before[1]:
        print("\nthe Bosnian letters got worse — not replacing the shipped reader")
        return 1

    keep = READ_DIR / "latin_g2-previous.pth"
    if not keep.exists():
        shutil.copy(weights, keep)
        print(f"kept the shipped reader at {keep}")
    save_weights(model.to("cpu"), weights)
    print(f"wrote {weights} — the app reads with this now")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
