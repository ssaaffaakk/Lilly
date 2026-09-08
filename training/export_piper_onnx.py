#!/usr/bin/env python3
"""Export a Piper checkpoint to the ONNX the app loads -- with the exporter that can.

piper.train.export_onnx calls torch.onnx.export without naming an exporter, and
since torch 2.9 that means the dynamo one, which refuses VITS's inference graph:
the duration predictor makes the output length data-dependent, and the export
dies in torch.export with a data-dependent error (seen on the Mac, torch 2.14).
The TorchScript exporter traces it, as Piper's own published voices were
exported. Everything else here is Piper 1.8.0's export, line for line: the same
inference wrapper, opset 15, the same input names and dynamic axes, so the file
is one the app's PiperVoice loads like any other.

    python3 training/export_piper_onnx.py --checkpoint last.ckpt --output-file voice.onnx
"""
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch

OPSET_VERSION = 15
_LOGGER = logging.getLogger(__name__)


def export(checkpoint: Path, output: Path) -> Path:
    from piper.train.vits.lightning import VitsModel
    torch.manual_seed(1234)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = VitsModel.load_from_checkpoint(checkpoint, map_location="cpu")
    model_g = model.model_g
    model_g.eval()
    with torch.no_grad():
        model_g.dec.remove_weight_norm()

    def infer_forward(text, text_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio = model_g.infer(text, text_lengths, noise_scale=noise_scale,
                              length_scale=length_scale, noise_scale_w=noise_scale_w,
                              sid=sid)[0].unsqueeze(1)
        return audio

    model_g.forward = infer_forward  # type: ignore[method-assign,assignment]
    num_symbols = model_g.n_vocab
    num_speakers = model_g.n_speakers
    dummy_input_length = 50
    sequences = torch.randint(low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long)
    sequence_lengths = torch.LongTensor([sequences.size(1)])
    sid: Optional[torch.LongTensor] = None
    if num_speakers > 1:
        sid = torch.LongTensor([0])
    scales = torch.FloatTensor([0.667, 1.0, 0.8])   # noise, length, noise_w
    dummy_input = (sequences, sequence_lengths, scales, sid)
    torch.onnx.export(
        model=model_g, args=dummy_input, f=str(output), verbose=False,
        opset_version=OPSET_VERSION,
        input_names=["input", "input_lengths", "scales", "sid"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size", 1: "phonemes"},
                      "input_lengths": {0: "batch_size"},
                      "output": {0: "batch_size", 2: "time"}},
        dynamo=False,
    )
    if not output.is_file() or output.stat().st_size < 1_000_000:
        raise SystemExit(f"export wrote nothing usable at {output}")
    _LOGGER.info("Exported %s -> %s (%d speakers)", checkpoint, output, num_speakers)
    return output


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output-file", required=True, type=Path)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    export(args.checkpoint, args.output_file)
    print(args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
