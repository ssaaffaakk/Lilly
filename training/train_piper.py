#!/usr/bin/env python3
"""piper.train's fit, with a checkpoint callback that needs no MOS predictor.

piper.train keeps its best checkpoints by val_mel and by val_mos. With the MOS
predictor switched off (it would download a model from the network on the
box), val_mos is never logged, and Lightning 2.x raises a
MisconfigurationException at the first validation end instead of the warning
piper.train's comment promises: version 2 of speak-bs died there, at epoch 5,
with 1,865 steps behind it and nothing saved but the trainer's own last.ckpt.

The pre-registration ("v5 -- speak") exports the LAST checkpoint and picks
nothing by loss, so last.ckpt is the only checkpoint the run needs. This is
piper.train's own CLI, model and data module -- same arguments, same defaults
-- with that one callback. Validation still runs and val_* still lands in
metrics.csv for the record.

The exit code is the fit's, not the interpreter teardown's. On the Mac the
process finished its fit, wrote last.ckpt and metrics.csv, and then died with
a segmentation fault while Python unloaded torch and the Cython alignment
module -- exit 139 for a training that had succeeded. The notebook's run()
turns any non-zero exit into a stop, as it must, so a fit that returned
normally ends the process with os._exit(0) after flushing; a fit that raised
still leaves through the exception, with its traceback and a non-zero exit.

    python3 training/train_piper.py fit --data.csv_path ... (exactly piper.train's arguments)
"""
import logging
import os
import sys

import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from piper.train.__main__ import VitsLightningCLI
from piper.train.vits.dataset import VitsDataModule
from piper.train.vits.lightning import VitsModel


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    VitsLightningCLI(  # noqa: F841
        VitsModel, VitsDataModule,
        trainer_defaults={"max_epochs": -1,
                          "callbacks": [ModelCheckpoint(save_last=True, save_top_k=0,
                                                        every_n_epochs=1)]},
    )
    # The fit returned. Leave before the native teardown can turn that into 139.
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
