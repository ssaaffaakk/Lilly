"""The recogniser architecture Lilly's reader was fine-tuned as.

easyocr imports this module by the name of the network and calls Model(...) with
whatever network_params the yaml beside it declares. The architecture is the one
the shipped latin_g2 weights use — a VGG feature extractor with a BiLSTM and CTC
— because the fine-tuning started from those weights and only moved them.

This file exists because a fine-tuned recogniser cannot be shipped by
overwriting latin_g2.pth. easyocr checks that file's MD5 against a table of the
weights it published, and our weights are deliberately different, so it refuses
to load them and reports a corrupt download. Registering as a network of our own
is the path easyocr provides for exactly this, and it leaves the integrity check
doing its job on the files it was written for.
"""
from easyocr.model.vgg_model import Model  # noqa: F401 - easyocr imports it by name
