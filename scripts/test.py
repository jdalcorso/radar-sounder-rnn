#!/usr/bin/env python3
"""
Test script.
Testing on MCORDS1 Dataset with ~100k rangelines, 410 samples

@author: Jordy Dal Corso
"""
import time
import logging
import scripting
import torch

from torch.cuda import device_count
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from dataset import RadargramDataset
from sklearn.metrics import classification_report, confusion_matrix

from utils import (
    pos_encode,
    get_model,
    set_seed,
)

set_seed(42)


def main(
    model,
    hidden_size,
    hidden_scaling,
    kernel_size,
    n_layers,
    n_classes,
    pos_enc,
    patch_len,
    seq_len,
    batch_size,
    data_dir,
    out_dir,
    **kwargs,
):
    logger = logging.getLogger("train")

    # Model
    in_channels = 2 if pos_enc else 1
    cfg = [in_channels, hidden_size, n_classes, n_layers, hidden_scaling, kernel_size]
    model = get_model(model, cfg)
    num_devices = device_count()
    if num_devices >= 2:
        model = DataParallel(model)
    model = model.to("cuda")
    model.load_state_dict(torch.load(out_dir + "/latest.pt"))
    logger.info(f"Total number of learnable parameters: {model.module.nparams}")

    # Dataset
    ds = RadargramDataset(data_dir, seq_len, patch_len, patch_len, seq_len)
    dl = DataLoader(ds, batch_size, shuffle=False)

    # Test
    model.train(False)
    labels = []
    preds = []
    for _, item in enumerate(dl):
        seq = item[0].to("cuda").unsqueeze(2)  # BTHW -> BTCHW
        seq = pos_encode(seq) if pos_enc else seq
        labels.append(item[1].long().flatten(0, 1))  # (BT)HW
        preds.append(model(seq).squeeze(2).argmax(2).flatten(0, 1))  # (BT)HW

    labels = torch.cat(labels, dim=0).permute(1, 0, 2).reshape(seq.shape[3], -1)
    preds = torch.cat(preds, dim=0).permute(1, 0, 2).reshape(seq.shape[3], -1)
    torch.save(preds.byte(), out_dir + "/pred.pt")

    print("Classification report:\n")
    print(classification_report(labels.flatten(), preds.flatten().cpu()))
    print("Confusion matrix:\n")
    print(confusion_matrix(labels.flatten(), preds.flatten().cpu()))


if __name__ == "__main__":
    scripting.logged_main(
        "Test",
        main,
    )
