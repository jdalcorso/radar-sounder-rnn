#!/usr/bin/env python3
"""
Test script.
Testing on MCORDS1 Dataset with ~100k rangelines, 410 samples

@author: Jordy Dal Corso
"""
import logging
import scripting
import torch

from torch.cuda import device_count
from torch.nn import DataParallel
from sklearn.metrics import classification_report, confusion_matrix

from model import NLURNNCell
from utils import get_dataloaders, load_best
import os
import glob
import json


def main(
    model,
    hidden_size,
    patch_len,
    seq_len,
    split,
    seed,
    batch_size,
    dataset,
    out_dir,
    **kwargs,
):
    logger = logging.getLogger("train")

    # Dataset
    first_only = False
    _, _, dl, patch_h, _, n_classes = get_dataloaders(
        dataset, seq_len, patch_len, batch_size, split, first_only, False, logger, seed
    )

    # Model
    in_channels = 1
    cfg = [in_channels, hidden_size, n_classes, (patch_h, patch_len)]
    model = NLURNNCell(*cfg)
    num_devices = device_count()
    if num_devices >= 2:
        model = DataParallel(model)
    model = model.to("cuda")
    try:
        model, best_loss = load_best(model, out_dir)
        logger.info("Loaded {}".format(best_loss))
    except:
        model.load_state_dict(torch.load(out_dir + "/best.pt"))
        logger.info("Loaded best.pt")
    nparams = model.module.nparams if num_devices >= 2 else model.nparams
    logger.info(f"Total number of learnable parameters: {nparams}")

    # Test
    model.train(False)
    labels = []
    preds = []
    with torch.no_grad():
        for _, item in enumerate(dl):
            seq = item[0].to("cuda").unsqueeze(2)  # BTHW -> BT1HW
            labels.append(item[1].long().flatten(0, 1))  # (BT)HW
            this_preds, hidden, cell = [], None, None
            for i in range(seq_len):
                pred, hidden, cell = model(seq[:, i], hidden, cell)  # BCHW
                this_preds.append(pred.unsqueeze(1))  # B1CHW * T
            this_preds = torch.cat(this_preds, dim=1)  # BTCHW
            preds.append(this_preds.argmax(2).flatten(0, 1))  # (BT)HW

    labels = torch.cat(labels, dim=0).permute(1, 0, 2).reshape(seq.shape[3], -1)
    preds = torch.cat(preds, dim=0).permute(1, 0, 2).reshape(seq.shape[3], -1)

    # Set to free-space predictions above the surface (trivial)
    if dataset == "mcords1":
        for col in range(preds.shape[1]):
            last_zero_idx = (preds[:, col] == 0).nonzero(as_tuple=True)[0][-1]
            preds[:last_zero_idx, col] = 0

    torch.save(preds.byte(), out_dir + "/pred.pt")
    labels = labels.flatten()
    preds = preds.flatten().cpu()

    # Remove class 5 as per Garcia et al. 2023
    if dataset == "mcords3":
        mask = labels != 5
        labels = labels[mask]
        preds = preds[mask]

    logger.info("Classification report:\n")
    report = classification_report(labels, preds, output_dict=True)
    torch.save(report, out_dir + "/report_seed_{}.pt".format(seed))
    report_str = json.dumps(report, indent=4)
    logger.info(report_str)
    logger.info("Confusion matrix:\n")
    logger.info(confusion_matrix(labels, preds))

    # Delete all files in the output folder that start with "epoch"
    torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
    for file in glob.glob(os.path.join(out_dir, "epoch*")):
        os.remove(file)


if __name__ == "__main__":
    scripting.logged_main(
        "Test",
        main,
    )
