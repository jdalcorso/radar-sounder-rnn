#!/usr/bin/env python3
"""
Test script.
Testing on MCORDS1 Dataset with ~100k rangelines, 410 samples

@author: Jordy Dal Corso
"""
import os
import glob
import logging
import scripting
import torch

from torch.cuda import device_count
from torch.nn import DataParallel
from sklearn.metrics import classification_report, confusion_matrix

from utils import (
    get_model,
    get_dataloaders,
    load_best,
)

hooked_outputs = []


def main(
    model,
    hidden_size,
    patch_len,
    seq_len,
    split,
    first_only,
    seed,
    batch_size,
    return_dict,
    dataset,
    out_dir,
    **kwargs,
):
    logger = logging.getLogger("train")

    # Dataset
    _, _, dl, patch_h, _, n_classes = get_dataloaders(
        dataset, seq_len, patch_len, batch_size, split, first_only, False, logger, seed
    )

    # Model
    model_name = model
    in_channels = 1
    cfg = [in_channels, hidden_size, n_classes, (patch_h, patch_len)]
    model = get_model(model, cfg)
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
    logger.info(f"Total number of learnable parameters: {model.module.nparams}")

    # Test
    model.train(False)
    labels = []
    preds = []
    with torch.no_grad():
        for _, item in enumerate(dl):
            seq = item[0].to("cuda").unsqueeze(2)  # BTHW -> BTCHW
            labels.append(item[1].long().flatten(0, 1))  # (BT)HW
            preds.append(model(seq).squeeze(2).argmax(2).flatten(0, 1))  # (BT)HW

    labels = torch.cat(labels, dim=0).permute(1, 0, 2).reshape(seq.shape[3], -1)
    preds = torch.cat(preds, dim=0).permute(1, 0, 2).reshape(seq.shape[3], -1)
    torch.save(preds.byte(), out_dir + "/pred.pt")

    logger.info("Classification report:\n")
    report = classification_report(
        labels.flatten(), preds.flatten().cpu(), output_dict=return_dict
    )
    logger.info(report)
    logger.info("Confusion matrix:\n")
    logger.info(confusion_matrix(labels.flatten(), preds.flatten().cpu()))

    # Delete all files in the output folder that start with "epoch"
    torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
    for file in glob.glob(os.path.join(out_dir, "epoch*")):
        os.remove(file)


def hook_fn(module, input, output):
    hooked_outputs.append(output[0].detach().cpu())


if __name__ == "__main__":
    scripting.logged_main(
        "Test",
        main,
    )
