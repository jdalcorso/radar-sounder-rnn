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
from utils import pos_encode, get_hooks, show_feature_maps, get_dataloaders, load_best
import os
import glob

hooked_outputs = []


def main(
    model,
    hidden_size,
    pos_enc,
    patch_len,
    seq_len,
    split,
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
        dataset, seq_len, patch_len, batch_size, split, logger, seed
    )

    # Model
    model_name = model
    in_channels = 2 if pos_enc else 1
    cfg = [in_channels, hidden_size, n_classes, (patch_h, patch_len)]
    model = NLURNNCell(*cfg)
    num_devices = device_count()
    if num_devices >= 2:
        model = DataParallel(model)
    model = model.to("cuda")
    try:
        model, best_loss = load_best(model, out_dir)
        logger.info("Loaded epoch_{}.pt".format(best_loss))
    except:
        model.load_state_dict(torch.load(out_dir + "/best.pt"))
        logger.info("Loaded best.pt")
    logger.info(f"Total number of learnable parameters: {model.module.nparams}")

    # Hooks
    hooks = get_hooks(model_name, model, hook_fn)

    # Test
    model.train(False)
    labels = []
    preds = []
    with torch.no_grad():
        for _, item in enumerate(dl):
            seq = item[0].to("cuda").unsqueeze(2)  # BTHW -> BT1HW
            seq = pos_encode(seq) if pos_enc else seq
            labels.append(item[1].long().flatten(0, 1))  # (BT)HW
            this_preds, hidden, cell = [], None, None
            for i in range(seq_len):
                pred, hidden, cell = model(seq[:, i], hidden, cell)  # BCHW
                this_preds.append(pred.unsqueeze(1))  # B1CHW * T
            this_preds = torch.cat(this_preds, dim=1)  # BTCHW
            preds.append(this_preds.argmax(2).flatten(0, 1))  # (BT)HW

    labels = torch.cat(labels, dim=0).permute(1, 0, 2).reshape(seq.shape[3], -1)
    preds = torch.cat(preds, dim=0).permute(1, 0, 2).reshape(seq.shape[3], -1)
    torch.save(preds.byte(), out_dir + "/pred.pt")

    show_feature_maps(hooked_outputs[: 2 * len(hooks)], out_dir)
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
