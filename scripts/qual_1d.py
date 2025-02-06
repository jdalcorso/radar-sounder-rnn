#!/usr/bin/env python3
"""
Script to plot qualitative results for a 1d model.
Testing on MCORDS-1 radargrams chosen in the config file.
Here batch size consists in the number of radargrams showed.
The model used is supposed to be saved as latest_<model>.pt.

@author: Jordy Dal Corso
"""
import logging
import scripting
import torch
import matplotlib.pyplot as plt

from torch.cuda import device_count
from torch.nn import DataParallel
from matplotlib.colors import ListedColormap
from matplotlib.image import imread

from utils import pos_encode, get_model, get_dataloaders


def main(
    hidden_size,
    n_classes,
    pos_enc,
    patch_len,
    seq_len,
    seed,
    split,
    batch_size,
    font_size,
    aspect,
    data_dir,
    out_dir,
    **kwargs,
):
    logger = logging.getLogger("train")

    # Dataset
    _, _, dl = get_dataloaders(data_dir, seq_len, patch_len, batch_size, split, seed)
    _, _, patch_h, _ = next(iter(dl))[0].shape

    preds = []
    for i, model in enumerate(["1d64"]):
        # Model
        model_name = model
        in_channels = 2 if pos_enc else 1
        cfg = [in_channels, hidden_size, n_classes, (patch_h, patch_len)]
        model = get_model("nlur", cfg)
        num_devices = device_count()
        if num_devices >= 2:
            model = DataParallel(model)
        model = model.to("cuda")
        model.load_state_dict(torch.load(out_dir + "/latest_" + model_name + ".pt"))

        # Test
        model.train(False)
        with torch.no_grad():
            item = next(iter(dl))
            seq = item[0].to("cuda").unsqueeze(2)  # BTHW -> BTCHW
            seq = pos_encode(seq) if pos_enc else seq
            if i == 0:
                rgrams = item[0]  # BTHW
                labels = item[1].long()  # BTHW
            preds.append(model(seq).squeeze(2).argmax(2))  # BTHW x 3

    rgrams = rgrams.permute([0, 2, 1, 3]).flatten(-2, -1).cpu()  # BHW
    labels = labels.permute([0, 2, 1, 3]).flatten(-2, -1).cpu()  # BHW
    preds = [pred.permute([0, 2, 1, 3]).flatten(-2, -1).cpu() for pred in preds]  # BHW

    colors = [
        (0, 0, 0, 1),  # black, free space
        (0.33, 0.33, 0.33, 1),  # grey, ice
        (1, 0, 0, 1),  # red, bedrock
        (1, 1, 1, 1),  # white, noise
    ]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(batch_size, 3, figsize=(15, 10))
    fs = font_size  # font size
    times_y = [0, 20, 40, 60]
    values_y = [0, 133, 266, 399]
    traces_x = [0, 1000, 2000]

    for i in range(batch_size):
        # Radargram
        ax[i, 0].imshow(rgrams[i], cmap="gray", aspect=aspect)
        ax[i, 0].set_yticks(values_y, times_y, fontsize=fs)
        ax[i, 0].set_xticks(traces_x, traces_x, fontsize=fs)

        # True
        ax[i, 1].imshow(labels[i], cmap=cmap, interpolation="nearest", aspect=aspect)
        ax[i, 1].set_yticks(values_y, times_y, fontsize=fs)
        ax[i, 1].set_xticks(traces_x, traces_x, fontsize=fs)

        # Pred-1
        ax[i, 2].imshow(preds[0][i], cmap=cmap, interpolation="nearest", aspect=aspect)
        ax[i, 2].set_yticks(values_y, times_y, fontsize=fs)
        ax[i, 2].set_xticks(traces_x, traces_x, fontsize=fs)

        # Retain time axis on 1st sample
        if i == 0:
            ax[0, i].set_ylabel("Time [μs]", fontsize=fs)
            ax[1, i].set_ylabel("Time [μs]", fontsize=fs)
            ax[2, i].set_ylabel("Time [μs]", fontsize=fs)
            ax[batch_size - 1, 0].set_xlabel("Trace", fontsize=fs)
            ax[batch_size - 1, 1].set_xlabel("Trace", fontsize=fs)
            ax[batch_size - 1, 2].set_xlabel("Trace", fontsize=fs)

        # Add letter
        if i == batch_size - 1:
            for j in range(3):
                offset = -0.3
                ax[i, j].text(
                    0.5,
                    offset,
                    f"({chr(ord('a') + j)})",
                    transform=ax[i, j].transAxes,
                    fontsize=fs,
                    ha="center",
                    va="center",
                    color="black",
                )

    # Add frame
    for a in ax.flatten():
        for spine in a.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1)

    plt.tight_layout()
    plt.savefig(out_dir + "/qual_1d.pdf", dpi=300)
    plt.close()


if __name__ == "__main__":
    scripting.logged_main(
        "Test",
        main,
    )
