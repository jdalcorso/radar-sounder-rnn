#!/usr/bin/env python3
"""
Script to plot qualitative results for a weakly-supervised model.
Testing on 2 MCORDS-1 radargrams chosen in the config file.
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

from model import NLURNNCell
from utils import pos_encode, get_model, get_dataloaders


def main(
    hidden_size,
    n_classes,
    pos_enc,
    patch_len,
    seq_len,
    seed,
    batch_size,
    font_size,
    aspect,
    data_dir,
    out_dir,
    **kwargs,
):
    logger = logging.getLogger("train")

    # Dataset
    _, dl = get_dataloaders(data_dir, seq_len, patch_len, batch_size, 0.5, seed)
    _, _, patch_h, _ = next(iter(dl))[0].shape

    preds = []

    # Weakly-supervised
    for model in ["cdouble", "wcmod"]:
        # Model
        model_name = model
        in_channels = 2 if pos_enc else 1
        cfg = [in_channels, hidden_size, n_classes, (patch_h, patch_len)]
        model = NLURNNCell(*cfg)
        num_devices = device_count()
        if num_devices >= 2:
            model = DataParallel(model)
        model = model.to("cuda")
        model.load_state_dict(torch.load(out_dir + "/latest_" + model_name + ".pt"))

        # Test
        model.train(False)
        with torch.no_grad():
            item = next(iter(dl))
            seq = item[0].to("cuda").unsqueeze(2)  # BTHW -> BT1HW
            seq = pos_encode(seq) if pos_enc else seq
            this_preds, hidden, cell = [], None, None
            for i in range(seq_len):
                pred, hidden, cell = model(seq[:, i], hidden, cell)  # BCHW
                this_preds.append(pred.unsqueeze(1))  # B1CHW * T
            this_preds = torch.cat(this_preds, dim=1)  # BTCHW
            preds.append(this_preds.argmax(2))  # (BT)HW

    # Fully-supervised
    # Model
    model_name = "aspp"
    in_channels = 2 if pos_enc else 1
    cfg = [in_channels, hidden_size, n_classes, (patch_h, patch_len)]
    model = get_model(model_name, cfg)
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

    fig, ax = plt.subplots(5, batch_size, figsize=(15, 15))
    fs = font_size  # font size
    times_y = [0, 20, 40, 60]
    values_y = [0, 133, 266, 399]
    traces_x = [0, 1000, 2000, 3000]

    for i in range(batch_size):
        # Radargram
        ax[0, i].imshow(rgrams[i], cmap="gray", aspect=aspect)
        ax[0, i].set_yticks(values_y, times_y, fontsize=fs)
        ax[0, i].set_xticks(traces_x, traces_x, fontsize=fs)

        # True
        ax[1, i].imshow(labels[i], cmap=cmap, interpolation="nearest", aspect=aspect)
        ax[1, i].set_yticks(values_y, times_y, fontsize=fs)
        ax[1, i].set_xticks(traces_x, traces_x, fontsize=fs)

        # Pred-1
        ax[2, i].imshow(preds[0][i], cmap=cmap, interpolation="nearest", aspect=aspect)
        ax[2, i].set_yticks(values_y, times_y, fontsize=fs)
        ax[2, i].set_xticks(traces_x, traces_x, fontsize=fs)

        # Pred-2
        ax[3, i].imshow(preds[1][i], cmap=cmap, interpolation="nearest", aspect=aspect)
        ax[3, i].set_yticks(values_y, times_y, fontsize=fs)
        ax[3, i].set_xticks(traces_x, traces_x, fontsize=fs)

        # Pred-3
        ax[4, i].imshow(preds[2][i], cmap=cmap, interpolation="nearest", aspect=aspect)
        ax[4, i].set_yticks(values_y, times_y, fontsize=fs)
        ax[4, i].set_xticks(traces_x, traces_x, fontsize=fs)
        ax[4, i].set_xlabel("Trace", fontsize=fs)

        # Retain time axis on 1st sample
        if i == 0:
            ax[0, i].set_ylabel("Time [μs]", fontsize=fs)
            ax[1, i].set_ylabel("Time [μs]", fontsize=fs)
            ax[2, i].set_ylabel("Time [μs]", fontsize=fs)
            ax[3, i].set_ylabel("Time [μs]", fontsize=fs)
            ax[4, i].set_ylabel("Time [μs]", fontsize=fs)

        # Add letter
        if i == batch_size // 2:
            for j in range(5):
                offset = -0.3 if j == 4 else -0.13
                ax[j, i].text(
                    0.5,
                    offset,
                    f"({chr(ord('a') + j)})",
                    transform=ax[j, i].transAxes,
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

    # Add legend
    img = imread(out_dir + "/legend.png")
    legend_ax = fig.add_axes([0.27, -0.25, 0.55, 0.55])  # [x, y, width, height]
    legend_ax.imshow(img)
    legend_ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.09)

    plt.savefig(out_dir + "/qual_weakly.pdf", dpi=300)
    plt.close()


if __name__ == "__main__":
    scripting.logged_main(
        "Test",
        main,
    )
