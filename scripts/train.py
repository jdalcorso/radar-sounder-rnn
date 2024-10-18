#!/usr/bin/env python3
"""
Training script.
The task is plain supervised semantic segmentation. We plan to use:
- CNN autoencoder
- U-Net autoencoder
- Youtube-VOS-style ConvLSTM autoencoder
- The same as above with U-net residial bridges
Input (and final output) channels are automatically set to be 1 as we are working
on 1 channel radar sounder data.

@author: Jordy Dal Corso
"""
import time
import logging
import scripting
import torch

from torch.cuda import device_count
from torch.nn import DataParallel
from torch.nn.functional import cross_entropy
from torch.optim import AdamW

from utils import (
    plot_loss,
    get_dataloaders,
    plot_results,
    pos_encode,
    get_model,
    set_seed,
)

set_seed(42)


def main(
    model,
    hidden_size,
    n_classes,
    pos_enc,
    patch_len,
    seq_len,
    test_size,
    epochs,
    batch_size,
    lr,
    data_dir,
    out_dir,
    **kwargs,
):
    logger = logging.getLogger("train")

    # Dataset
    train_dl, test_dl = get_dataloaders(
        data_dir, seq_len, patch_len, batch_size, test_size
    )
    _, _, patch_h, _ = next(iter(train_dl))[0].shape
    logger.info("Number of sequences TRAIN: {}".format(batch_size * len(train_dl)))
    logger.info("Number of sequences TEST : {}".format(batch_size * len(test_dl)))
    logger.info(
        "Shape of dataloader items: {}\n".format(list(next(iter(train_dl))[0].shape))
    )

    # Model
    in_channels = 2 if pos_enc else 1
    cfg = [in_channels, hidden_size, n_classes, (patch_h, patch_len)]
    model = get_model(model, cfg)
    num_devices = device_count()
    if num_devices >= 2:
        model = DataParallel(model)
    model = model.to("cuda")
    logger.info(f"Total number of learnable parameters: {model.module.nparams}")

    # # Optimizer
    optimizer = AdamW(model.parameters(), lr)

    # Train and validation
    loss_train_tot = []
    loss_val_tot = []
    for epoch in range(epochs):
        t0 = time.time()
        loss_train = []

        # Train
        model.train(True)
        for _, item in enumerate(train_dl):
            seq = item[0].to("cuda").unsqueeze(2)  # BTHW -> BTCHW
            seq = pos_encode(seq) if pos_enc else seq
            label = item[1].to("cuda").long()  # BTHW
            pred = model(seq).squeeze(2)  # BTCHW
            loss = cross_entropy(
                pred.flatten(0, 1),
                label.flatten(0, 1),
                weight=torch.tensor([0.36, 0.04, 0.54, 0.06]).to("cuda"),
            )  # weight=torch.tensor([0.04,0.2,0.18,0.54,0.04]).to('cuda') [0.36,0.04,0.54,0.06]
            loss_train.append(loss)
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        plot_results(
            seq[0],
            label[0],
            pred[0].argmax(1),
            out_dir + "/train" + str(epoch + 1) + ".png",
        )

        # Validation
        loss_val = []
        model.train(False)
        for _, item in enumerate(test_dl):
            seq = item[0].to("cuda").unsqueeze(2)  # Adding channel dimension
            seq = pos_encode(seq) if pos_enc else seq
            label = item[1].to("cuda").long()
            pred = model(seq)
            loss_t = cross_entropy(pred.flatten(0, 1), label.flatten(0, 1))
            loss_val.append(loss_t)
        plot_results(
            seq[0],
            label[0],
            pred[0].argmax(1),
            out_dir + "/val" + str(epoch + 1) + ".png",
        )

        loss_train, loss_val = (
            torch.tensor(loss_train).mean(),
            torch.tensor(loss_val).mean(),
        )
        loss_train_tot.append(loss_train)
        loss_val_tot.append(loss_val)

        logger_str = "Epoch: {}, Loss train: {:.3f}, Loss val: {:.3f}, Time: {:.3f}"
        logger.info(
            logger_str.format(
                epoch + 1, loss_train.item(), loss_val.item(), time.time() - t0
            )
        )

    plot_loss(loss_train_tot, loss_val_tot, out_dir)
    torch.save(model.state_dict(), out_dir + "/latest.pt")


if __name__ == "__main__":
    scripting.logged_main(
        "Training",
        main,
    )
