#!/usr/bin/env python3
"""
Cycle-Consistency-based training script.

@author: Jordy Dal Corso
"""
import time
import logging
import scripting
import torch

from torch.cuda import device_count
from torch.nn import DataParallel
from torch.nn.functional import cross_entropy, softmax
from torch.optim import AdamW

from model import NLURNNCell
from utils import (
    plot_loss,
    get_dataloaders,
    plot_results,
    pos_encode,
    set_seed,
)

set_seed(42)


def main(
    hidden_size,
    pos_enc,
    patch_len,
    seq_len,
    split,
    seed,
    epochs,
    batch_size,
    lr,
    dataset,
    out_dir,
    **kwargs,
):
    logger = logging.getLogger("train")

    # Dataset
    train_dl, val_dl, _, _, n_classes = get_dataloaders(
        dataset, seq_len, patch_len, batch_size, split, seed
    )
    _, _, patch_h, _ = next(iter(train_dl))[0].shape
    logger.info("Number of sequences TRAIN: {}".format(batch_size * len(train_dl)))
    logger.info("Number of sequences TEST : {}".format(batch_size * len(val_dl)))
    logger.info(
        "Shape of dataloader items: {}\n".format(list(next(iter(train_dl))[0].shape))
    )

    # Model
    in_channels = 2 if pos_enc else 1
    cfg = [in_channels, hidden_size, n_classes, (patch_h, patch_len)]
    model = NLURNNCell(*cfg)
    num_devices = device_count()
    if num_devices >= 2:
        model = DataParallel(model)
    model = model.to("cuda")
    logger.info(f"Total number of learnable parameters: {model.module.nparams}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr)

    # Train
    loss_train_tot = []
    for epoch in range(epochs):
        t0 = time.time()
        loss_train = []

        # Train
        model.train(True)
        class_weights = torch.tensor([0.16, 0.04, 0.74, 0.06]).to("cuda")
        for _, item in enumerate(train_dl):
            seq = item[0].to("cuda").unsqueeze(2)  # BTHW -> BTCHW
            seq = pos_encode(seq) if pos_enc else seq
            seq = torch.cat([seq, torch.flip(seq, dims=(1, -1))], dim=1)  # B(2T)CHW
            label = item[1].to("cuda").long()  # BTHW
            preds = []
            hidden, cell = None, None
            optimizer.zero_grad()
            loss = 0
            for i in range(seq_len * 2):
                pred, hidden, cell = model(seq[:, i], hidden, cell)  # BCHW
                preds.append(pred.unsqueeze(1))  # B1CHW * T
                # preds has len = seq_len when going in 2nd if
                # if i == 0:
                #    loss += cross_entropy(pred, label[:, 0], class_weights)
                if i >= seq_len and i != seq_len * 2 - 1:
                    loss += ((seq_len * 2) * 0.1 - i * 0.1 + 0.1) * cross_entropy(
                        pred,
                        softmax(
                            torch.flip(
                                preds[2 * seq_len - i - 1].squeeze(1), dims=(-1,)
                            ),
                            dim=1,
                        ),
                        class_weights,
                    )
                if i == seq_len * 2 - 1:
                    loss += cross_entropy(
                        pred, torch.flip(label[:, 0], dims=(-1,)), class_weights
                    )

            loss_train.append(loss)
            loss.backward()
            optimizer.step()

        preds = torch.cat(preds, 1)  # BTCHW
        plot_results(
            seq[0, :seq_len],
            label[0],
            preds[0, :seq_len].argmax(1),  # THW
            out_dir + "/train" + str(epoch + 1) + ".png",
        )

        loss_train = torch.tensor(loss_train).mean()
        loss_train_tot.append(loss_train)

        logger_str = "Epoch: {}, Loss train: {:.3f}, Time: {:.3f}"
        logger.info(logger_str.format(epoch + 1, loss_train.item(), time.time() - t0))

    plot_loss(loss_train_tot, loss_train_tot, out_dir)
    torch.save(model.state_dict(), out_dir + "/latest.pt")


if __name__ == "__main__":
    scripting.logged_main(
        "Training",
        main,
    )
