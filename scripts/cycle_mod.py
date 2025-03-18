#!/usr/bin/env python3
"""
Modified Cycle-Consistency-based training script.
Cycles are constructed always starting from the same column
with sequences of increasing length as per the original CRW
paper: https://arxiv.org/abs/2006.14613

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
    validation_weak,
    save_latest,
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
    log_every,
    log_last,
    out_dir,
    **kwargs,
):
    logger = logging.getLogger("train")

    # Dataset
    first_only = False
    train_dl, val_dl, _, patch_h, ce_weights, n_classes = get_dataloaders(
        dataset, seq_len, patch_len, batch_size, split, first_only, logger, seed
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
    loss_val_tot = []
    for epoch in range(epochs):
        t0 = time.time()

        # Train
        model.train(True)
        t0 = time.time()
        seq, label, pred, loss_train = train(
            model, optimizer, train_dl, seq_len, ce_weights, pos_enc
        )
        t1 = time.time() - t0
        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            plot_results(
                seq[0, :seq_len],
                label[0],
                pred[0, :seq_len].argmax(1),  # THW
                out_dir + "/train" + str(epoch + 1) + ".png",
            )

        # Validation
        seq, label, this_preds, loss_val = validation_weak(
            model, val_dl, seq_len, ce_weights, pos_enc
        )
        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            plot_results(
                seq[0],
                label[0],
                this_preds[0].argmax(1),
                out_dir + "/val" + str(epoch + 1) + ".png",
            )

        loss_train, loss_val = (
            torch.tensor(loss_train).mean(),
            torch.tensor(loss_val).mean(),
        )
        loss_train_tot.append(loss_train)
        loss_val_tot.append(loss_val)
        loss_train_show = [(lt / loss_train_tot[0]).item() for lt in loss_train_tot]
        loss_val_show = [(lt / loss_val_tot[0]).item() for lt in loss_val_tot]
        plot_loss(loss_train_show, loss_val_show, out_dir)

        # Save
        if epochs - epoch <= log_last or epoch == epochs - 1:
            save_latest(model, out_dir, loss_val_tot)
        logger_str = "Epoch: {}, Loss train: {:.3f}, Loss val: {:.3f}, Time: {:.3f}"
        logger.info(
            logger_str.format(epoch + 1, loss_train.item(), loss_val.item(), t1)
        )

    torch.save(model.state_dict(), out_dir + "/latest.pt")


# @torch.compile
def train(model, optimizer, dataloader, seq_len, ce_weights, pos_enc):
    loss_train = []
    ce_weights = torch.tensor(ce_weights, device="cuda")
    for _, item in enumerate(dataloader):
        seq = item[0].to("cuda").unsqueeze(2)  # BTHW -> BTCHW
        seq = pos_encode(seq) if pos_enc else seq
        label = item[1].to("cuda").long()  # BTHW
        for sub_len in range(1, seq_len):  # to create each sub-sequence
            sub = torch.cat(
                [seq[:, :sub_len], torch.flip(seq[:, :sub_len], dims=(1, -1))],
                dim=1,
            )  # B(2T)CHW
            preds = []
            hidden, cell = None, None
            optimizer.zero_grad()
            loss = 0
            for i in range(sub_len * 2):  # iterate on the sub-sequence
                pred, hidden, cell = model(sub[:, i], hidden, cell)  # BCHW
                preds.append(pred.unsqueeze(1))  # B1CHW * T
                # First with reference (supervised loss)
                if i == 0:
                    loss += cross_entropy(pred, label[:, 0], ce_weights)
                # Second with flipped reference (cycle-consistency loss)
                if i == sub_len * 2 - 1:
                    loss += cross_entropy(
                        pred, torch.flip(label[:, 0], dims=(-1,)), ce_weights
                    )
            loss.backward()
            optimizer.step()
        loss_train.append(loss)
    preds = torch.cat(preds, 1)  # BTCHW
    return seq, label, preds, loss_train


if __name__ == "__main__":
    scripting.logged_main(
        "Training",
        main,
    )
