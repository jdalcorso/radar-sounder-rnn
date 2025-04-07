#!/usr/bin/env python3
"""
CC + Modified CC-based training script.
Cycles are constructed always starting from the same column
with sequences of increasing length as per the original CRW
paper: https://arxiv.org/abs/2006.14613
At each timestep during the processing of the flipped
sequence the standard CC loss (as cycle.py) is computed.

@author: Jordy Dal Corso
"""
import random
import logging
import scripting
import torch

from torch.cuda import device_count
from torch.cuda.amp import autocast, GradScaler
from torch.nn import DataParallel
from torch.nn.functional import cross_entropy, softmax
from torch.optim import AdamW

from model import NLURNNCell
from utils import (
    plot_loss,
    get_dataloaders,
    plot_results,
    set_seed,
    validation_weak,
    save_latest,
)

set_seed(42)


def main(
    hidden_size,
    patch_len,
    seq_len,
    split,
    seed,
    epochs,
    batch_size,
    batch_number,
    lr,
    wd,
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
    nparams = model.module.nparams if num_devices >= 2 else model.nparams
    logger.info(f"Total number of learnable parameters: {nparams}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr, weight_decay=wd)
    scaler = GradScaler()

    # Batches
    num_batches = len(train_dl)  # Get total number of batches
    selected_indices = random.sample(range(num_batches), batch_number)
    logger.info(f"Selected {len(selected_indices)*batch_size} samples for training.")

    # Train
    loss_train_tot = []
    loss_val_tot = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for epoch in range(epochs):
        # Train
        model.train(True)
        start_event.record()
        with autocast():
            seq, label, preds, loss_train = train(
                model,
                optimizer,
                scaler,
                train_dl,
                selected_indices,
                seq_len,
                ce_weights,
            )
        end_event.record()
        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            plot_results(
                seq[0, :seq_len],
                label[0],
                preds[0, :seq_len].argmax(1),  # THW
                out_dir + "/train" + str(epoch + 1) + ".png",
            )

        # Validation
        with autocast():
            seq, label, this_preds, loss_val = validation_weak(
                model, val_dl, seq_len, ce_weights
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
            logger_str.format(
                epoch + 1,
                loss_train.item(),
                loss_val.item(),
                start_event.elapsed_time(end_event),
            )
        )

    torch.save(model.state_dict(), out_dir + "/latest.pt")
    return torch.min(torch.tensor(loss_val_tot))


@torch.compile
def train(model, optimizer, scaler, dataloader, selected_indices, seq_len, ce_weights):
    loss_train = []
    ce_weights = torch.tensor(ce_weights).to("cuda")
    for n, item in enumerate(dataloader):
        if n not in selected_indices:
            continue
        seq = item[0].to("cuda").unsqueeze(2)  # BTHW -> BTCHW
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
                # First with reference
                if i == 0:
                    loss += cross_entropy(pred, label[:, 0], ce_weights)
                # Vanilla CC as in cycle.py
                if i >= sub_len and i != sub_len * 2 - 1:
                    loss += (1 / seq_len) * cross_entropy(
                        pred,
                        softmax(
                            torch.flip(
                                preds[2 * sub_len - i - 1].squeeze(1), dims=(-1,)
                            ),
                            dim=1,
                        ),
                        ce_weights,
                    )
                # Last with first
                if i == sub_len * 2 - 1:
                    loss += cross_entropy(
                        pred, torch.flip(label[:, 0], dims=(-1,)), ce_weights
                    )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        loss_train.append(loss)
    preds = torch.cat(preds, 1)  # BTCHW
    return seq, label, preds, loss_train


if __name__ == "__main__":
    scripting.logged_main(
        "Training",
        main,
    )
