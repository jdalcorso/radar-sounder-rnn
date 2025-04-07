#!/usr/bin/env python3
"""
Training script.
The task is plain supervised semantic segmentation. We plan to use:
- CNN autoencoder
- U-Net autoencoder
- Youtube-VOS-style ConvLSTM autoencoder
- The same as above with U-net residual bridges
Input (and final output) channels are automatically set to be 1 as we are working
on 1 channel radar sounder data.

@author: Jordy Dal Corso
"""
import random
import logging
import scripting
import torch

from torch.cuda import device_count
from torch.cuda.amp import autocast, GradScaler
from torch.nn import DataParallel
from torch.nn.functional import cross_entropy
from torch.optim import AdamW

from utils import (
    plot_loss,
    get_dataloaders,
    plot_results,
    get_model,
    save_latest,
)


def main(
    model,
    hidden_size,
    patch_len,
    seq_len,
    chunk_len,
    split,
    first_only,
    data_aug,
    seed,
    epochs,
    batch_size,
    batch_number,
    lr,
    wd,
    log_every,
    log_last,
    dataset,
    out_dir,
    **kwargs,
):
    logger = logging.getLogger("train")

    # Dataset
    train_dl, val_dl, _, patch_h, ce_weights, n_classes = get_dataloaders(
        dataset,
        seq_len,
        patch_len,
        batch_size,
        split,
        first_only,
        data_aug,
        logger,
        seed,
    )

    # Model
    in_channels = 1
    cfg = [in_channels, hidden_size, n_classes, (patch_h, patch_len)]
    model = get_model(model, cfg)
    num_devices = device_count()
    if num_devices >= 2:
        model = DataParallel(model)
    model = model.to("cuda")
    logger.info(f"Total number of learnable parameters: {model.module.nparams}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr, weight_decay=wd)
    scaler = GradScaler()

    # Batches
    num_batches = len(train_dl)  # Get total number of batches
    batch_number = num_batches if batch_number < 1 else batch_number
    selected_indices = random.sample(range(num_batches), batch_number)
    logger.info(f"Selected {len(selected_indices)*batch_size} samples for training.")

    # Train and validation
    loss_train_tot = []
    loss_val_tot = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for epoch in range(epochs):
        # Train
        model.train(True)
        start_event.record()
        with autocast():
            seq, label, pred, loss_train = train(
                model, optimizer, scaler, train_dl, seq_len, chunk_len, ce_weights
            )
        end_event.record()
        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            plot_results(
                seq[0],
                label[0],
                pred[0].argmax(1),
                out_dir + "/train" + str(epoch + 1) + ".png",
            )

        # Validation
        with autocast():
            seq, label, pred, loss_val = validation(model, val_dl, ce_weights)
        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
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
        plot_loss(loss_train_tot, loss_val_tot, out_dir)

        # Save
        if epochs - epoch <= log_last or epoch == epochs - 1:
            save_latest(model, out_dir, loss_val_tot)
        logger_str = "Epoch: {}, Loss train: {:.3f}, Loss val: {:.3f}, Time: {:.3f}ms"
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


# @torch.compile
def train(model, optimizer, scaler, dataloader, seq_len, chunk_len, ce_weights):
    loss_train = []
    num_chunks = seq_len // chunk_len
    for _, item in enumerate(dataloader):
        seq = item[0].to("cuda").unsqueeze(2)  # BTHW -> BTCHW
        label = item[1].to("cuda").long()  # BTHW
        pred = []
        for i in range(num_chunks):
            if num_chunks == 1:
                seq_chunk = seq
                label_chunk = label
            else:
                start = i * chunk_len
                end = (i + 1) * chunk_len
                seq_chunk = seq[:, start:end]
                label_chunk = label[:, start:end]
            pred_chunk = model(seq_chunk).squeeze(2)  # BTCHW
            pred.append(pred_chunk)
            loss = cross_entropy(
                pred_chunk.flatten(0, 1),
                label_chunk.flatten(0, 1),
                weight=torch.tensor(ce_weights, device="cuda"),
            )
            loss_train.append(loss)
            # Optimize
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        pred = torch.cat(pred, dim=1)
        return seq, label, pred, loss_train


# @torch.compile
def validation(model, dataloader, ce_weights):
    loss_val = []
    model.train(False)
    with torch.no_grad():
        for _, item in enumerate(dataloader):
            seq = item[0].to("cuda").unsqueeze(2)  # Adding channel dimension
            label = item[1].to("cuda").long()
            pred = model(seq)
            loss_t = cross_entropy(
                pred.flatten(0, 1),
                label.flatten(0, 1),
                weight=torch.tensor(ce_weights, device="cuda"),
            )
            loss_val.append(loss_t)
    return seq, label, pred, loss_val


if __name__ == "__main__":
    scripting.logged_main(
        "Training",
        main,
    )
