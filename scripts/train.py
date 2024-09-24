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
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import Model
from utils import plot_loss


def main(
    hidden_size,
    patch_len,
    seq_len,
    test_size,
    epochs,
    batch_size,
    lr,
    data_dir,
    label_dir,
    out_dir,
    **kwargs,
):
    logger = logging.getLogger("train")

    # Model
    model = Model()
    num_devices = device_count()
    if num_devices >= 2:
        model = DataParallel(model)
    model = model.to("cuda")

    # Dataset
    data = torch.load(data_dir).to("cuda")
    data = data.unfold(1, patch_len, patch_len).unfold(1, seq_len, seq_len)
    data = torch.permute(data, [1,3,0,2]) # NTHW
    labels = torch.load(label_dir).to("cuda")
    labels = labels.unfold(1, patch_len, patch_len).unfold(1, seq_len, seq_len)
    labels = torch.permute(labels, [1,3,0,2]) # NTHW
    dataset = TensorDataset(data, labels)
    train_ds, test_ds = random_split(dataset, (1 - test_size, test_size))
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False)
    logger.info("Number of sequences TRAIN: {}".format(len(train_ds)))
    logger.info("Number of sequences TEST : {}".format(len(test_ds)))
    logger.info("Shape of items: {}".format(list(train_ds[0][0].shape)))

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
            label = item[1].to("cuda")
            # pred = model(seq)
            # loss = cross_entropy(pred.transpose(1, 2), label)
            # loss_train.append(loss)
            # # Optimize
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

    #     # Validation
    #     loss_val = []
    #     model.train(False)
    #     for _, item in enumerate(test_dl):
    #         seq = item[0].to("cuda").unsqueeze(2)  # Adding channel dimension
    #         label = item[1].to("cuda")
    #         pred = model(seq)
    #         loss_t = cross_entropy(pred.transpose(1, 2),label)
    #         loss_val.append(loss_t)

    #     loss_train, loss_val= torch.tensor(loss_train).mean(), torch.tensor(loss_val).mean()
    #     loss_train_tot.append(loss_train)
    #     loss_val_tot.append(loss_val)
    #     logger.info(
    #         "Epoch:",
    #         epoch,
    #         "Loss train:",
    #         loss_train.item(),
    #         "Loss val:",
    #         loss_val.item(),
    #         "Time:",
    #         time.time() - t0,
    #     )

    # plot_loss(loss_train_tot, loss_val_tot, out_dir)
    # # torch.save(model.state_dict(), out_dir + "/latest.pt")




if __name__ == "__main__":
    scripting.logged_main(
        "Training",
        main,
    )
