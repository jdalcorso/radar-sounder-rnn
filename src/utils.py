import os
import torch
import numpy
import random
import matplotlib.pyplot as plt
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, random_split
from model import UNetWrapper, NLUNetWrapper, UNetASPPWrapper, URNN, NLURNN, NLURNN1D
from dataset import RadargramDataset


# @torch.compile
def validation_weak(model, dataloader, seq_len, ce_weights):
    loss_val = []
    ce_weights = torch.tensor(ce_weights, device="cuda")
    model.train(False)
    with torch.no_grad():
        for _, item in enumerate(dataloader):
            seq = item[0].to("cuda").unsqueeze(2)  # BTHW -> BT1HW
            label = item[1].to("cuda").long()  # BTHW
            this_preds, hidden, cell = [], None, None
            for i in range(seq_len):
                pred, hidden, cell = model(seq[:, i], hidden, cell)  # BCHW
                this_preds.append(pred.unsqueeze(1))  # B1CHW * T
            this_preds = torch.cat(this_preds, dim=1)  # BTCHW
            loss_t = cross_entropy(
                this_preds.flatten(0, 1), label.flatten(0, 1), ce_weights
            )
            loss_val.append(loss_t)
    return seq, label, this_preds, loss_val


def get_model(model, cfg):
    """
    Input: model configuration
    Output: model Module object
    """
    match model:
        case "u":
            model = UNetWrapper(*cfg)
        case "nlu":
            model = NLUNetWrapper(*cfg)
        case "ur":
            model = URNN(*cfg)
        case "nlur":
            if cfg[3][1] != 1:
                model = NLURNN(*cfg)
            else:
                model = NLURNN1D(*cfg)
        case "aspp":
            model = UNetASPPWrapper(*cfg)
    return model


def plot_loss(loss_train, loss_val, out_dir):
    """
    Wrapper for line plotting utilities.
    """
    plt.plot(loss_train)
    plt.plot(loss_val)
    plt.legend(["loss train", "loss validation"])
    plt.savefig(out_dir + "/loss.png")
    plt.close()


def get_dataloaders(
    dataset, seq_len, patch_len, batch_size, split, first_only, data_aug, logger, seed
):
    """
    Creates a dataset with the given input configuration, then creates dataloaders
    for test and training using a random split.
    """
    match dataset:
        case "mcords3":
            data_dir = "/home/jordydalcorso/workspace/datasets/MCORDS3_Miguel/tif"
            ce_weights = [0.9267, 0.6172, 12.1344, 1.0000, 2.3384]
            n_classes = 5
        case "mcords1":
            data_dir = "/home/jordydalcorso/workspace/datasets/MCORDS1"
            ce_weights = [1.0000, 0.1320, 2.4106, 0.1823]
            n_classes = 4
        case _:
            raise ValueError("Choose dataset between mcords1 and mcords3")
    dataset = RadargramDataset(
        dataset_path=data_dir,
        seq_len=seq_len,
        patch_width=patch_len,
        stride=patch_len,
        seq_stride=seq_len,
    )

    test_ds = RadargramDataset(
        dataset_path=data_dir + "/test",
        seq_len=seq_len,
        patch_width=patch_len,
        stride=patch_len,
        seq_stride=seq_len,
    )

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, split, generator)
    train_ds.dataset.data_aug = data_aug
    train_ds.dataset.first_only = first_only
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, 1, shuffle=False)

    _, _, patch_h, _ = next(iter(train_dl))[0].shape
    logger.info("Number of sequences TRAIN: {}".format(len(train_ds)))
    logger.info("Number of sequences VAL: {}".format(len(val_ds)))
    logger.info("Number of sequences TEST : {}".format(len(test_ds)))
    logger.info(
        "Shape of dataloader items: {}\n".format(list(next(iter(train_dl))[0].shape))
    )
    assert n_classes == len(ce_weights), "Mismatch between n_classes and ce_weights"
    return train_dl, val_dl, test_dl, patch_h, ce_weights, n_classes


def plot_results(seq, label, pred, name):
    """
    Utility for plotting qualitative results.
    """
    # seq, label, pred has to be THW (seq TCHW -> THW)
    seq = seq[:, 0] if len(seq.shape) > 3 else seq
    T = seq.shape[0]
    if seq.shape[-1] != 1:
        _, axes = plt.subplots(3, T, figsize=(20, 20))
        for i in range(T):
            axes[0, i].imshow(seq[i].cpu().numpy(), cmap="gray", aspect="auto")
            axes[0, i].axis("off")
            axes[1, i].imshow(
                label[i].cpu().numpy(), aspect="auto", interpolation="nearest"
            )
            axes[1, i].axis("off")
            axes[2, i].imshow(
                pred[i].cpu().numpy(), aspect="auto", interpolation="nearest"
            )
            axes[2, i].axis("off")
    else:
        _, axes = plt.subplots(3, 1, figsize=(20, 20))
        S = torch.zeros((seq.shape[1], seq.shape[0]), device="cuda")
        L = torch.zeros((seq.shape[1], seq.shape[0]), device="cuda")
        P = torch.zeros((seq.shape[1], seq.shape[0]), device="cuda")
        for i in range(T):
            S[:, i] = seq[i].squeeze()
            L[:, i] = label[i].squeeze()
            P[:, i] = pred[i].squeeze()
        axes[0].imshow(S.cpu().numpy(), cmap="gray", aspect="auto")
        axes[0].axis("off")
        axes[1].imshow(L.cpu().numpy(), aspect="auto", interpolation="nearest")
        axes[1].axis("off")
        axes[2].imshow(P.cpu().numpy(), aspect="auto", interpolation="nearest")
        axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    numpy.random.seed(seed)
    random.seed(seed)


def save_latest(model, out_dir, loss_val_tot):
    """
    Saves the model checkpoint in the out_dir folder and names it latest_xxx.pt
    where xxx is the last validation loss (only decimals).
    """
    try:
        loss_val = round(loss_val_tot[-1].item(), 3)
        loss_val = str(loss_val).split(".")[0] + str(loss_val).split(".")[1]
        while len(loss_val) <= 4:
            loss_val = loss_val + "0"
        torch.save(model.state_dict(), out_dir + "/epoch_" + loss_val + ".pt")
    except:
        torch.save(model.state_dict(), out_dir + "/epoch_11111.pt")


def load_best(model, out_dir):
    """
    Loads the model checkpoint with the lowest validation loss in the out_dir folder.
    """
    files = os.listdir(out_dir)
    files = [f for f in files if f.startswith("epoch")]
    losses = [int(f.split("_")[-1].split(".")[0]) for f in files]
    best = files[losses.index(min(losses))]
    model.load_state_dict(torch.load(out_dir + "/" + best))
    return model, best
