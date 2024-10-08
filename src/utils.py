import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import Model, UNetWrapper, URNN


def get_model(model, cfg):
    match model:
        case "base":
            model = Model(*cfg)
        case "unet":
            model = UNetWrapper(*cfg)
        case "urnet":
            model = URNN(*cfg)
    return model


def plot_loss(loss_train, loss_val, out_dir):
    plt.plot(loss_train)
    plt.plot(loss_val)
    plt.legend(["loss train", "loss validation"])
    plt.savefig(out_dir + "/loss.png")
    plt.close()


def get_dataloaders(data_dir, label_dir, seq_len, patch_len, batch_size, test_size):
    data = torch.load(data_dir).to("cuda")
    data = data.unfold(1, patch_len, patch_len).unfold(1, seq_len, seq_len)
    data = torch.permute(data, [1, 3, 0, 2])  # NTHW

    labels = torch.load(label_dir).to("cuda")
    labels = labels.unfold(1, patch_len, patch_len).unfold(1, seq_len, seq_len)
    labels = torch.permute(labels, [1, 3, 0, 2])  # NTHW

    dataset = TensorDataset(data, labels)
    train_ds, test_ds = random_split(dataset, (1 - test_size, test_size))

    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False)
    return train_dl, test_dl


def plot_results(seq, label, pred, name):
    # seq, label, pred has to be THW (seq TCHW -> THW)
    seq = seq[:, 0] if len(seq.shape) > 3 else seq
    T = seq.shape[0]
    fig, axes = plt.subplots(3, T, figsize=(20, 20))
    for i in range(T):
        axes[0, i].imshow(seq[i].cpu().numpy(), cmap="gray", aspect="auto")
        axes[0, i].axis("off")
        axes[1, i].imshow(
            label[i].cpu().numpy(), aspect="auto", interpolation="nearest"
        )
        axes[1, i].axis("off")
        axes[2, i].imshow(pred[i].cpu().numpy(), aspect="auto", interpolation="nearest")
        axes[2, i].axis("off")
    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def pos_encode(seq):
    # Set to 1 all pixels above max (in H)
    B, T, C, H, W = seq.shape
    m = torch.argmax(seq, dim=3)
    m_expanded = m.unsqueeze(3).expand(-1, -1, -1, H, -1)
    range_H = torch.arange(H).reshape(1, 1, 1, H, 1).to("cuda")
    pos = (
        (torch.clone(range_H) / torch.arange(H).sum())
        .repeat([B, T, C, 1, W])
        .flip(dims=(3,))
    )
    mask = range_H <= m_expanded
    pos[mask] = 1.0
    seq = torch.cat([seq, pos], dim=2)
    return seq
