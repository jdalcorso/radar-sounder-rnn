import torch
import numpy
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from model import UNetWrapper, NLUNetWrapper, URNN, NLURNN
from aspp import UNetASPPWrapper
from dataset import RadargramDataset


def get_model(model, cfg):
    match model:
        case "u":
            model = UNetWrapper(*cfg)
        case "nlu":
            model = NLUNetWrapper(*cfg)
        case "ur":
            model = URNN(*cfg)
        case "nlur":
            model = NLURNN(*cfg)
        case "aspp":
            model = UNetASPPWrapper(*cfg)
    return model


def plot_loss(loss_train, loss_val, out_dir):
    plt.plot(loss_train)
    plt.plot(loss_val)
    plt.legend(["loss train", "loss validation"])
    plt.savefig(out_dir + "/loss.png")
    plt.close()


def get_dataloaders(data_dir, seq_len, patch_len, batch_size, test_size, seed):
    dataset = RadargramDataset(
        dataset_path=data_dir,
        seq_len=seq_len,
        patch_width=patch_len,
        stride=patch_len,
        seq_stride=seq_len,
    )

    generator = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(dataset, (1 - test_size, test_size), generator)
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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    numpy.random.seed(seed)
    random.seed(seed)


def get_hooks(model_name, model, hook_fn):
    hooks = []
    assert isinstance(model, torch.nn.DataParallel)
    match model_name:
        case "u" | "nlu":
            for layer in model.module.unet.encoder.children():
                hooks.append(layer.register_forward_hook(hook_fn))
            for layer in model.module.unet.decoder.children():
                hooks.append(layer.register_forward_hook(hook_fn))
            hooks.pop()
        case "ur" | "nlur":
            for layer in model.module.encoder.children():
                hooks.append(layer.register_forward_hook(hook_fn))
            for layer in model.module.decoder.children():
                hooks.append(layer.register_forward_hook(hook_fn))
        case "aspp":
            for layer in model.module.unet.children():
                hooks.append(layer.register_forward_hook(hook_fn))
    return hooks


def show_feature_maps(maps, out_dir):
    """
    Maps should be a list of CHW images.
    This algorithm does PCA to transform each image into a 3HW,
    min-max normalize and shows it in a subplot
    """
    fig, axes = plt.subplots(1, len(maps), figsize=(30, 10))
    axes = axes.flatten()
    for i in range(len(axes)):
        C, H, W = maps[i].shape
        map = maps[i].permute(1, 2, 0).flatten(0, 1)  # (HW)C
        U, S, _ = torch.pca_lowrank(map, 3)
        img = U[:, :3] @ torch.diag(S)  # (HW)3
        img = img.view(H, W, 3)  # HW3
        mi, ma = img.min(), img.max()
        img = (img - mi) / (ma - mi)
        axes[i].imshow(img.numpy(), aspect="auto", interpolation="nearest")
        axes[i].axis("off")
    plt.savefig(out_dir + "/maps.png")
    plt.close()
