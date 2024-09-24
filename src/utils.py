import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split

def plot_loss(loss_train, loss_val, out_dir):
    plt.plot(loss_train)
    plt.plot(loss_val)
    plt.legend(["loss train", "loss validation"])
    plt.savefig(out_dir + "/loss.png")
    plt.close()

def get_dataloaders(data_dir, label_dir, seq_len, patch_len, batch_size, test_size):
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
    return train_dl, test_dl