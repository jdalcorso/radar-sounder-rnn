import matplotlib.pyplot as plt

def plot_loss(loss_train, loss_val, out_dir):
    plt.plot(loss_train)
    plt.plot(loss_val)
    plt.legend(["loss train", "loss validation"])
    plt.savefig(out_dir + "/loss.png")
    plt.close()