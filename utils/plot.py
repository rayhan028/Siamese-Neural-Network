import matplotlib.pyplot as plt

def plot_losses(loss_dict):
    for name, losses in loss_dict.items():
        plt.plot(losses, label=name)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Siamese Loss Function Comparison")
    plt.show()
