import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from data.dataset import SiameseMNIST
from models.siamese import SiameseNetwork
from losses.contrastive import ContrastiveLoss
from losses.triplet import TripletLoss
from training.train import train
from utils.plot import plot_losses

def main():
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # Contrastive setup
    dataset_pair = SiameseMNIST(mnist_train, mode="pair")
    loader_pair = DataLoader(dataset_pair, batch_size=64, shuffle=True)
    model_pair = SiameseNetwork().cuda()
    optimizer_pair = optim.Adam(model_pair.parameters(), lr=0.001)
    contrastive_losses = train(model_pair, loader_pair, ContrastiveLoss(), optimizer_pair, epochs=5, mode="pair")

    # Triplet setup
    dataset_triplet = SiameseMNIST(mnist_train, mode="triplet")
    loader_triplet = DataLoader(dataset_triplet, batch_size=64, shuffle=True)
    model_triplet = SiameseNetwork().cuda()
    optimizer_triplet = optim.Adam(model_triplet.parameters(), lr=0.001)
    triplet_losses = train(model_triplet, loader_triplet, TripletLoss(), optimizer_triplet, epochs=5, mode="triplet")

    # Plot comparison
    plot_losses({"Contrastive": contrastive_losses, "Triplet": triplet_losses})

if __name__ == "__main__":
    main()
