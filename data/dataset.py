import torch
from torch.utils.data import Dataset
import random

class SiameseMNIST(Dataset):
    def __init__(self, mnist_dataset, mode="pair"):
        self.mnist = mnist_dataset
        self.data = mnist_dataset.data
        self.targets = mnist_dataset.targets
        self.mode = mode

    def __getitem__(self, index):
        img1, label1 = self.data[index], self.targets[index]

        if self.mode == "pair":
            should_get_same_class = random.randint(0, 1)
            if should_get_same_class:
                idx2 = random.choice((self.targets == label1).nonzero().flatten())
            else:
                idx2 = random.choice((self.targets != label1).nonzero().flatten())
            img2, label2 = self.data[idx2], self.targets[idx2]
            img1 = img1.unsqueeze(0).float() / 255.0
            img2 = img2.unsqueeze(0).float() / 255.0
            return img1, img2, torch.tensor(int(label1 == label2), dtype=torch.float)

        elif self.mode == "triplet":
            pos_idx = random.choice((self.targets == label1).nonzero().flatten())
            neg_idx = random.choice((self.targets != label1).nonzero().flatten())
            img_pos, img_neg = self.data[pos_idx], self.data[neg_idx]
            img1 = img1.unsqueeze(0).float() / 255.0
            img_pos = img_pos.unsqueeze(0).float() / 255.0
            img_neg = img_neg.unsqueeze(0).float() / 255.0
            return img1, img_pos, img_neg

    def __len__(self):
        return len(self.mnist)
