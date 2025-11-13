import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 256), nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, x1, x2=None, x3=None):
        if x2 is not None and x3 is None:  # pair
            return self.forward_once(x1), self.forward_once(x2)
        elif x2 is not None and x3 is not None:  # triplet
            return self.forward_once(x1), self.forward_once(x2), self.forward_once(x3)
        else:
            return self.forward_once(x1)
