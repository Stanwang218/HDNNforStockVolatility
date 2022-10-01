import torch
from torch import nn


class mymse(nn.Module):
    def __init__(self):
        super(mymse, self).__init__()

    def forward(self, x, y):
        res = x - y
        return (res ** 2).sum()


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.block1 = Block(3, 32)
        self.block2 = Block(32, 64)
        self.block3 = Block(64, 128)

        self.fc = nn.Linear(128 * 10 * 10, 200)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.view(-1, 128 * 10 * 10)
        x = self.fc(x)

        return x


class hdnn(nn.Module):
    def __init__(self):
        super(hdnn, self).__init__()
        self.cnn = CNN()
        self.sequence = nn.Sequential(
            nn.Linear(280,128),
            nn.Linear(128,128),
            nn.Linear(128,1),
        )

    def forward(self, x, r, rv):
        x = self.cnn(x)
        x = torch.cat([x, r, rv], dim=1)
        x = self.sequence(x)
        return x

