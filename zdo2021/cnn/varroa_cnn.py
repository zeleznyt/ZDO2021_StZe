import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 128, 3)
        self.fc1 = nn.Linear(128 * 6 * 6, 1024)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.drop2 = nn.Dropout(p=0.25)
        self.fc4 = nn.Linear(64, 2)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1) 

        x = F.relu(self.fc1(x))
        x = self.drop(x)

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.drop2(x)

        x = self.fc4(x)
        return x