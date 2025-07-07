import torch
from torch import nn
import torch.nn.functional as F

class OcularLMGenerator(nn.Module):
    def __init__(self):
        super(OcularLMGenerator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 64 * 64, 500)
        self.fc2 = nn.Linear(500, 66)  # Output the maximum number of landmarks

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = OcularLMGenerator()
    print(model)