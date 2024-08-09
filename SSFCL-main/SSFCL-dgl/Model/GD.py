"""
Time: 2024.1.6
Author: Yiran Shi
"""

import torch.nn as nn

class discriminator(nn.Module):
    def __init__(self, len):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(len, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 16),
            nn.ReLU(True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        result = self.dis(data)
        return result

class generator(nn.Module):
    def __init__(self, len):
        self.len = len
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(self.len, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, len),
            nn.Sigmoid()
        )

    def forward(self, noise):
        result = self.gen(noise)
        return result