import torch
import torch.nn as nn
import torch.nn.functional as F

class PixSafeCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(PixSafeCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 256x256 -> 256x256
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # -> 128x128

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # -> 64x64

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 64x64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # -> 32x32

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# 32x32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # -> 16x16

            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),# 16x16
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                  # -> 1x1
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),                                # (B, 512, 1, 1) → (B, 512)
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)                  # Raw logits (no softmax)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
