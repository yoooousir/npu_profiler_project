import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, mobilenet_v2
import time

# -----------------------------
# SimpleCNN 정의
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, profile=False):
        timings = {}
        t0 = time.time()
        x = F.relu(self.conv1(x))
        t1 = time.time()
        timings["conv1"] = (t1 - t0) * 1000

        x = F.relu(self.conv2(x))
        t2 = time.time()
        timings["conv2"] = (t2 - t1) * 1000

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)

        # 동적 FC
        if x.size(1) != self.fc1.in_features:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)
        t3 = time.time()
        x = F.relu(self.fc1(x))
        t4 = time.time()
        timings["fc1"] = (t4 - t3) * 1000

        x = self.dropout2(x)
        t5 = time.time()
        x = self.fc2(x)
        t6 = time.time()
        timings["fc2"] = (t6 - t5) * 1000

        if profile:
            return x, timings
        return x


# -----------------------------
# 모델 선택 함수
# -----------------------------
def get_model(name: str):
    name = name.lower()
    if name == "simplecnn":
        return SimpleCNN()
    elif name == "resnet18":
        return resnet18(num_classes=10)
    elif name == "mobilenet_v2":
        return mobilenet_v2(num_classes=10)
    else:
        raise ValueError(f"Unknown model: {name}")
