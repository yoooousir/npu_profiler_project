import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = None
        self.fc2 = None

    def forward(self, x):
        print(f"Input shape: {x.shape}") #Input shape: torch.Size([1, 28, 28])
        x = nn.ReLU()(self.conv1(x))
        print(f"After conv1: {x.shape}") #After conv1: torch.Size([32, 26, 26])
        x = nn.MaxPool2d(2)(x)
        print(f"After maxpool1: {x.shape}") #After maxpool1: torch.Size([32, 13, 13])
        x = nn.ReLU()(self.conv2(x))
        print(f"After conv2: {x.shape}") #After conv2: torch.Size([64, 11, 11])
        x = nn.MaxPool2d(2)(x)

        # flatten
        if not self.fc1:
            self.fc1 = nn.Linear(x.numel() // x.size(0), 128).to(x.device)

        print(f"After maxpool2: {x.shape}") #After maxpool2: torch.Size([64, 5, 5])
        x = x.view(x.size(0), -1)  # Flatten
        print(f"After view: {x.shape}") #After view: torch.Size([64, 25])

        # fc2 초기화
        if not self.fc2:
            self.fc2 = nn.Linear(128, 10).to(x.device)

        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x
