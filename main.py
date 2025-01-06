from model import SimpleCNN
from profiler import profile_model
import torch
from torchvision import datasets, transforms

# MNIST 데이터셋을 PyTorch로 로드
train_data = datasets.MNIST(root='./data', train=True, download=True,
                            transform=transforms.ToTensor())

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# 모델 성능 프로파일링
profile_model(model, train_data, device)
