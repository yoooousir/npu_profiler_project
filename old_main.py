import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import get_model
from profiler import profile_model

# -----------------------------
# 설정
# -----------------------------
MODELS = ["simplecnn", "resnet18", "mobilenet_v2"]
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
DEVICES.append("npu")  # 시뮬레이션

BATCH_LIMIT = 10  # 테스트용 배치 수 제한
BATCH_SIZE = 32   # DataLoader 배치 사이즈

# -----------------------------
# 데이터 로드 및 DataLoader
# -----------------------------
# -----------------------------
# SimpleCNN용
# -----------------------------
transform_simplecnn = transforms.Compose([
    transforms.ToTensor()
])
train_dataset_simplecnn = datasets.MNIST(root='./data', train=True, download=True,
                                         transform=transform_simplecnn)
train_loader_simplecnn = DataLoader(train_dataset_simplecnn, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# ResNet/MobileNet용
# -----------------------------
transform_torchvision = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])
train_dataset_torchvision = datasets.MNIST(root='./data', train=True, download=True,
                                           transform=transform_torchvision)
train_loader_torchvision = DataLoader(train_dataset_torchvision, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# 멀티 모델 × 멀티 디바이스 프로파일링
# -----------------------------
for model_name in MODELS:
    for dev_name in DEVICES:
        if dev_name == "cuda" and not torch.cuda.is_available():
            continue
        device = torch.device("cuda" if dev_name=="cuda" else "cpu")
        print(f"\nProfiling {model_name} on {device}...")

        model = get_model(model_name).to(device)

        # 모델별 적절한 DataLoader 선택
        if model_name == "simplecnn":
            dataloader = train_loader_simplecnn
        else:
            dataloader = train_loader_torchvision

        profile_model(model, dataloader, device, max_batches=BATCH_LIMIT)
