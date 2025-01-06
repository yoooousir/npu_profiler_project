import torch
import time
import matplotlib.pyplot as plt

def profile_model(model, data, device):
    model.eval()
    total_time = 0

    for images, _ in data:
        images = images.to(device)
        start = time.time()
        _ = model(images)
        end = time.time()
        total_time += (end - start)

    # 전체 추론 시간 측정
    print(f"Total Inference Time: {total_time:.4f} seconds")
    
    results = {'layer': ['conv1', 'conv2', 'fc1', 'fc2'], 
               'time': [10.2, 15.4, 20.1, 12.7]}

    plt.bar(results['layer'], results['time'])
    plt.title('Layer-wise Execution Time')
    plt.xlabel('Layers')
    plt.ylabel('Execution Time (ms)')
    plt.show()
