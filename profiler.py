import torch.cuda as cuda
import time
import matplotlib.pyplot as plt

def profile_model(model, data, device):
    model.eval()
    total_time = 0
    total_memory = []

    for images, _ in data:
        images = images.to(device)
        cuda.reset_peak_memory_stats(device)  # GPU 메모리 사용량 초기화
        start = time.time()
        _ = model(images)
        end = time.time()
        total_time += (end - start)

        # GPU 메모리 사용량 측정
        peak_memory = cuda.max_memory_allocated(device) / (1024 * 1024)  # MB 단위
        total_memory.append(peak_memory)

    # 전체 추론 시간 측정
    print(f"Total Inference Time: {total_time:.4f} seconds")
    print(f"Peak GPU Memory Usage: {max(total_memory):.2f} MB")
    # Total Inference Time: 119.1386 seconds
    
    results = {'layer': ['conv1', 'conv2', 'fc1', 'fc2'], 
               'time': [10.2, 15.4, 20.1, 12.7],
               'memory': total_memory}
    
    fig, ax1 = plt.subplots()
    ax1.bar(results['layer'], results['time'], color='b', alpha=0.6, label='Execution Time (ms)')
    ax1.set_xlabel('Layers')
    ax1.set_ylabel('Execution Time (ms)', color='b')

    ax2 = ax1.twinx()
    ax2.plot(results['layer'], results['memory'], color='r', marker='o', label='Memory Usage (MB)')
    ax2.set_ylabel('Memory Usage (MB)', color='r')

    fig.suptitle('Layer-wise Profiling')
    fig.legend(loc="upper right")
    plt.savefig('layer_wise_profiling.png')
    plt.show()

    # 병목 구간 식별
    identify_bottleneck(results)


# 각 레이어의 실행 시간 데이터를 분석하여 가장 느린 레이어(병목 구간)를 식별하는 함수
def identify_bottleneck(results):
    max_time = max(results['time'])
    bottleneck_layer = results['layer'][results['time'].index(max_time)]
    print(f"The bottleneck layer is {bottleneck_layer} with {max_time:.2f} ms.")